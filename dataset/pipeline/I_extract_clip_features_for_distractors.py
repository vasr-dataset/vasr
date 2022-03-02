import argparse
import json
import os
import traceback
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm

from dataset.config import imsitu_path, SPLIT, split_to_files, data_path, swig_images_path, columns_to_serialize, \
    swig_path

imsitu_space_path = os.path.join(imsitu_path, f'imsitu_space.json')
imsitu_space = json.loads(open(imsitu_space_path).read())
verbs = imsitu_space["verbs"]
nouns = imsitu_space["nouns"]
data_split = json.load(open(os.path.join(swig_path, f"{SPLIT}.json")))

template_replace_counter = {'complete': 0, 'incomplete': 0}

class CLIPDistractorsFilter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = model
        self.preprocess = preprocess

    def get_clip_img(self, img_name):
        img_path = os.path.join(swig_images_path, img_name)
        sol_img = Image.open(img_path)
        sol_img_data = {'tensor': self.preprocess(sol_img).unsqueeze(0).to(self.device)}
        return sol_img_data

    def calculate_AB_similar_feats(self, A_data, B_data):
        A_clip_sents = list(set([self.get_clip_text(x, A_data['verb']) for x in A_data['frames']]))
        B_clip_sents = list(set([self.get_clip_text(x, B_data['verb']) for x in B_data['frames']]))
        AB_classes = self.tokenize_sents(A_clip_sents, B_clip_sents)

        A_img_data = self.get_clip_img(A_data['img'])
        B_img_data = self.get_clip_img(B_data['img'])

        A_indices = np.array(list(range(len(A_clip_sents))))
        B_indices = np.array(list(range(len(A_clip_sents), len(A_clip_sents) + len(B_clip_sents))))
        A_img_AB_probs, A_logits_per_class = self.get_max_probs(AB_classes, A_indices, B_indices, A_img_data['tensor'])
        B_img_AB_probs, B_logits_per_class = self.get_max_probs(AB_classes, A_indices, B_indices, B_img_data['tensor'])

        all_clip_data = {'A_clip_sents': A_clip_sents, 'B_clip_sents': B_clip_sents, 'A_img_AB_probs': A_img_AB_probs, 'B_img_AB_probs': B_img_AB_probs, 'A_logits_per_class': A_logits_per_class, 'B_logits_per_class': B_logits_per_class}
        return all_clip_data

    def get_max_probs(self, AB_classes, A_indices, B_indices, X_tensor):
        X_img_logits_per_AB_class, _ = self.model(X_tensor, AB_classes)
        X_logits_per_class = X_img_logits_per_AB_class.cpu().detach().numpy()[0]
        # X_img_A_frames_logits_mean = X_logits_per_class[A_indices].mean()
        # X_img_B_frames_logits_mean = X_logits_per_class[B_indices].mean()
        # X_img_AB_probs = [float(x) for x in softmax([X_img_A_frames_logits_mean, X_img_B_frames_logits_mean])]
        X_img_A_frames_logits_max = X_logits_per_class[A_indices].max()
        X_img_B_frames_logits_max = X_logits_per_class[B_indices].max()
        X_img_AB_probs = [float(x) for x in softmax([X_img_A_frames_logits_max, X_img_B_frames_logits_max])]
        return X_img_AB_probs, [float(x) for x in X_logits_per_class]

    def get_clip_text(self, frame, verb):
        verb_framenet = verbs[verb]
        sent_template = verb_framenet['abstract']
        roles_identical, sent_template_subst = self.get_templative_sent_from_framenet_abstract(frame, sent_template,
                                                                                               verb_framenet)
        global template_replace_counter
        if roles_identical:
            template_replace_counter['complete'] += 1
        else:
            template_replace_counter['incomplete'] += 1

        return sent_template_subst

    def get_templative_sent_from_framenet_abstract(self, frame, sent_template, verb_framenet):
        sent_template_subst = sent_template.lower()
        target_roles = set(verb_framenet['roles'].keys())
        existing_roles = set(frame.keys())
        roles_identical = target_roles == existing_roles
        for k, v in frame.items():
            if k not in target_roles:
                continue
            if v == '':
                continue
            try:
                word = nouns[v]['gloss'][0]
            except Exception as ex:
                traceback.print_exc()
            sent_template_subst = sent_template_subst.replace(k, word)
        return roles_identical, sent_template_subst

    def tokenize_sents(self, sol_clip_sents, cand_clip_sents):
        classes_texts = sol_clip_sents + cand_clip_sents
        classes_tokenized_texts = clip.tokenize(classes_texts).to(self.device)
        return classes_tokenized_texts


def main(args, split_file_name):
    df, in_path = read_df(split_file_name, args)
    # df = df.query("A_img == 'bowing_72.jpg' and diff_item_A_str_first == 'bowing' and diff_item_B_str_first == 'laughing'")

    clip_feature_extractor = CLIPDistractorsFilter()

    B_distractors_data_and_clip_features = []
    C_distractors_data_and_clip_features = []
    for idx, (r_idx, r) in tqdm(enumerate(df.iterrows()), total=len(df), desc='Extracting CLIP features'):
        B_img_r_clip_feats, C_img_r_clip_feats = extract_clip_feats(r, clip_feature_extractor)
        B_distractors_data_and_clip_features.append(B_img_r_clip_feats)
        C_distractors_data_and_clip_features.append(C_img_r_clip_feats)

    df['B_distractors_data'] = B_distractors_data_and_clip_features
    df['C_distractors_data'] = C_distractors_data_and_clip_features
    print('template_replace_counter')
    print(template_replace_counter)
    out_path = in_path.replace("distractors_before_clip_filter", "distractors_with_clip_features")
    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.dumps)
    if args.workers_cnt is not None:
        out_path = out_path.replace(".csv", f"{args.worker_idx}_out_{args.workers_cnt}.csv")
    df.to_csv(out_path)
    print(f"Dumped {len(df)} analogies of SPLIT {SPLIT} to {out_path}")


def read_df(split_file_name, args):
    split_file_name_in = get_analogies_name(split_file_name)
    in_path = os.path.join(data_path, 'split_distractors', split_file_name_in)
    print(f"Reading: {in_path}")
    df = pd.read_csv(in_path)
    for c in columns_to_serialize:
        if c in df.columns:
            if c in ['B_distractors_data', 'C_distractors_data']:
                df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                df[c] = df[c].apply(json.loads)
    df_len_before = len(df)
    df = df[~df['B_distractors_data'].isna()]
    df = df[~df['C_distractors_data'].isna()]
    print(f"Stared with {df_len_before}\nAfter taking cases with existing distractors data, achieved {len(df)} items")

    if args.workers_cnt is not None:
        assert args.worker_idx is not None
        print(f"Running worker {args.worker_idx}/{args.workers_cnt}")
        df = df.iloc[args.worker_idx::args.workers_cnt]
    print(f"df is at length: {len(df)}")

    return df, in_path


def extract_clip_feats(r, clip_feature_extractor):
    D_frames = data_split[r['D_img']]['frames']
    sol_data = {'img': r['D_img'], 'frames': D_frames, 'verb': r['D_verb']}
    B_distractors_data_with_clip_feats = get_dist_clip_feats_for_lst(clip_feature_extractor, r['B_distractors_data'], sol_data)
    C_distractors_data_with_clip_feats = get_dist_clip_feats_for_lst(clip_feature_extractor, r['C_distractors_data'], sol_data)
    return B_distractors_data_with_clip_feats, C_distractors_data_with_clip_feats


def get_dist_clip_feats_for_lst(clip_feature_extractor, dist_data_lst, sol_data):
    distractors_data_with_clip_feats = []
    for distractor_data in dist_data_lst:
        distractor_frames = data_split[distractor_data['img_name']]['frames']
        curr_distractor_data = {'img': distractor_data['img_name'], 'frames': distractor_frames,
                           'verb': distractor_data['verb']}
        # if distractor_data['img_name'] != 'pulling_325.jpg':
        #     continue
        clip_feats = clip_feature_extractor.calculate_AB_similar_feats(sol_data, curr_distractor_data)
        distractor_data['clip_features'] = clip_feats
        distractors_data_with_clip_feats.append(distractor_data)
    return distractors_data_with_clip_feats


def get_analogies_name(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors_before_clip_filter.csv')
"""
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 0 --workers_cnt 16
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 1 --workers_cnt 16
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 2 --workers_cnt 16
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 3 --workers_cnt 16
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 4 --workers_cnt 16
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 5 --workers_cnt 16
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 6 --workers_cnt 16
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 7 --workers_cnt 16
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 8 --workers_cnt 16
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 9 --workers_cnt 16
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 10 --workers_cnt 16
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 11 --workers_cnt 16
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 12 --workers_cnt 16
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 13 --workers_cnt 16
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 14 --workers_cnt 16
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/I_extract_clip_features_for_distractors.py --worker_idx 15 --workers_cnt 16

"""

if __name__ == '__main__':
    print('Important: If you ran with --indices, run "merge_train_clip_VL_feats_for_distractors_filter.py" later')
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers_cnt', required=False, help="For parallel infernece, Number of workers are running", type=int)
    parser.add_argument('--worker_idx', required=False, help="For parallel infernece, this workier index", type=int)
    args = parser.parse_args()
    print(args)

    for split_file_name in split_to_files[SPLIT]:
            # extract to the full file only
            if SPLIT == 'train' and split_file_name != 'all_ABCD_matches_rule_based_sampled_train_full.csv':
                continue
            print(f"Extracting CLIP features for {split_file_name}")
            main(args, split_file_name)

    print("Done")
