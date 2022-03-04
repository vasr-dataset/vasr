import argparse
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# In order to save time if you already extracted features once, run this file with EXTRACT_FEATS = false

import clip
import cv2
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from nltk.corpus import wordnet as wn

from dataset.utils.PairFilter import PairsFilter
from dataset.config import SPLIT, AB_matches_filtered_path, imsitu_path, plots_path, \
    AB_matches_filtered_visual, AB_matches_vision_and_language_feats_path, \
    AB_matches_objects_no_bbox_feats_path, \
    AB_matches_vision_and_language_feats_to_filter, AB_matches_vision_and_language_feats_to_keep, swig_images_path, \
    columns_to_serialize, swig_path

import torch
from PIL import Image

vision_language_sim_plots_path = os.path.join(plots_path, 'vision_and_language_clip_filter_ab_similarity')
plots_filtered_path = os.path.join(vision_language_sim_plots_path, 'filtered')
plots_good_pairs_path = os.path.join(vision_language_sim_plots_path, 'good_pairs')
for d in [vision_language_sim_plots_path, plots_filtered_path, plots_good_pairs_path]:
    if not os.path.exists(d):
        os.mkdir(d)

absolute_truth_path = os.path.join(swig_path, f'{SPLIT}.json')
imsitu_space_path = os.path.join(swig_path, f'imsitu_space.json')
absolute_truth = json.loads(open(absolute_truth_path).read())
imsitu_space = json.loads(open(imsitu_space_path).read())
verbs = imsitu_space["verbs"]
nouns = imsitu_space["nouns"]
from tqdm import tqdm

SAMPLE = False
if SAMPLE:
    print(f"***** SAMPLE *****")

filter_stats = defaultdict(int)

def main():

    print(f"D EXTRACT_FEATS")
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default=False, help='to extract V&L features in parallel')
    args = parser.parse_args()
    extract_clip_sim(args.indices)


def extract_clip_sim(indices=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    if SAMPLE:
        # df = pd.read_csv(AB_matches_filtered_visual, nrows=2000)
        df = pd.read_csv(AB_matches_filtered_visual)
        # df = df.query("A_img == 'plowing_188.jpg' and B_img == 'plowing_211.jpg'")
        df = df[df['different_key'] != 'verb']
        df = df[0:100]
    else:
        df = pd.read_csv(AB_matches_filtered_visual)
    print(f'Read df at length {len(df)}')
    if indices:
        start_idx, end_idx = [int(x) for x in indices.split(",")]
        df = df.iloc[start_idx: end_idx]
        print(f"Taking indices, start: {start_idx}, end: {end_idx}, working on dataset in length {len(df)}")
    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.loads)
    df['A_bounding_box'] = df['A_data'].apply(lambda x: x['A_bounding_box'])
    df['B_bounding_box'] = df['B_data'].apply(lambda x: x['B_bounding_box'])
    all_items_with_feats_objects_or_verb = []
    all_items_without_feats_objects_no_bbox = []
    print(df['different_key'].value_counts().head(10))
    for idx, (r_idx, r) in tqdm(enumerate(df.iterrows()), desc="Extracting CLIP V&L features...", total=len(df)):
        A_img_path = os.path.join(swig_images_path, r['A_img'])
        B_img_path = os.path.join(swig_images_path, r['B_img'])
        if r['different_key'] != 'verb':
            A_bbox_diff_key = r['A_bounding_box'][r['different_key']]
            B_bbox_diff_key = r['B_bounding_box'][r['different_key']]
            if [-1, -1, -1, -1] in [A_bbox_diff_key, B_bbox_diff_key]:
                A_img_data_full_img = get_clip_img(device, preprocess, A_img_path)
                B_img_data_full_img = get_clip_img(device, preprocess, B_img_path)
                vl_feats_full_img = get_feats_based_on_img(A_img_data_full_img, B_img_data_full_img, device, model, r, img_type='full_img')
                r['vl_feats_full_img'] = vl_feats_full_img
                all_items_without_feats_objects_no_bbox.append(r)
                continue
            # continue # REMOVE ME!
            A_img_data_bbox = get_clip_img(device, preprocess, A_img_path, bbox=A_bbox_diff_key)
            B_img_data_bbox = get_clip_img(device, preprocess, B_img_path, bbox=B_bbox_diff_key)
            vl_feats_bbox = get_feats_based_on_img(A_img_data_bbox, B_img_data_bbox, device, model, r, img_type='bbox')
            r['vl_feats_bbox'] = vl_feats_bbox
        else:
            r['vl_feats_bbox'] = None
        # continue  # REMOVE ME!
        A_img_data_full_img = get_clip_img(device, preprocess, A_img_path)
        B_img_data_full_img = get_clip_img(device, preprocess, B_img_path)
        vl_feats_full_img = get_feats_based_on_img(A_img_data_full_img, B_img_data_full_img, device, model, r, img_type='full_img')
        r['vl_feats_full_img'] = vl_feats_full_img
        all_items_with_feats_objects_or_verb.append(r)

    print(f"from {len(df)}, {len(all_items_with_feats_objects_or_verb)} with feats, {len(all_items_without_feats_objects_no_bbox)} all_items_without_feats_objects_no_bbox")
    all_items_objects_no_bbox_feats_df = pd.DataFrame(all_items_without_feats_objects_no_bbox)
    all_items_with_feats_objects_or_verb_df = pd.DataFrame(all_items_with_feats_objects_or_verb)
    for c in columns_to_serialize:
        if c in all_items_with_feats_objects_or_verb_df.columns:
            all_items_with_feats_objects_or_verb_df[c] = all_items_with_feats_objects_or_verb_df[c].apply(json.dumps)
        if c in all_items_objects_no_bbox_feats_df.columns:
            all_items_objects_no_bbox_feats_df[c] = all_items_objects_no_bbox_feats_df[c].apply(json.dumps)

    AB_matches_objects_or_verb_feats_path_updated = AB_matches_vision_and_language_feats_path
    AB_matches_objects_no_bbox_feats_path_updated = AB_matches_objects_no_bbox_feats_path
    if SAMPLE:
        AB_matches_objects_or_verb_feats_path_updated = AB_matches_vision_and_language_feats_path.replace(".csv","_debug.csv")
        AB_matches_objects_no_bbox_feats_path_updated = AB_matches_objects_no_bbox_feats_path.replace(".csv", "_debug.csv")
    elif indices:
        AB_matches_objects_or_verb_feats_path_updated = AB_matches_vision_and_language_feats_path.replace(".csv",f"_indices_{indices}.csv")
        AB_matches_objects_no_bbox_feats_path_updated = AB_matches_objects_no_bbox_feats_path.replace(".csv", f"_indices_{indices}.csv")

    all_items_objects_no_bbox_feats_df.to_csv(AB_matches_objects_no_bbox_feats_path_updated)
    print(f"Wrote {len(all_items_objects_no_bbox_feats_df)} (from {len(df)}) to {AB_matches_objects_no_bbox_feats_path_updated}")

    # print("Exiting, remove me!")
    # exit()

    all_items_with_feats_objects_or_verb_df.to_csv(AB_matches_objects_or_verb_feats_path_updated)
    print(f"Wrote {len(all_items_with_feats_objects_or_verb_df)} (from {len(df)}) to {AB_matches_objects_or_verb_feats_path_updated}")

    print("Done")


def get_feats_based_on_img(A_img_data, B_img_data, device, model, r, img_type):
    A_img_synsets_strs = get_all_values_of_key(r['A_img'], r['different_key'])
    B_img_synsets_strs = get_all_values_of_key(r['B_img'], r['different_key'])

    AB_txt_data = get_ab_txt_data(img_type, r) if img_type == 'full_img' else None

    A_item, A_item_str = find_best_str_value(model, device, A_img_data['tensor'], A_img_synsets_strs, AB_txt_data['A'] if AB_txt_data else None)
    B_item, B_item_str = find_best_str_value(model, device, B_img_data['tensor'], B_img_synsets_strs, AB_txt_data['B'] if AB_txt_data else None)

    probs_A_img_logits_per_AB_class_round, probs_B_img_logits_per_AB_class_round, A_item_clip_sent, B_item_clip_sent = is_AB_Similar(model, device, A_item_str,
                                                                                                 B_item_str, A_img_data,
                                                                                                 B_img_data, AB_txt_data)
    mesh_data_AB = is_AB_similar_mesh(model, device, A_img_data['tensor'], A_img_synsets_strs, B_img_synsets_strs, AB_txt_data['A'] if AB_txt_data else None)
    mesh_data_BA = is_AB_similar_mesh(model, device, B_img_data['tensor'], B_img_synsets_strs, A_img_synsets_strs, AB_txt_data['B'] if AB_txt_data else None)
    vl_feats = {}
    vl_feats[f'A_img_strs'] = A_img_synsets_strs
    vl_feats[f'B_img_strs'] = B_img_synsets_strs
    vl_feats[f'A_item_img'] = A_item
    vl_feats[f'A_item_str'] = A_item_str
    vl_feats[f'B_item_img'] = B_item
    vl_feats[f'B_item_str'] = B_item_str
    vl_feats[f'mesh_data_AB'] = mesh_data_AB
    vl_feats[f'mesh_data_BA'] = mesh_data_BA
    vl_feats[f'probs_A_img_logits_per_AB_class_round'] = probs_A_img_logits_per_AB_class_round
    vl_feats[f'probs_B_img_logits_per_AB_class_round'] = probs_B_img_logits_per_AB_class_round
    vl_feats['A_item_clip_sent'] = A_item_clip_sent
    vl_feats['B_item_clip_sent'] = B_item_clip_sent
    return vl_feats

def get_clip_text(item, txt_data=None):
    item = item.lower()
    if txt_data:
        # changing ann[different_key['different_key']
        verb = txt_data['verb'] if txt_data['different_key'] != 'verb' else item
        verb_template = verbs[verb]['abstract']
        sent_template_subst = verb_template.lower()
        for k, v in {k:v for k,v in txt_data.items() if k not in ['verb', 'different_key', txt_data['different_key']]}.items():
            if v is None or v == '':
                continue
            word = v[0]
            sent_template_subst = sent_template_subst.replace(k, word)
        if txt_data['different_key'] != 'verb':
            sent_template_subst = sent_template_subst.replace(txt_data['different_key'], item)
        return sent_template_subst
    else:
        vowels = ['a', 'e', 'i', 'o', 'u']
        if any(item.startswith(x) for x in vowels):
            return f"A photo of an {item}"
        else:
            return f"A photo of a {item}"



def aggregate_sents_to_classses(sent1, sent2, device):
    classes_tokenized = clip.tokenize([sent1, sent2]).to(device)
    return classes_tokenized


def get_all_values_of_key(image_name, key):
    if key == 'verb':
        verb_lst = [(absolute_truth[image_name]['verb'], absolute_truth[image_name]['verb'])]
        return verb_lst
    key_value_cache = []
    key_str_values = set()
    for frame in absolute_truth[image_name]['frames']:
        if key in frame and frame[key] != '':
            key_value = frame[key]
            if key_value in key_value_cache:
                continue
            for str_value in imsitu_space['nouns'][key_value]['gloss']:
                key_str_values.add((key_value, str_value))
            key_value_cache.append(key_value)

    return list(key_str_values)


def find_best_str_value(model, device, image, item_classes, txt_data=None):
    if len(item_classes) == 1:
        return item_classes[0]
    item_classes_str = [x[1] for x in item_classes]
    item_classes_sent = list(map(lambda item_class: get_clip_text(item_class, txt_data), item_classes_str))
    A_classes_tokenized = clip.tokenize(item_classes_sent).to(device)
    A_img_logits_per_AB_class, _ = model(image, A_classes_tokenized)
    probs_A_img_logits_per_AB_class = A_img_logits_per_AB_class.softmax(dim=-1).cpu().detach().numpy()[0]
    best_item_class_index = np.argmax(probs_A_img_logits_per_AB_class)
    return item_classes[best_item_class_index]


def is_AB_Similar(model, device, A_item, B_item, A_img_data, B_img_data, AB_txt_data):
    A_item_clip_sent = get_clip_text(A_item, AB_txt_data['A'] if AB_txt_data else None)
    B_item_clip_sent = get_clip_text(B_item, AB_txt_data['B'] if AB_txt_data else None)
    AB_classes = aggregate_sents_to_classses(A_item_clip_sent, B_item_clip_sent, device)

    A_img_logits_per_AB_class, _ = model(A_img_data['tensor'], AB_classes)
    probs_A_img_logits_per_AB_class = A_img_logits_per_AB_class.softmax(dim=-1).cpu().detach().numpy()[0]
    probs_A_img_logits_per_AB_class_round = [round(float(x), 2) for x in probs_A_img_logits_per_AB_class]

    B_img_logits_per_AB_class, _ = model(B_img_data['tensor'], AB_classes)
    probs_B_img_logits_per_AB_class = B_img_logits_per_AB_class.softmax(dim=-1).cpu().detach().numpy()[0]
    probs_B_img_logits_per_AB_class_round = [round(float(x), 2) for x in probs_B_img_logits_per_AB_class]

    return probs_A_img_logits_per_AB_class_round, probs_B_img_logits_per_AB_class_round, A_item_clip_sent, B_item_clip_sent


def get_ab_txt_data(img_type, r):
    A_txt_data = {**r['A_data']['A_str'], **{'verb': r['A_verb'], 'different_key': r['different_key']}}
    B_txt_data = {**r['B_data']['B_str'], **{'verb': r['B_verb'], 'different_key': r['different_key']}}
    AB = {'A': A_txt_data, 'B': B_txt_data}
    return AB

def is_AB_similar_mesh(model, device, img, first_img_synsets_strs, second_img_synsets_strs, txt_data):
    first_img_strs = [x[1] for x in first_img_synsets_strs]
    second_img_strs = [x[1] for x in second_img_synsets_strs]
    # AB_strs = list(set(first_img_strs) - set(second_img_strs)) + second_img_strs
    AB_strs = first_img_strs + second_img_strs
    AB_strs_sent = list(map(lambda item_class: get_clip_text(item_class, txt_data), AB_strs))
    AB_classes_tokenized = clip.tokenize(AB_strs_sent).to(device)
    img_logits_per_AB_class, _ = model(img, AB_classes_tokenized)
    best_class_index = np.argmax(img_logits_per_AB_class.softmax(dim=-1).cpu().detach().numpy()[0])
    logits_first = [float(x) for x in list(img_logits_per_AB_class.cpu().detach().numpy()[0])[:len(first_img_strs)]]
    logits_second = [float(x) for x in list(img_logits_per_AB_class.cpu().detach().numpy()[0])[len(first_img_strs):]]
    most_suitable_logit_first = max(logits_first)
    most_suitable_logit_second = max(logits_second)
    mesh_data = {'AB_strs': AB_strs, 'AB_strs_sent': AB_strs_sent, 'best_class_index': int(best_class_index),
                 'first_img_strs': first_img_strs, 'second_img_strs': second_img_strs,
                 'best_label': AB_strs[best_class_index], 'logits_first': logits_first, 'logits_second': logits_second,
                 'most_suitable_logit_first': most_suitable_logit_first, 'most_suitable_logit_second': most_suitable_logit_second}
    if AB_strs[best_class_index] in second_img_strs:
        mesh_data['filter'] = True
    else:
        mesh_data['filter'] = False
    return mesh_data


def get_clip_img(device, preprocess, file_path, bbox=None):
    img = Image.open(file_path)
    if bbox:
        x1, y1, x2, y2 = bbox
        img_cropped = img.crop((x1, y1, x2, y2))
        full_img = np.asarray(img)
        full_img_bbox = cv2.rectangle(full_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        img_data = {'full_img_bbox': full_img_bbox, 'cropped_img': np.asarray(img_cropped),
                    'tensor': preprocess(img_cropped).unsqueeze(0).to(device)}
    else:
        img_data = {'tensor': preprocess(img).unsqueeze(0).to(device)}
    return img_data


def list2str(lst):
    if len(lst) < 5:
        return lst
    elif 5 <= len(lst) <= 8:
        m_idx = int(len(lst) / 2)
        first_half = lst[:m_idx]
        second_half = lst[m_idx:]
        return f"{first_half}..\n..{second_half}"
    else:
        third_idx = int(len(lst) / 3)
        first_third = lst[:third_idx]
        second_third = lst[third_idx:third_idx*2]
        third_third = lst[third_idx*2:]
        return f"{first_third}..\n..{second_third}..\n..{third_third}"


'''
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 0,100000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 100000,200000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 200000,300000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 300000,400000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 400000,500000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 500000,600000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 600000,700000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 700000,800000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 800000,900000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 900000,1000000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 1000000,1100000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 1100000,1200000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 1200000,1300000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 1300000,1400000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 1400000,1500000
CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_extract_clip_features.py --indices 1500000,1650000


CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_extract_clip_features.py --indices 0,20
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 0,50000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 50000,100000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_extract_clip_features.py --indices 100000,150000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 150000,200000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 200000,250000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_extract_clip_features.py --indices 300000,350000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 250000,300000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 350000,400000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_extract_clip_features.py --indices 400000,450000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 450000,500000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 500000,550000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_extract_clip_features.py --indices 550000,600000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 600000,650000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 650000,700000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_extract_clip_features.py --indices 700000,750000
CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_extract_clip_features.py --indices 750000,850000
'''

if __name__ == '__main__':
    print('Important: If you ran with --indices, run "merge_train_clip_VL_feats_for_AB_filter.py" later')
    main()