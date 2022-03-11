import argparse
import json
import os
from collections import defaultdict

import clip
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import swig_images_path
from utils.utils import SPLIT, AB_matches_filtered_visual, AB_matches_vision_and_language_feats_path, \
    AB_matches_objects_no_bbox_feats_path, \
    columns_to_serialize, swig_path

absolute_truth_path = os.path.join(swig_path, f'{SPLIT}.json')
imsitu_space_path = os.path.join(swig_path, f'imsitu_space.json')
absolute_truth = json.loads(open(absolute_truth_path).read())
imsitu_space = json.loads(open(imsitu_space_path).read())
verbs = imsitu_space["verbs"]
nouns = imsitu_space["nouns"]
from tqdm import tqdm

filter_stats = defaultdict(int)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default=False, help='to extract V&L features in parallel')
    args = parser.parse_args()
    extract_clip_sim(args.indices)


def extract_clip_sim(indices=False):
    device, df, model, preprocess = initialize_model_and_data(indices)

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
                extract_full_img_feats(A_img_path, B_img_path, all_items_without_feats_objects_no_bbox, device, model,
                                       preprocess, r)
            else:
                extract_bbox_feats(A_bbox_diff_key, A_img_path, B_bbox_diff_key, B_img_path, device, model, preprocess,
                                   r)
        else:
            r['vl_feats_bbox'] = None
        extract_full_img_feats(A_img_path, B_img_path, all_items_with_feats_objects_or_verb, device, model,
                               preprocess, r)

    print(f"from {len(df)}, {len(all_items_with_feats_objects_or_verb)} with feats, {len(all_items_without_feats_objects_no_bbox)} all_items_without_feats_objects_no_bbox")
    serialize_and_dump(all_items_with_feats_objects_or_verb, all_items_without_feats_objects_no_bbox, df, indices)
    print("Done")


def serialize_and_dump(all_items_with_feats_objects_or_verb, all_items_without_feats_objects_no_bbox, df, indices):
    all_items_objects_no_bbox_feats_df = pd.DataFrame(all_items_without_feats_objects_no_bbox)
    all_items_with_feats_objects_or_verb_df = pd.DataFrame(all_items_with_feats_objects_or_verb)
    for c in columns_to_serialize:
        if c in all_items_with_feats_objects_or_verb_df.columns:
            all_items_with_feats_objects_or_verb_df[c] = all_items_with_feats_objects_or_verb_df[c].apply(json.dumps)
        if c in all_items_objects_no_bbox_feats_df.columns:
            all_items_objects_no_bbox_feats_df[c] = all_items_objects_no_bbox_feats_df[c].apply(json.dumps)
    AB_matches_objects_or_verb_feats_path_updated = AB_matches_vision_and_language_feats_path
    AB_matches_objects_no_bbox_feats_path_updated = AB_matches_objects_no_bbox_feats_path
    if indices:
        AB_matches_objects_or_verb_feats_path_updated = AB_matches_vision_and_language_feats_path.replace(".csv",
                                                                                                          f"_indices_{indices}.csv")
        AB_matches_objects_no_bbox_feats_path_updated = AB_matches_objects_no_bbox_feats_path.replace(".csv",
                                                                                                      f"_indices_{indices}.csv")
    all_items_objects_no_bbox_feats_df.to_csv(AB_matches_objects_no_bbox_feats_path_updated)
    print(
        f"Wrote {len(all_items_objects_no_bbox_feats_df)} (from {len(df)}) to {AB_matches_objects_no_bbox_feats_path_updated}")
    all_items_with_feats_objects_or_verb_df.to_csv(AB_matches_objects_or_verb_feats_path_updated)
    print(
        f"Wrote {len(all_items_with_feats_objects_or_verb_df)} (from {len(df)}) to {AB_matches_objects_or_verb_feats_path_updated}")


def extract_bbox_feats(A_bbox_diff_key, A_img_path, B_bbox_diff_key, B_img_path, device, model, preprocess, r):
    A_img_data_bbox = get_clip_img(device, preprocess, A_img_path, bbox=A_bbox_diff_key)
    B_img_data_bbox = get_clip_img(device, preprocess, B_img_path, bbox=B_bbox_diff_key)
    vl_feats_bbox = get_feats_based_on_img(A_img_data_bbox, B_img_data_bbox, device, model, r, img_type='bbox')
    r['vl_feats_bbox'] = vl_feats_bbox


def extract_full_img_feats(A_img_path, B_img_path, all_items_without_feats_objects_no_bbox, device, model, preprocess,
                           r):
    A_img_data_full_img = get_clip_img(device, preprocess, A_img_path)
    B_img_data_full_img = get_clip_img(device, preprocess, B_img_path)
    vl_feats_full_img = get_feats_based_on_img(A_img_data_full_img, B_img_data_full_img, device, model, r,
                                               img_type='full_img')
    r['vl_feats_full_img'] = vl_feats_full_img
    all_items_without_feats_objects_no_bbox.append(r)


def initialize_model_and_data(indices):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
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
    return device, df, model, preprocess


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


if __name__ == '__main__':
    print('Important: If you ran with --indices, run "merge_train_clip_VL_feats_for_AB_filter.py" later')
    main()