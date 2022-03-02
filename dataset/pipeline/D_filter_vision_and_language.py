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
EXTRACT_FEATS = True
if SAMPLE:
    print(f"***** SAMPLE *****")

filter_stats = defaultdict(int)

def main():

    print(f"EXTRACT_FEATS: {EXTRACT_FEATS}")
    if EXTRACT_FEATS:
        parser = argparse.ArgumentParser()
        parser.add_argument('--indices', default=False, help='to extract V&L features in parallel')
        args = parser.parse_args()
        extract_clip_sim(args.indices)
        classify_df_given_feats()
    else:
        classify_df_given_feats()

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


def classify_df_given_feats():
    df_verb_objects_feats, df_objects_no_bbox_feats, initial_df = get_all_dataframes()
    df_with_feats = pd.concat([df_verb_objects_feats, df_objects_no_bbox_feats])
    # df_with_feats = df_with_feats.query(f'diff_item_A_str_first == "vegetable" and diff_item_B_str_first == "meat" and A_img == "frying_43.jpg" and B_img == "frying_237.jpg"')
    print(f'Started with {len(initial_df)}, of which {len(df_with_feats)} has V&L feats, and {len(df_objects_no_bbox_feats)} dont')
    all_items_to_filter = []
    all_items_to_keep = []
    all_items_with_same_clip_keys = []
    pairs_filter = PairsFilter()
    modulo = 10000 if SPLIT != 'train' else 100000
    for idx, (r_idx, r) in tqdm(enumerate(df_with_feats.iterrows()), desc='Classifying...', total=len(df_with_feats)):
        if type(r['vl_feats_bbox']) != dict and np.isnan(r['vl_feats_bbox']):
            r['vl_feats_bbox'] = None
        if r['different_key'] == 'verb' or r['vl_feats_bbox'] is None:
            should_remove_bbox, filter_reason_bbox = False, None
        else:
            should_remove_bbox, filter_reason_bbox = classify_given_feats(r, pairs_filter, feats_type='vl_feats_bbox')
        if not should_remove_bbox:
            should_remove_full_img, filter_reason_img = classify_given_feats(r, pairs_filter, feats_type='vl_feats_full_img')
        else:
            should_remove_full_img, filter_reason_img = False, None
        should_remove = should_remove_bbox or should_remove_full_img
        r['should_remove_bbox'] = should_remove_bbox
        r['should_remove_full_img'] = should_remove_full_img
        r['filter_reason_bbox'] = filter_reason_bbox
        r['filter_reason_img'] = filter_reason_img

        update_clip_chosen_items(r, img_name='A')
        update_clip_chosen_items(r, img_name='B')

        if r[f'diff_item_A_str_first'] == r[f'diff_item_B_str_first']:
            all_items_with_same_clip_keys.append(r)
            should_remove = True

        # if idx % 1 == 0:
        #     additional_txt = get_title_text(filter_reason_bbox, filter_reason_img, should_remove, should_remove_bbox)
        #     visualize_pair_clip_all_feats(r, additional_txt, all_items_to_filter, all_items_to_keep)
        if idx == 1000 or (0 < idx and idx % modulo == 0):
            print_stats(all_items_to_filter, all_items_to_keep, idx)
        if should_remove:
            all_items_to_filter.append(r)
        else:
            all_items_to_keep.append(r)

    print_stats(all_items_to_filter, all_items_to_keep, idx)
    print("filter_stats")
    print(filter_stats)

    all_items_to_filter_df = pd.DataFrame(all_items_to_filter)
    print(f"Final. From {len(df_with_feats)} V&L feats, filter: {len(all_items_to_filter_df)} "
          f"({round(len(all_items_to_filter_df) / len(df_with_feats) * 100, 1)}%), "
          f"keeping: {len(all_items_to_keep)}")
    print(f"Writing filtered items to: {AB_matches_vision_and_language_feats_to_filter}, and items to keep to {AB_matches_vision_and_language_feats_to_keep}")
    all_items_to_filter_df.to_csv(AB_matches_vision_and_language_feats_to_filter)
    all_items_to_keep_df = pd.DataFrame(all_items_to_keep)
    all_items_to_keep_df.to_csv(AB_matches_vision_and_language_feats_to_keep)
    # if SAMPLE:
    #     df_objects_no_bbox_feats = df_objects_no_bbox_feats.sample(len(all_items_to_keep_df) * 10)

    AB_matches_filtered_path_final = AB_matches_filtered_path if SAMPLE == False else AB_matches_filtered_path.replace(".csv", "_debug.csv")
    print(f"In addition, {len(df_objects_no_bbox_feats)} dont have feats. Merging with the items to keep: {len(all_items_to_keep_df)}\n"
          f"and writing to the final path: {AB_matches_filtered_path_final}")
    # df_filtered = pd.concat([all_items_to_keep_df, df_objects_no_bbox_feats])
    df_filtered = all_items_to_keep_df
    print(f"Finally writing {len(df_filtered)} to {AB_matches_filtered_path_final}")
    for c in columns_to_serialize:
        if c in df_filtered.columns:
            df_filtered[c] = df_filtered[c].apply(json.dumps)
    df_filtered.to_csv(AB_matches_filtered_path_final)


def update_clip_chosen_items(r, img_name):
    r[f'diff_item_{img_name}_str_first_original'] = r[f'diff_item_{img_name}_str_first']
    r[f'diff_item_{img_name}_str_original'] = r[f'diff_item_{img_name}_str']
    r[f'diff_item_{img_name}_original'] = r[f'diff_item_{img_name}']
    if r[f'vl_feats_bbox'] is not None:
        r[f'diff_item_{img_name}_str_first'] = r[f'vl_feats_bbox'][f'{img_name}_item_str']
        r[f'diff_item_{img_name}_str'] = nouns[r[f'vl_feats_bbox'][f'{img_name}_item_img']]['gloss']
        r[f'diff_item_{img_name}'] = r[f'vl_feats_bbox'][f'{img_name}_item_img']


def get_all_dataframes():
    AB_matches_vision_and_language_feats_path_final = AB_matches_vision_and_language_feats_path
    if SAMPLE:
        AB_matches_vision_and_language_feats_path_final = AB_matches_vision_and_language_feats_path.replace(".csv", "_debug.csv")
    df_with_feats = read_and_json_loads(AB_matches_vision_and_language_feats_path_final)
    initial_df = read_and_json_loads(AB_matches_filtered_visual)
    AB_matches_objects_no_bbox_feats_path_final = AB_matches_objects_no_bbox_feats_path
    if SAMPLE:
        AB_matches_objects_no_bbox_feats_path_final = AB_matches_objects_no_bbox_feats_path.replace(".csv","_debug.csv")
    df_without_feats = read_and_json_loads(AB_matches_objects_no_bbox_feats_path_final)
    return df_with_feats, df_without_feats, initial_df


def read_and_json_loads(df_p):
    df = pd.read_csv(df_p)
    for c in columns_to_serialize:
        if c in df.columns:
            if c in ['vl_feats_bbox', 'vl_feats_full_img']:
                df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                df[c] = df[c].apply(lambda x: json.loads(x))
    return df


def print_stats(all_items_to_filter, all_items_to_keep, idx):
    print(f"From {idx} V&L feats, filter: {len(all_items_to_filter)} "
          f"({round(len(all_items_to_filter) / idx * 100, 2)}%), "
          f"keeping: {len(all_items_to_keep)}")
    filter_bbox = {k: v for k, v in filter_stats.items() if 'bbox' in k}
    sum_filter_bbox = sum(list(filter_bbox.values()))

    def rename_key(k):
        for c in ['_vl_feats_bbox', '_vl_feats_full_img', '_filter_1', '_filter_2', '_filter_3']:
            k  = k.replace(c, '')
        return k

    filter_bbox_percentages = {rename_key(k): f"{int(v / sum_filter_bbox * 100)}%" for k,v in filter_bbox.items()}
    filter_bbox_percentages['total'] = sum_filter_bbox
    filter_full_img = {k: v for k, v in filter_stats.items() if 'full_img' in k}
    sum_filter_full_img = sum(list(filter_full_img.values()))
    filter_full_img_percentages = {rename_key(k): f"{int(v / sum_filter_full_img * 100)}%" for k,v in filter_full_img.items()}
    filter_full_img_percentages['total'] = sum_filter_full_img
    total_filter = sum_filter_bbox + sum_filter_full_img
    stats_df = pd.DataFrame([filter_bbox_percentages, filter_full_img_percentages], index=['bbox', 'full img'])
    print(f'filter stats filter: {total_filter}, stayed: {len(all_items_to_keep)}')
    print(stats_df)
    # print(f'bbox ({sum_filter_bbox}), which is {round(sum_filter_bbox / total_filter * 100, 1)}% of total filter')
    # print(filter_bbox)
    # print(f'full image ({sum_filter_full_img})')
    # print(filter_full_img)


def load_dataset():
    AB_matches_vision_and_language_feats_path_final = AB_matches_vision_and_language_feats_path
    if SAMPLE:
        AB_matches_vision_and_language_feats_path_final = AB_matches_vision_and_language_feats_path.replace(".csv", "_debug.csv")
    df_with_feats = pd.read_csv(AB_matches_vision_and_language_feats_path_final)
    for c in ['vl_feats_bbox', 'vl_feats_full_img', 'A_bounding_box', 'B_bounding_box', 'diff_item_A_str',
              'diff_item_B_str', 'A_data', 'B_data']:
        print(c)
        df_with_feats[c] = df_with_feats[c].apply(lambda x: json.loads(x))
    return df_with_feats


def get_title_text(filter_reason_bbox, filter_reason_img, should_remove, should_remove_bbox):
    if should_remove:
        additional_txt = 'bbox' if should_remove_bbox else 'full img'
        additional_txt = f'should filter ({additional_txt})\n'
        if should_remove_bbox:
            additional_txt += f"{filter_reason_bbox}"
        else:
            additional_txt += f"{filter_reason_img}"
    else:
        additional_txt = 'keep'
    return additional_txt


def visualize_pair_clip_all_feats(r, additional_txt, all_items_to_filter, all_items_to_keep):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
    should_remove = additional_txt != 'keep'
    title = f"Difference: {r['different_key']}, {additional_txt}"
    plt.suptitle(title, fontsize=18)
    if r['vl_feats_bbox'] is not None:
        visualize_pair_clip_img_feats(axs, r, feats_type='vl_feats_bbox')
    visualize_pair_clip_img_feats(axs, r, feats_type='vl_feats_full_img')
    plots_dir = plots_filtered_path if should_remove else plots_good_pairs_path
    idx = len(all_items_to_filter) if should_remove else len(all_items_to_keep)
    plt.tight_layout()
    out_p = os.path.join(plots_dir, f"{idx}.png")
    plt.savefig(out_p)
    plt.close(fig)
    plt.cla()
    # print(f"out_p: {out_p}")


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

def classify_given_feats(r, pairs_filter, feats_type):
    vl_feats = {k: v for k,v in r[feats_type].items()}
    pair_tuple = (vl_feats['A_item_str'], vl_feats['B_item_str'], vl_feats['A_item_img'], vl_feats['B_item_img'])
    is_legit_k_chagnge_filter_1 = pairs_filter.is_legit_k_chagnge(r['different_key'], pair_tuple)
    should_filter = False
    filter_reason = None
    if not is_legit_k_chagnge_filter_1:
        filter_reason = f'is_legit_k_chagnge_filter_1_{feats_type}'
        should_filter = True
    is_image_ambiguous_filter_2 = is_image_ambiguous(vl_feats['probs_A_img_logits_per_AB_class_round'],
                                                     vl_feats['probs_B_img_logits_per_AB_class_round'])
    if is_image_ambiguous_filter_2:
        filter_reason = f'is_image_ambiguous_filter_2_{feats_type}'
        should_filter = True
    is_AB_similar_mesh_filter_3 = vl_feats['mesh_data_AB']['filter'] and vl_feats['mesh_data_BA']['filter']
    if is_AB_similar_mesh_filter_3:
        filter_reason = f'is_AB_similar_mesh_filter_3_{feats_type}'
        should_filter = True
    global filter_stats
    if should_filter:
        filter_stats[filter_reason] += 1
    return should_filter, filter_reason


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



# Prob_A is for image A with sent AB
def is_image_ambiguous(prob_A_class_AB, prob_B_class_AB):
    return maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB) or opposite_prob(prob_A_class_AB, prob_B_class_AB)


def maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB):
    return (0.41 <= prob_A_class_AB[0] <= 0.61 and prob_B_class_AB[0] > 0) or (
            0.41 <= prob_B_class_AB[0] <= 0.61 and prob_A_class_AB[1] > 0)

# def maximal_wrong_prob_v2(prob_A_class_AB, prob_B_class_AB):
#     return (0.30 <= prob_A_class_AB[0] <= 0.70 and prob_B_class_AB[0] > 0) or (
#             0.30 <= prob_B_class_AB[0] <= 0.70 and prob_A_class_AB[1] > 0)

# def same_prob(prob_A_class_AB, prob_B_class_AB):
#   return prob_A_class_AB == prob_B_class_AB and ((prob_B_class_AB[0] >= 0.60 and prob_B_class_AB[1] >= 0.30) or (
#              prob_B_class_AB[1] >= 0.60 and prob_B_class_AB[0] >= 0.30))

def opposite_prob(prob_A_class_AB, prob_B_class_AB):
    return prob_B_class_AB[0] >= 0.66 or prob_A_class_AB[1] >= 0.66


def is_same_label(img_a_str, img_b_str):
    try:
        a = wn.synsets(img_a_str.replace(" ", "_").lower(), pos='n')
        b = wn.synsets(img_b_str.replace(" ", "_").lower(), pos='n')
        return wn.wup_similarity(a[0], b[0]) >= 0.96
    except:
        return False


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



def visualize_pair_clip_img_feats(axs, r, feats_type):
    A_original_item = r['diff_item_A_str'][0]
    B_original_item = r['diff_item_B_str'][0]

    A_img = get_img_full_or_bbox(r['A_img'], r['A_bounding_box'][r['different_key']], feats_type)
    B_img = get_img_full_or_bbox(r['B_img'], r['B_bounding_box'][r['different_key']], feats_type)
    row_idx = int(feats_type == 'vl_feats_bbox')
    axs[row_idx, 0].imshow(A_img)
    axs[row_idx, 1].imshow(B_img)
    vl_feats = {k: v for k,v in r[feats_type].items()}
    A_title = f"{vl_feats['A_item_clip_sent']}\nA: {A_original_item} -> {vl_feats['A_item_str']}, P(A,B): {vl_feats['probs_A_img_logits_per_AB_class_round']}"
    A_mesh_data = f"labels: {list2str(vl_feats['mesh_data_AB']['first_img_strs'])}\nbest label: {vl_feats['mesh_data_AB']['best_label']}"
    if vl_feats['mesh_data_AB']['best_label'] in vl_feats['mesh_data_AB']['second_img_strs']:
        A_mesh_data += "(!!!)"
    A_title += f"\n{A_mesh_data}"
    B_title = f"{vl_feats['B_item_clip_sent']}\nB: {B_original_item} -> {vl_feats['B_item_str']}, P(A,B): {vl_feats['probs_B_img_logits_per_AB_class_round']}"
    B_mesh_data = f"labels: {list2str(vl_feats['mesh_data_BA']['first_img_strs'])}\nbest label: {vl_feats['mesh_data_BA']['best_label']}"
    if vl_feats['mesh_data_BA']['best_label'] in vl_feats['mesh_data_BA']['second_img_strs']:
        B_mesh_data += "(!!!)"
    B_title += f"\n{B_mesh_data}"
    axs[row_idx, 0].set_title(A_title, fontsize=14)
    axs[row_idx, 1].set_title(B_title, fontsize=14)

def get_img_full_or_bbox(img_name, bbox, feats_type):
    img_path = os.path.join(swig_images_path, img_name)
    img = Image.open(img_path)
    x1, y1, x2, y2 = bbox
    if feats_type == 'vl_feats_bbox':
        img = img.crop((x1, y1, x2, y2))
    else:
        full_img = np.asarray(img)
        img = cv2.rectangle(full_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def get_clip_img_at_bbox(device, preprocess, file_path, bbox):
    full_img_bbox, img_cropped = get_img_and_bbox(bbox, file_path)
    img_data = {'full_img_bbox': full_img_bbox, 'cropped_img': np.asarray(img_cropped),
                'tensor': preprocess(img_cropped).unsqueeze(0).to(device)}
    return img_data


def get_img_and_bbox(bbox, file_path):
    img = Image.open(file_path)
    x1, y1, x2, y2 = bbox
    img_cropped = img.crop((x1, y1, x2, y2))
    full_img = np.asarray(img)
    full_img_bbox = cv2.rectangle(full_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return full_img_bbox, img_cropped


'''
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 0,100000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 100000,200000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 200000,300000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 300000,400000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 400000,500000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 500000,600000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 600000,700000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 700000,800000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 800000,900000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 900000,1000000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 1000000,1100000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 1100000,1200000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 1200000,1300000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 1300000,1400000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 1400000,1500000
CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_filter_vision_and_language.py --indices 1500000,1650000


CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_filter_vision_and_language.py --indices 0,20
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 0,50000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 50000,100000
CUDA_VISIBLE_DEVICES=7 python src_dataset_generation/D_filter_vision_and_language.py --indices 100000,150000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 150000,200000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 200000,250000
CUDA_VISIBLE_DEVICES=6 python src_dataset_generation/D_filter_vision_and_language.py --indices 300000,350000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 250000,300000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 350000,400000
CUDA_VISIBLE_DEVICES=5 python src_dataset_generation/D_filter_vision_and_language.py --indices 400000,450000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 450000,500000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 500000,550000
CUDA_VISIBLE_DEVICES=4 python src_dataset_generation/D_filter_vision_and_language.py --indices 550000,600000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 600000,650000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 650000,700000
CUDA_VISIBLE_DEVICES=3 python src_dataset_generation/D_filter_vision_and_language.py --indices 700000,750000
CUDA_VISIBLE_DEVICES=2 python src_dataset_generation/D_filter_vision_and_language.py --indices 750000,850000
'''

if __name__ == '__main__':
    print('Important: If you ran with --indices, run "merge_train_clip_VL_feats_for_AB_filter.py" later')
    main()