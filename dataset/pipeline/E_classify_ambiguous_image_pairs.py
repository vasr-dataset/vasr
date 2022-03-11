import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.PairFilter import PairsFilter
from utils.utils import SPLIT, AB_matches_filtered_path, plots_path, AB_matches_filtered_visual, \
    AB_matches_vision_and_language_feats_path, AB_matches_objects_no_bbox_feats_path, \
    AB_matches_vision_and_language_feats_to_filter, AB_matches_vision_and_language_feats_to_keep, \
    columns_to_serialize, swig_path

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

filter_stats = defaultdict(int)

def main():
    df_verb_objects_feats, df_objects_no_bbox_feats, initial_df = get_all_dataframes()
    df_with_feats = pd.concat([df_verb_objects_feats, df_objects_no_bbox_feats])
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

        if idx == 1000 or (0 < idx and idx % modulo == 0):
            print_stats(all_items_to_filter, all_items_to_keep, idx)
        if should_remove:
            all_items_to_filter.append(r)
        else:
            all_items_to_keep.append(r)

    print_stats(all_items_to_filter, all_items_to_keep, idx)
    print("filter_stats")
    print(filter_stats)

    dump_outputs(all_items_to_filter, all_items_to_keep, df_objects_no_bbox_feats, df_with_feats)


def dump_outputs(all_items_to_filter, all_items_to_keep, df_objects_no_bbox_feats, df_with_feats):
    all_items_to_filter_df = pd.DataFrame(all_items_to_filter)
    print(f"Final. From {len(df_with_feats)} V&L feats, filter: {len(all_items_to_filter_df)} "
          f"({round(len(all_items_to_filter_df) / len(df_with_feats) * 100, 1)}%), "
          f"keeping: {len(all_items_to_keep)}")
    print(
        f"Writing filtered items to: {AB_matches_vision_and_language_feats_to_filter}, and items to keep to {AB_matches_vision_and_language_feats_to_keep}")
    all_items_to_filter_df.to_csv(AB_matches_vision_and_language_feats_to_filter)
    all_items_to_keep_df = pd.DataFrame(all_items_to_keep)
    all_items_to_keep_df.to_csv(AB_matches_vision_and_language_feats_to_keep)
    AB_matches_filtered_path_final = AB_matches_filtered_path if SAMPLE == False else AB_matches_filtered_path.replace(
        ".csv", "_debug.csv")
    print(
        f"In addition, {len(df_objects_no_bbox_feats)} dont have feats. Merging with the items to keep: {len(all_items_to_keep_df)}\n"
        f"and writing to the final path: {AB_matches_filtered_path_final}")
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
    df_with_feats = read_and_json_loads(AB_matches_vision_and_language_feats_path_final)
    initial_df = read_and_json_loads(AB_matches_filtered_visual)
    AB_matches_objects_no_bbox_feats_path_final = AB_matches_objects_no_bbox_feats_path
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


# Prob_A is for image A with sent AB
def is_image_ambiguous(prob_A_class_AB, prob_B_class_AB):
    return maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB) or opposite_prob(prob_A_class_AB, prob_B_class_AB)


def maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB):
    return (0.41 <= prob_A_class_AB[0] <= 0.61 and prob_B_class_AB[0] > 0) or (
            0.41 <= prob_B_class_AB[0] <= 0.61 and prob_A_class_AB[1] > 0)

def opposite_prob(prob_A_class_AB, prob_B_class_AB):
    return prob_B_class_AB[0] >= 0.66 or prob_A_class_AB[1] >= 0.66

if __name__ == '__main__':
    print('Important: If you ran with --indices, run "merge_train_clip_VL_feats_for_AB_filter.py" later')
    main()