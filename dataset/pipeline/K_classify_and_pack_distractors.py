import json
import numpy as np
import os
import random

import pandas as pd
from tqdm import tqdm

from config import imsitu_path
from utils.PairFilter import PairsFilter
from utils.utils import data_path, SPLIT, split_to_files, columns_to_serialize, FINAL_COLS_TRAIN

imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
nouns = imsitu["nouns"]

total_items_received = 0
total_items_after_filtering = 0
distractors_filtered_by_soft_similarity = 0
analogies_with_distractors_filtered_by_soft_similarity = 0

def main(split_file_name):
    split_file_name_in = get_analogies_name(split_file_name)
    in_path = os.path.join(data_path, 'split_distractors', split_file_name_in)
    print(f'Reading: {in_path}')
    df = pd.read_csv(in_path)
    for c in columns_to_serialize:
        if c in df.columns:
            if c in ['B_distractors_data', 'C_distractors_data', 'vl_feats_bbox_AB', 'vl_feats_bbox_CD']:
                df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                df[c] = df[c].apply(json.loads)

    pairs_filter = PairsFilter()

    all_final_distractors_data = []
    all_candidates = []
    all_labels = []
    not_enough_distractors_count = 0

    for r_idx, r in tqdm(df.iterrows(), total=len(df), desc='Classifying...'):
        D_annotations_plus_verb = r['D_annotations']
        D_annotations_plus_verb['verb'] = r['D_verb']
        if random.random() < 0.5:
            B_distractors_filtered, B_bad_distractors = get_top_images_after_clip_filter(r, r['B_distractors_data'], D_annotations_plus_verb, pairs_filter)
            C_distractors_filtered, C_bad_distractors = get_top_images_after_clip_filter(r, r['C_distractors_data'], D_annotations_plus_verb, pairs_filter, ignore_images=[x['img_name'] for x in B_distractors_filtered])
        else:
            C_distractors_filtered, C_bad_distractors = get_top_images_after_clip_filter(r, r['C_distractors_data'], D_annotations_plus_verb, pairs_filter)
            B_distractors_filtered, B_bad_distractors = get_top_images_after_clip_filter(r, r['B_distractors_data'], D_annotations_plus_verb, pairs_filter, ignore_images=[x['img_name'] for x in C_distractors_filtered])

        merged_distractors_data_possible_dups = B_distractors_filtered + C_distractors_filtered
        random.shuffle(merged_distractors_data_possible_dups)
        existing_images = []
        merged_distractors_data = []
        for x in merged_distractors_data_possible_dups:
            if x['img_name'] not in existing_images:
                existing_images.append(x['img_name'])
                merged_distractors_data.append(x)
        if len(merged_distractors_data) == 3:
            final_distractors_data = merged_distractors_data
        elif len(merged_distractors_data) == 4:
            final_distractors_data = choose_top_distractors(merged_distractors_data, 3, include_sim_with_input=False)
        else:
            not_enough_distractors_count += 1
            final_distractors_data = None
        if final_distractors_data:
            candidates = [x['img_name'] for x in final_distractors_data] + [r['D_img']]
            random.shuffle(candidates)
            label = candidates.index(r['D_img'])
        else:
            candidates, label = None, None
        all_candidates.append(candidates)
        all_labels.append(label)
        all_final_distractors_data.append(final_distractors_data)

    df['candidates'] = all_candidates
    df['label'] = all_labels
    df['distractors_data'] = all_final_distractors_data
    len_before_taking_only_valid_distractors = len(df)
    df = df[~df['distractors_data'].isna()]
    print(f"Started with {len_before_taking_only_valid_distractors} items, after taking valid distractors achieved final {len(df)} items")
    df['distractors'] = df['distractors_data'].apply(lambda lst: [x['img_name'] for x in lst])
    print(f"not_enough_distractors_count: {not_enough_distractors_count}")

    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.dumps)
    split_file_name_out = get_analogies_name_out_path(split_file_name)
    out_path = os.path.join(data_path, 'split_distractors', split_file_name_out)
    if SPLIT == 'train':
        df.drop(columns=['distractors_data'], inplace=True)

    df.to_csv(out_path, index=False)
    print(f"analogies_with_distractors_filtered_by_soft_similarity: {analogies_with_distractors_filtered_by_soft_similarity}, "
          f"distractors_filtered_by_soft_similarity: {distractors_filtered_by_soft_similarity}")
    print(f'total_items_received: {total_items_received}, total_items_after_filtering: {total_items_after_filtering}, distractors after filter: {round(total_items_after_filtering / total_items_received * 100, 2)}%')

    print(f"Dumped {len(df)} analogies of SPLIT {SPLIT} to {out_path}")


def distractor_is_identical_to_solution_by_soft_match_pairs_filter(dist_data, pairs_filter, D_annotations):
    if 'farmer' in dist_data.keys() and 'agent' not in dist_data.keys():
        dist_data['agent'] = dist_data['farmer']
        del dist_data['farmer']
    if 'farmer' in D_annotations.keys() and 'agent' not in D_annotations.keys():
        D_annotations['agent'] = D_annotations['farmer']
        del D_annotations['farmer']
    if 'place' in dist_data:
        del dist_data['place']
    if 'place' in D_annotations:
        del D_annotations['place']
    intersecting_keys = set(dist_data.keys()).intersection(set(D_annotations.keys()))
    same_values_according_to_pairs_filter = []
    for k in intersecting_keys:
        value_distractor = dist_data[k]
        value_D = D_annotations[k]
        if value_distractor == '' or value_D == '':
            is_legit_k_change = True
        else:
            value_distractor_str = nouns[value_distractor]['gloss'][0] if k != 'verb' else value_distractor
            value_D_str = nouns[value_D]['gloss'][0] if k != 'verb' else value_D
            t = (value_distractor_str, value_D_str, value_distractor, value_D)
            is_legit_k_change = pairs_filter.is_legit_k_chagnge(k, t)
        if not is_legit_k_change:
            same_values_according_to_pairs_filter.append((k, value_distractor, value_D))
    num_shared_values = len(same_values_according_to_pairs_filter)
    num_shared_values_divided_by_number_of_keys = round(num_shared_values / len(intersecting_keys), 2)
    dist_data['cand_sim_with_sol_soft_similarity'] = num_shared_values_divided_by_number_of_keys
    if num_shared_values_divided_by_number_of_keys == 1:
        return True
    else:
        return False


def get_top_images_after_clip_filter(r, curr_dist_lst, D_annotations_plus_verb, pairs_filter, ignore_images=None):
    curr_dist_lst_not_ambiguous = []
    found_bad_distractor_for_analogy = False
    bad_images = []
    for dist_data in curr_dist_lst:
        if ignore_images and dist_data['img_name'] in ignore_images:
            continue
        bad_image = is_image_ambiguous(dist_data['clip_features']['A_img_AB_probs'],dist_data['clip_features']['B_img_AB_probs']) \
                    or is_image_ambiguous_mesh(dist_data['clip_features']['A_clip_sents'], dist_data['clip_features']['B_clip_sents'],
                                               dist_data['clip_features']['A_logits_per_class'], dist_data['clip_features']['B_logits_per_class'])
        if not bad_image and SPLIT != 'train':
            bad_image = distractor_is_identical_to_solution_by_soft_match_pairs_filter(dist_data, pairs_filter, D_annotations_plus_verb)
            if bad_image:
                found_bad_distractor_for_analogy = True
                global distractors_filtered_by_soft_similarity
                distractors_filtered_by_soft_similarity += 1
        if not bad_image:
            curr_dist_lst_not_ambiguous.append(dist_data)
        else:
            bad_images.append(dist_data)

    if found_bad_distractor_for_analogy:
        global analogies_with_distractors_filtered_by_soft_similarity
        analogies_with_distractors_filtered_by_soft_similarity += 1

    items_received = len(curr_dist_lst)
    items_after_filtering = len(curr_dist_lst_not_ambiguous)
    global total_items_received, total_items_after_filtering
    total_items_received += items_received
    total_items_after_filtering += items_after_filtering

    curr_dist_lst_not_ambiguous_top_2 = choose_top_distractors(curr_dist_lst_not_ambiguous, 2)
    return curr_dist_lst_not_ambiguous_top_2, bad_images



def choose_top_distractors(distractors, num, include_sim_with_input=True):
    if include_sim_with_input:
        distractors.sort(reverse=True, key=lambda x: (x['cand_sim_with_input'], calculate_sim_score(x['clip_features']['A_img_AB_probs'], x['clip_features']['B_img_AB_probs'],
                                                                                                    x['clip_features']['A_logits_per_class'], x['clip_features']['B_logits_per_class'])))
    else:
        distractors.sort(reverse=True, key=lambda x: calculate_sim_score(x['clip_features']['A_img_AB_probs'], x['clip_features']['B_img_AB_probs'],
                                                                         x['clip_features']['A_logits_per_class'], x['clip_features']['B_logits_per_class']))
    return distractors[:num]


def calculate_sim_score(A_img_AB_probs, B_img_AB_probs,logits_per_class_A, logits_per_class_B):
    average_AB_prob = (A_img_AB_probs[0] + B_img_AB_probs[1])/2
    max_logit_A = np.max(logits_per_class_A)
    max_logit_B = np.max(logits_per_class_B)
    return ((max_logit_A + max_logit_B) / 100) + average_AB_prob


def is_image_ambiguous(prob_A_class_AB, prob_B_class_AB):
    return maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB) or opposite_prob(prob_A_class_AB, prob_B_class_AB)

def is_image_ambiguous_mesh(A_clip_sents, B_clip_sents, logits_per_class_A, logits_per_class_B):
    return is_AB_similar_mesh(A_clip_sents, B_clip_sents, logits_per_class_A, 'A') or is_AB_similar_mesh(A_clip_sents, B_clip_sents, logits_per_class_B, 'B')

def maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB, min_prob=0.41, max_prob=0.61):
    return (min_prob <= prob_A_class_AB[0] <= max_prob and prob_B_class_AB[0] > 0) or (
            min_prob <= prob_B_class_AB[0] <= max_prob and prob_A_class_AB[1] > 0)

def opposite_prob(prob_A_class_AB, prob_B_class_AB, prob_threshold=0.66):
    return prob_B_class_AB[0] >= prob_threshold or prob_A_class_AB[1] >= prob_threshold

def is_AB_similar_mesh(A_sents, B_sents, logits_per_class, A_or_B):
    if len(A_sents) == 0 or len(B_sents) == 0:
        print('Invalid sents, returning similar True')
        return True

    max_index = np.argmax(logits_per_class)
    if A_or_B == 'A':
        return max_index > (len(A_sents) - 1)
    elif A_or_B == 'B':
        return max_index <= (len(A_sents) - 1)
    else:
        Exception('Invalid parameter sent')
    return True


def get_analogies_name(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors_with_clip_features.csv')

def get_analogies_name_out_path(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors.csv')

if __name__ == '__main__':
    for split_file_name in split_to_files[SPLIT]:
        print(f"Packing {split_file_name}")
        main(split_file_name)
    print("Done")
