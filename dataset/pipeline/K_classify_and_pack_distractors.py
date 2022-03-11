import json
import os
import random

import pandas as pd
from tqdm import tqdm

from utils.utils import data_path, SPLIT, split_to_files, columns_to_serialize, FINAL_COLS_TRAIN

total_items_received = 0
total_items_after_filtering = 0

def main(split_file_name):
    split_file_name_in = get_analogies_name(split_file_name)
    in_path = os.path.join(data_path, 'split_distractors', split_file_name_in)
    print(f'Reading: {in_path}')
    df = pd.read_csv(in_path)
    for c in columns_to_serialize:
        if c in df.columns:
            if c in ['B_distractors_data', 'C_distractors_data']:
                df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                df[c] = df[c].apply(json.loads)

    all_final_distractors_data = []

    visualize_bad_distractors = False

    for r_idx, r in tqdm(df.iterrows(), total=len(df), desc='Classifying...'):
        B_distractors_clip_filtered, B_bad_distractors = get_top_images_after_clip_filter(r['B_distractors_data'])
        C_distractors_clip_filtered, C_bad_distractors = get_top_images_after_clip_filter(r['C_distractors_data'])

        merged_distractors_data = B_distractors_clip_filtered + C_distractors_clip_filtered
        if len(merged_distractors_data) == 3:
            final_distractors_data = merged_distractors_data
        elif len(merged_distractors_data) == 4:
            final_distractors_data = random.sample(merged_distractors_data, 3)
        else:
            final_distractors_data = None
        if final_distractors_data:
            random.shuffle(final_distractors_data)
        all_final_distractors_data.append(final_distractors_data)

    df['distractors_data'] = all_final_distractors_data
    len_before_taking_only_valid_distractors = len(df)
    df = df[~df['distractors_data'].isna()]
    print(f"Started with {len_before_taking_only_valid_distractors} items, after taking valid distractors achieved final {len(df)} items")
    df['distractors'] = df['distractors_data'].apply(lambda lst: [x['img_name'] for x in lst])

    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.dumps)
    split_file_name_out = get_analogies_name_out_path(split_file_name)
    out_path = os.path.join(data_path, 'split_distractors', split_file_name_out)
    if SPLIT == 'train':
        df.drop(columns=['distractors_data'], inplace=True)

    if split_file_name_out == 'analogies_train_full_at_size_of_ood_distractors.csv':
        ood_df_path = os.path.join(data_path, 'split_distractors', 'analogies_train_ood_distractors.csv')
        ood_df = pd.read_csv(ood_df_path)
        if len(df) > len(ood_df):
            print(f"Case of Train Full size of OOD, sampling {len(df)} -> {len(ood_df)}")
            df = df.sample(len(ood_df))

    df.to_csv(out_path, index=False)

    print(f'total_items_received: {total_items_received}, total_items_after_filtering: {total_items_after_filtering}, distractors after filter: {round(total_items_after_filtering / total_items_received * 100, 2)}%')

    print(f"Dumped {len(df)} analogies of SPLIT {SPLIT} to {out_path}")
    return df


def get_top_images_after_clip_filter(curr_dist_lst):
    curr_dist_lst_not_ambiguous = []
    bad_images = []
    for dist_data in curr_dist_lst:
        bad_image = is_image_ambiguous(dist_data['clip_features']['A_img_AB_probs'],
                                       dist_data['clip_features']['B_img_AB_probs'])
        if not bad_image:
            curr_dist_lst_not_ambiguous.append(dist_data)
        else:
            bad_images.append(dist_data)
    items_received = len(curr_dist_lst)
    items_after_filtering = len(curr_dist_lst_not_ambiguous)
    global total_items_received, total_items_after_filtering
    total_items_received += items_received
    total_items_after_filtering += items_after_filtering

    curr_dist_lst_not_ambiguous_top_2 = curr_dist_lst_not_ambiguous[:2]
    return curr_dist_lst_not_ambiguous_top_2, bad_images


def is_image_ambiguous(prob_A_class_AB, prob_B_class_AB):
    return maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB) or opposite_prob(prob_A_class_AB, prob_B_class_AB)


def maximal_wrong_prob(prob_A_class_AB, prob_B_class_AB):
    return (0.41 <= prob_A_class_AB[0] <= 0.61 and prob_B_class_AB[0] > 0) or (
            0.41 <= prob_B_class_AB[0] <= 0.61 and prob_A_class_AB[1] > 0)

def opposite_prob(prob_A_class_AB, prob_B_class_AB):
    return prob_B_class_AB[0] >= 0.66 or prob_A_class_AB[1] >= 0.66


def get_analogies_name(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors_with_clip_features.csv')

def get_analogies_name_out_path(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors.csv')

if __name__ == '__main__':
    for split_file_name in split_to_files[SPLIT]:
        # if split_file_name != 'all_ABCD_matches_rule_based_sampled_train_full_at_size_of_ood.csv':
        #     continue
        print(f"Packing {split_file_name}")
        all_ABCD_matches_df = main(split_file_name)
        if split_file_name == 'all_ABCD_matches_rule_based_sampled_train_ood':
            all_ABCD_matches_df = all_ABCD_matches_df[FINAL_COLS_TRAIN]
        if split_file_name == f'all_ABCD_matches_rule_based_sampled_train_ood.csv' or split_file_name == f'all_ABCD_matches_rule_based_sampled_{SPLIT}.csv':
            all_ABCD_matches_df.to_csv(os.path.join(data_path, 'split_distractors', f"analogies_distractors_{SPLIT}_final.csv"))
            print(f"In case of {split_file_name}, choosing it as final analogies file: analogies_distractors_{SPLIT}_final.csv")
    print("Done")
