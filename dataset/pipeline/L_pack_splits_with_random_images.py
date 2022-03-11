import json
import os
import random

import pandas as pd
from tqdm import tqdm

from utils.utils import NUM_CANDIDATES, imsitu_path, SPLIT, split_to_files, data_path, BAD_IMAGES

def main(split_file_name):
    data_split = json.load(open(os.path.join(imsitu_path, f"{SPLIT}.json")))
    split_images = list(data_split.keys())
    split_file_path = os.path.join(data_path, 'ABCD_matches', split_file_name)
    all_ABCD_matches_df = pd.read_csv(split_file_path)
    len_all = len(all_ABCD_matches_df)
    all_ABCD_matches_df = all_ABCD_matches_df[all_ABCD_matches_df['different_key'] != 'place']
    print(f"-- Removed place. Now length is {len(all_ABCD_matches_df)}, was {len_all}")

    all_random_candidates = []
    for r_idx, r in tqdm(all_ABCD_matches_df.iterrows(), desc='Iteration analogies...', total=len(all_ABCD_matches_df)):
        random_candidates = get_random_images(r, split_images)
        all_random_candidates.append(random_candidates)
    all_ABCD_matches_df['random_candidates'] = all_random_candidates
    print(all_ABCD_matches_df['different_key'].value_counts())
    split_file_name_out = split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_random_candidates.csv')
    out_path = os.path.join(data_path, 'split_random', split_file_name_out)
    all_ABCD_matches_df.drop(columns=['A_annotations_str', 'B_annotations_str', 'C_annotations_str', 'D_annotations_str'], inplace=True)
    for c in ['A_bounding_box', 'B_bounding_box', 'C_bounding_box', 'D_bounding_box']:
        if c in all_ABCD_matches_df:
            all_ABCD_matches_df[c] = all_ABCD_matches_df[c].apply(lambda x: json.loads(x.replace("'",'"')))
            all_ABCD_matches_df[c] = all_ABCD_matches_df[c].apply(json.dumps)
    parallel_distractors_file_path = out_path.replace("random_candidates","distractors").replace("random","distractors")
    assert os.path.exists(parallel_distractors_file_path)
    parallel_distractors_df = pd.read_csv(parallel_distractors_file_path)
    parallel_distractors_df_size = len(parallel_distractors_df)
    print(f"Sampling df at len {len(all_ABCD_matches_df)} to parallel distractors df: {parallel_distractors_df_size}")
    all_ABCD_matches_df = all_ABCD_matches_df.sample(parallel_distractors_df_size)
    all_ABCD_matches_df.to_csv(out_path, index=False)
    print(f"Dumped {len(all_ABCD_matches_df)} analogies of SPLIT {SPLIT} to {out_path}")
    return all_ABCD_matches_df


def get_random_images(r, split_images):
    r_images = {r['A_img'], r['B_img'], r['C_img'], r['D_img']}
    ab_verb = r['A_verb']
    cd_verb = r['C_verb']
    relevant_images = [f for f in split_images if f not in r_images and ab_verb not in f and cd_verb not in f]
    relevant_images = [f for f in relevant_images if f not in BAD_IMAGES]
    sampled_random_images = random.sample(relevant_images, NUM_CANDIDATES-1)
    return json.dumps(sampled_random_images)


if __name__ == '__main__':
    for split_idx, split_file_name in enumerate(split_to_files[SPLIT]):
        print(f"Packing {split_file_name}")
        all_ABCD_matches_df = main(split_file_name)

        if split_file_name == f'all_ABCD_matches_rule_based_sampled_train_ood.csv' or split_file_name == f'all_ABCD_matches_rule_based_sampled_{SPLIT}.csv':
            all_ABCD_matches_df.to_csv(os.path.join(data_path, 'split_random', f"analogies_random_candidates_{SPLIT}_final.csv"))
            print(f"In case of {split_file_name}, choosing it as final analogies file: analogies_random_candidates_{SPLIT}_final.csv")
    print("Done")