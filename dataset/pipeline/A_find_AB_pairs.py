import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# The split should be train and testdev

from dataset.config import AB_matches_path, imsitu_path, SPLIT, swig_path, columns_to_serialize, BAD_IMAGES, \
    get_difference


def main():
    imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
    nouns = imsitu["nouns"]
    data_split = json.load(open(os.path.join(swig_path, f"{SPLIT}.json")))
    all_AB_matches = []
    matches_num_for_single_img = []
    for a_img_idx, A_img in tqdm(enumerate(data_split), total=len(data_split), desc=f'Iterating A'):
        A_B_matches = get_AB_matches_for_A_img(A_img, nouns, data_split)
        matches_num_for_single_img.append(len(A_B_matches))
        all_AB_matches += A_B_matches
        if a_img_idx % 1000 == 0:
            print(f"total matches: {len(all_AB_matches)}, average matches_num_for_single_img: {round(np.mean(matches_num_for_single_img), 2)}")
    all_AB_matches_df = pd.DataFrame(all_AB_matches)
    for c in columns_to_serialize:
        if c in all_AB_matches_df:
            all_AB_matches_df[c] = all_AB_matches_df[c].apply(json.dumps)

    print(f"Dumping total {len(all_AB_matches_df)} AB matches to {AB_matches_path}")
    all_AB_matches_df.to_csv(AB_matches_path, index=False)

    print("Done")


def get_AB_matches_for_A_img(A_img, nouns, data_split):
    all_A_and_B_matches = []
    if A_img in BAD_IMAGES:
        return all_A_and_B_matches
    A_img_df = pd.DataFrame(data_split[A_img]['frames'])
    A_bounding_box = data_split[A_img]['bb']
    for A in data_split[A_img]['frames']:
        A_verb = data_split[A_img]['verb']
        if '' in {k:v for k,v in A.items() if k != 'place'}:
            continue

        A_data = {"A": A, "A_str": {k:nouns[v]['gloss'] if v in nouns else None for k,v in A.items()},
         "A_img": A_img, "A_verb": A_verb, 'A_bounding_box': A_bounding_box}
        all_B_matches_data_given_A_img = find_B_for_A(data_split, nouns, A_img, A, A_verb, A_img_df)
        for B_data in all_B_matches_data_given_A_img:
            AB_match_dict = {'A_data': A_data, 'B_data': B_data}
            diff_item_A, diff_item_B, different_key = get_difference(AB_match_dict, A_data, B_data, str_fmt=False)
            diff_item_A_str, diff_item_B_str, different_key = get_difference(AB_match_dict, A_data, B_data, str_fmt=True,
                                                                             all_str=True)
            keys = list(A_data['A'].keys())
            AB_match_dict = {**AB_match_dict, 'diff_item_A': diff_item_A, 'diff_item_B': diff_item_B, 'different_key': different_key, 'keys': keys,
                                      'A_str': A_data['A_str'], 'B_str': B_data['B_str'],
                                      'diff_item_A_str': diff_item_A_str, 'diff_item_B_str': diff_item_B_str,
                             'A_img': A_data['A_img'], 'B_img': B_data['B_img'], 'A_verb': A_data['A_verb'], 'B_verb': B_data['B_verb']}
            if AB_match_dict not in all_A_and_B_matches:
                all_A_and_B_matches.append(AB_match_dict)
    return all_A_and_B_matches


def find_B_for_A(data_split, nouns, A_img, A, A_verb, A_img_df):
    """ Searching B for A that is the same except one key """
    all_B_matches_data_given_A_img = []
    for B_img_cand in data_split:
        if B_img_cand in BAD_IMAGES:
            continue
        if A_img == B_img_cand:  # We want different images
            continue
        all_B_matches_data_for_A_img_given_B_img = search_B_cand_frame(A, A_verb, B_img_cand, data_split, A_img_df, nouns)
        all_B_matches_data_given_A_img += all_B_matches_data_for_A_img_given_B_img
    return all_B_matches_data_given_A_img

def search_B_cand_frame(A, A_verb, B_img_cand, data_split, A_img_df, nouns):
    all_B_matches_data_for_img = []
    B_bounding_box = data_split[B_img_cand]['bb']
    for B_cand in data_split[B_img_cand]['frames']:
        if B_cand.keys() != A.keys():  # We want the annotation keys to be the same
            continue
        B_cand_verb = data_split[B_img_cand]['verb']

        found_B, key_to_differ = B_is_different_from_A_in_one_key_that_is_not_in_A(A, A_verb, B_cand, B_cand_verb, A_img_df)
        if found_B:
            B_data = {'B': B_cand, 'B_verb': B_cand_verb, 'different_key': key_to_differ,
                      'B_str': {k:nouns[v]['gloss'] if v in nouns else None for k,v in B_cand.items()},
                      'B_img': B_img_cand, 'B_bounding_box': B_bounding_box}
            all_B_matches_data_for_img.append(B_data)
    return all_B_matches_data_for_img


def B_is_different_from_A_in_one_key_that_is_not_in_A(A, A_verb, B_cand, B_cand_verb, A_img_df):
    found_B = False
    key_to_differ = None
    if all([A[k] == B_cand[k] for k in A.keys()]) and A_verb != B_cand_verb:
        found_B = True
        key_to_differ = 'verb'
    else:
        for key_to_differ in B_cand.keys():
            if all([A[k] == B_cand[k] for k in A.keys() if k != key_to_differ]) and A_verb == B_cand_verb:
                if A[key_to_differ] != B_cand[key_to_differ] and '' not in [A[key_to_differ], B_cand[key_to_differ]]:
                    if B_cand[key_to_differ] not in A_img_df[key_to_differ].values:
                        found_B = True
                        break
    return found_B, key_to_differ


if __name__ == '__main__':
    main()
