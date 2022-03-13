import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.dataset_config import imsitu_path, SPLIT
from dataset.utils.utils import get_dict_sim

path = 'dataset/analogies_outputs/data/split_distractors/analogies_distractors_test_final.csv'

def solve_analogy(r, data_split):
    r['B_annotations']['verb'] = r['B_verb']
    shared_key_but_not_val = {k: r['B_annotations'][k] for k in r['B_annotations'] if k in r['A_annotations'] and r['B_annotations'][k] != r['A_annotations'][k]}
    assert len(shared_key_but_not_val) == 1

    shared_key_but_not_val_key = list(shared_key_but_not_val.keys())[0]
    shared_key_but_not_val_val = shared_key_but_not_val[shared_key_but_not_val_key]

    assert shared_key_but_not_val_key == r['different_key']

    dict_to_search = deepcopy(r['C_annotations'])
    dict_to_search[shared_key_but_not_val_key] = shared_key_but_not_val_val

    found_solution = False
    solution_cand = None
    for cand in r['distractors']:
        cand_data = data_split[cand]
        if dict_to_search['verb'] != cand_data['verb']:
            continue
        for frame in cand_data['frames']:
            frame_cpy = deepcopy(frame)
            frame_cpy['verb'] = cand_data['verb']
            if frame_cpy == dict_to_search:
                found_solution = True
                solution_cand = cand
                break
        if found_solution:
            break

    if not found_solution:
        return True
    print(f"D: {r['D_img']}, sol: {solution_cand}")
    return False

def solve_analogy_all_frames(r, data_split):
    r['B_annotations']['verb'] = r['B_verb']
    found_solution = False
    solution_cand = None

    for A_annotations in data_split[r['A_img']]['frames']:
        A_annotations['verb'] = r['A_verb']
        for B_annotations in data_split[r['B_img']]['frames']:
            B_annotations['verb'] = r['B_verb']
            shared_key_but_not_val = {k: B_annotations[k] for k in B_annotations if k in A_annotations and B_annotations[k] != A_annotations[k]}
            if not len(shared_key_but_not_val) == 1:
                continue

            shared_key_but_not_val_key = list(shared_key_but_not_val.keys())[0]
            shared_key_but_not_val_val = shared_key_but_not_val[shared_key_but_not_val_key]

            for C_annotations in data_split[r['C_img']]['frames']:
                C_annotations['verb'] = r['C_verb']
                dict_to_search = deepcopy(C_annotations)
                dict_to_search[shared_key_but_not_val_key] = shared_key_but_not_val_val

                for cand in r['distractors']:
                    cand_data = data_split[cand]
                    if dict_to_search['verb'] != cand_data['verb']:
                        continue
                    for frame in cand_data['frames']:
                        frame_cpy = deepcopy(frame)
                        frame_cpy['verb'] = cand_data['verb']
                        if frame_cpy == dict_to_search:
                            found_solution = True
                            solution_cand = cand
                            break
                    if found_solution:
                        break

    if not found_solution:
        return True
    print(f"D: {r['D_img']}, sol: {solution_cand}")
    return False

def solve_analogy_all_frames_given_distractor(r, data_split, distractor):
    r['B_annotations']['verb'] = r['B_verb']
    found_solution = False
    solution_cand = None

    for A_annotations in data_split[r['A_img']]['frames']:
        A_annotations['verb'] = r['A_verb']
        for B_annotations in data_split[r['B_img']]['frames']:
            B_annotations['verb'] = r['B_verb']
            shared_key_but_not_val = {k: B_annotations[k] for k in B_annotations if k in A_annotations and B_annotations[k] != A_annotations[k]}
            if not len(shared_key_but_not_val) == 1:
                continue

            shared_key_but_not_val_key = list(shared_key_but_not_val.keys())[0]
            shared_key_but_not_val_val = shared_key_but_not_val[shared_key_but_not_val_key]

            for C_annotations in data_split[r['C_img']]['frames']:
                C_annotations['verb'] = r['C_verb']
                dict_to_search = deepcopy(C_annotations)
                dict_to_search[shared_key_but_not_val_key] = shared_key_but_not_val_val

                cand_data = data_split[distractor]
                if dict_to_search['verb'] != cand_data['verb']:
                    continue
                for frame in cand_data['frames']:
                    frame_cpy = deepcopy(frame)
                    frame_cpy['verb'] = cand_data['verb']
                    if frame_cpy == dict_to_search:
                        found_solution = True
                        solution_cand = distractor
                        break
                if found_solution:
                    break

    if not found_solution:
        return True
    return False


def observe_d_sim(r, data_split):
    all_sims = []
    for dist in r['distractors']:
        max_sim_for_dist = -1
        for frame in data_split[dist]['frames']:
            sim = get_dict_sim(r['D_annotations'], frame)
            if sim > max_sim_for_dist:
                max_sim_for_dist = sim
        all_sims.append(max_sim_for_dist)
    return all_sims, max(all_sims)


if __name__ == '__main__':

    analogies = pd.read_csv(path)
    annotations_cols = ['A_annotations', 'A_annotations_str', 'B_annotations', 'B_annotations_str', 'C_annotations', 'C_annotations_str', 'D_annotations', 'D_annotations_str', 'distractors', 'distractors_data']
    for c in annotations_cols:
        print(c)
        if c in analogies.columns:
            analogies[c] = analogies[c].apply(json.loads)

    data_split = json.load(open(os.path.join(imsitu_path, f"{SPLIT}.json")))
    all_res = []
    all_max_sims = []
    all_sims_lst = []
    for r_idx, r in tqdm(analogies.iterrows(), desc='solving', total=len(analogies)):
        success = solve_analogy_all_frames(r, data_split)
        all_res.append(success)

    print(f"Accuracy: # {np.mean(all_res)} items: {len(all_res)}, errors: {len(all_res) - sum(all_res)}")
