import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.utils.run_gsr_solver import solve_analogy_all_frames_given_distractor
from dataset.utils.visualization import visualize_analogy_and_distractors, plot_distractors
from utils.utils import imsitu_path, SPLIT, split_to_files, data_path, \
    distractors_cache_by_keys_path, get_dict_sim, BAD_IMAGES, columns_to_serialize

counter_found_match = {'try': 0, 'success_all': 0, 'fails': 0}
verbs_matchs = {'didnt_find': 0, 'tries': 0, 'succeeded': 0}
cand_sim_with_input_lst = []
iteration_lists_len = {'single': [], 'tuple': [], 'chosen': []}
choose_tuple = {'true': 0, 'false': 0}
distractors_stats = {'available': 0, 'solveable': 0}
clash_with_d_counter = {'true': 0, 'false': 0}
cases_with_pos_sim = 0
cases_with_filtered_solution = 0
distractor_solver_clashes_d = 0

MAX_NUM_CANDIDATES = 10
print(f'MAX_NUM_CANDIDATES: {MAX_NUM_CANDIDATES}')

def main(split_file_name):
    all_ABCD_matches_df, all_B_distractors_data, all_C_distractors_data, data_split,\
    distractors_cache_by_keys, modolu, plot = init_analogies(split_file_name)

    for idx, (r_idx, r) in enumerate(tqdm(all_ABCD_matches_df.iterrows(), desc='Iteration analogies...', total=len(all_ABCD_matches_df))):
        B_distractors_solveable, C_distractors_solveable = get_distractor_images(r, data_split, distractors_cache_by_keys)
        all_B_distractors_data.append(B_distractors_solveable)
        all_C_distractors_data.append(C_distractors_solveable)
        if plot:
            visualize_analogy_and_distractors(r, show_analogy_type=True, show_analogy_answer=True)

        if r_idx % modolu == 0:
            print_stats(all_B_distractors_data, all_C_distractors_data, r_idx, split_file_name)

    return dump_output(all_ABCD_matches_df, all_B_distractors_data, all_C_distractors_data, split_file_name)


def dump_output(all_ABCD_matches_df, all_B_distractors_data, all_C_distractors_data, split_file_name):
    print(
        f"cases_with_pos_sim: {cases_with_pos_sim}, cases_with_filtered_solution: {cases_with_filtered_solution}, clash_with_d_counter: {clash_with_d_counter}, distractor_solver_clashes_d: {distractor_solver_clashes_d}, distractors_stats: {distractors_stats}")
    all_ABCD_matches_df['B_distractors_data'] = all_B_distractors_data
    all_ABCD_matches_df['C_distractors_data'] = all_C_distractors_data
    print(all_ABCD_matches_df['different_key'].value_counts())
    for c in columns_to_serialize:
        if c in all_ABCD_matches_df:
            all_ABCD_matches_df[c] = all_ABCD_matches_df[c].apply(json.dumps)
    split_file_name_out = get_analogies_name(split_file_name)
    out_path = os.path.join(data_path, 'split_distractors', split_file_name_out)
    all_ABCD_matches_df.to_csv(out_path, index=False)
    print(f"Dumped {len(all_ABCD_matches_df)} analogies of SPLIT {SPLIT} to {out_path}")
    return all_ABCD_matches_df


def init_analogies(split_file_name):
    if not os.path.exists(distractors_cache_by_keys_path):
        create_cache()
    else:
        distractors_cache_by_keys = pickle.load(open(distractors_cache_by_keys_path, 'rb'))
    split_file_path = os.path.join(data_path, 'ABCD_matches', split_file_name)
    print(f"Reading {split_file_path}")
    all_ABCD_matches_df = pd.read_csv(split_file_path)
    len_all = len(all_ABCD_matches_df)
    all_ABCD_matches_df = all_ABCD_matches_df[all_ABCD_matches_df['different_key'] != 'place']
    print(f"-- Removed place. Now length is {len(all_ABCD_matches_df)}, was {len_all}")
    for c in columns_to_serialize:
        if c in all_ABCD_matches_df.columns:
            all_ABCD_matches_df[c] = all_ABCD_matches_df[c].apply(json.loads)
    data_split = json.load(open(os.path.join(imsitu_path, f"{SPLIT}.json")))
    all_B_distractors_data = []
    all_C_distractors_data = []
    plot = False
    modolu = 1000 if SPLIT != 'train' else 10000
    # modolu = 100 if SPLIT != 'train' else 10000
    return all_ABCD_matches_df, all_B_distractors_data, all_C_distractors_data, data_split, distractors_cache_by_keys, modolu, plot


def print_stats(all_B_distractors_data, all_C_distractors_data, r_idx, split_file_name):
    print(r_idx)
    print(counter_found_match)
    if r_idx > 1:
        print(f"split_file_name: {split_file_name}")
        print(
            f"cand_sim_with_input_lst: # items {len(cand_sim_with_input_lst)}, mean: {np.mean(cand_sim_with_input_lst)}")
        iteration_lists_len_summary = {k: np.mean(v) for k, v in iteration_lists_len.items()}
        print(f"iteration_lists_len_summary: {iteration_lists_len_summary}")
        print(
            f"cases_with_pos_sim: {cases_with_pos_sim}, cases_with_filtered_solution: {cases_with_filtered_solution}, clash_with_d_counter: {clash_with_d_counter}, distractor_solver_clashes_d: {distractor_solver_clashes_d}")

        B_stats = get_stat_for_dist_data(all_B_distractors_data)
        C_stats = get_stat_for_dist_data(all_C_distractors_data)
        sim_stats = {'B': B_stats,
                     'C': C_stats}
        print(sim_stats)


def get_stat_for_dist_data(dist_data):
    sim_with_input_mean = np.mean(
        [np.mean([x['cand_sim_with_input'] for x in lst]) if lst != [] else 0 for lst in dist_data if lst is not None])
    sim_with_input_mean_top_5 = np.mean(
        [np.mean([x['cand_sim_with_input'] for x in lst[:5]]) if lst != [] else 0 for lst in dist_data if
         lst is not None])
    sim_with_input_mean_top_2 = np.mean(
        [np.mean([x['cand_sim_with_input'] for x in lst[:2]]) if lst != [] else 0 for lst in dist_data if
         lst is not None])
    sim_with_sol_mean = np.mean(
        [np.mean([x['cand_sim_with_sol'] for x in lst]) if lst != [] else 0 for lst in dist_data if lst is not None])
    sim_with_sol_mean_top_5 = np.mean(
        [np.mean([x['cand_sim_with_sol'] for x in lst[:5]]) if lst != [] else 0 for lst in dist_data if
         lst is not None])
    sim_with_sol_mean_top_2 = np.mean(
        [np.mean([x['cand_sim_with_sol'] for x in lst[:2]]) if lst != [] else 0 for lst in dist_data if
         lst is not None])
    curr_dist_stat = {'input': round(sim_with_input_mean, 2), 'input top 5': round(sim_with_input_mean_top_5, 2),
                      'input top 2': round(sim_with_input_mean_top_2, 2),
                      'solution': round(sim_with_sol_mean, 2), 'solution top 5': round(sim_with_sol_mean_top_5, 2),
                      'solution top 2': round(sim_with_sol_mean_top_2, 2)}
    return curr_dist_stat


def create_cache():
    data_split = json.load(open(os.path.join(imsitu_path, f"{SPLIT}.json")))
    cache_by_keys = defaultdict(lambda: defaultdict(list))
    for img_name, img_data in tqdm(data_split.items(), desc="Creating cache", total=len(data_split)):
        if img_name in BAD_IMAGES:
            continue
        for frame_idx, img_frame in enumerate(img_data['frames']):
            img_frame['verb'] = img_data['verb']
            img_frame['img_name'] = img_name
            keys = [x for x in img_frame.keys() if x not in ['img_name']]
            for key in keys:
                cache_by_keys[key][img_frame[key]].append(img_frame)
            for key_pair in combinations(keys, 2):
                key1 = key_pair[0]
                key2 = key_pair[1]
                key_pair_key = "_".join(key_pair)
                # if img_frame[key1] is not None and img_frame[key1] != '' and img_frame[key2] is not None and img_frame[key1] != '':
                key_pair_val = "_".join([img_frame[key1], img_frame[key2]])
                cache_by_keys[key_pair_key][key_pair_val].append(img_frame)
    cache_by_keys_dict = dict({k:dict(v) for k,v in cache_by_keys.items()})
    pickle.dump(cache_by_keys_dict, open(distractors_cache_by_keys_path, 'wb'))
    print(f"Created cash. Run again and use it.")
    exit()


def get_analogies_name(split_file_name):
    return split_file_name.replace('all_ABCD_matches_rule_based_sampled', 'analogies').replace(".csv", '_distractors_before_clip_filter.csv')

def get_distractor_images(r, data_split, distractors_cache_by_keys):
    """ Improvements: verbs with similar meaning, images with same number of keys / same keys """
    r_images = {r['A_img'], r['B_img'], r['C_img'], r['D_img']}
    global counter_found_match
    counter_found_match['try'] += 1
    sim_to_B_lst, success_B = get_pair_for_X(r['B_annotations'], r['B_verb'], r['B_img'], r_images, r, data_split, distractors_cache_by_keys, can_be_similar_to_sol=False)
    for item in sim_to_B_lst:
        if 'img_name' in item:
            r_images = r_images.union(item)
    sim_to_C_lst, success_C = get_pair_for_X(r['C_annotations'], r['C_verb'], r['C_img'], r_images, r, data_split, distractors_cache_by_keys, can_be_similar_to_sol=True)
    if success_B and success_C:
        counter_found_match['success_all'] += 1
    if success_B and success_C:
        B_distractors_solveable = get_solveable_distractors(data_split, sim_to_B_lst, r)
        C_distractors_solveable = get_solveable_distractors(data_split, sim_to_C_lst, r)
        r['B_distractors'] = B_distractors_solveable
        r['C_distractors'] = C_distractors_solveable
        return B_distractors_solveable, C_distractors_solveable
    else:
        counter_found_match['fails'] += 1
        return None, None


def get_solveable_distractors(data_split, distractors_data, r):
    d_annotations = deepcopy(r['D_annotations'])
    d_annotations['verb'] = r['D_verb']
    for distractor in distractors_data:
        shared_items = {k: d_annotations[k] for k in d_annotations if
                        k in distractor and d_annotations[k] == distractor[k]}
        assert len(shared_items) != len(d_annotations)
    distractors_solveable = [x for x in distractors_data if
                             solve_analogy_all_frames_given_distractor(r, data_split, x['img_name'])]
    global distractors_stats
    distractors_stats['available'] += len(distractors_data)
    distractors_stats['solveable'] += len(distractors_solveable)
    return distractors_solveable


def get_pair_for_X(annotations, verb, img, r_images, r, data_split, distractors_cache_by_keys, can_be_similar_to_sol):
    success_X = False
    diff_key = r['different_key']
    all_D_annotations = data_split[r['D_img']]
    frames_D = all_D_annotations['frames']
    verb_D = all_D_annotations['verb']
    ann_diff_key = annotations[diff_key] if diff_key != 'verb' else verb

    items_singles, items_tuples, got_tuple = get_items_by_tuples_heuristic(ann_diff_key, annotations, diff_key,
                                                            distractors_cache_by_keys,
                                                            r_images, verb)

    if got_tuple:
        iteration_lists_len['chosen'].append(len(items_tuples))
    else:
        iteration_lists_len['chosen'].append(len(items_singles))
    max_sim = 0
    relevant_candidates_dict = {}
    if got_tuple:
        sim_to_X_lst, success_X = get_data_based_on_cand_list(ann_diff_key, annotations, can_be_similar_to_sol,
                                                                        frames_D, verb_D, diff_key, got_tuple, img,
                                                                        items_tuples, max_sim, r_images,
                                                                        relevant_candidates_dict, success_X, verb)
        if not success_X:
            sim_to_X_lst, success_X = get_data_based_on_cand_list(ann_diff_key, annotations, can_be_similar_to_sol,
                                                                            frames_D, verb_D, diff_key, got_tuple, img,
                                                                            items_singles, max_sim, r_images,
                                                                            relevant_candidates_dict, success_X, verb)
    else:
        sim_to_X_lst, success_X = get_data_based_on_cand_list(ann_diff_key, annotations, can_be_similar_to_sol,
                                                                        frames_D, verb_D, diff_key, got_tuple, img,
                                                                        items_singles, max_sim, r_images,
                                                                        relevant_candidates_dict, success_X, verb)

    return sim_to_X_lst, success_X


def get_data_based_on_cand_list(ann_diff_key, annotations, can_be_similar_to_sol, frames_D, verb_D, diff_key,
                                got_tuple, img, items_with_existing_tup, max_sim, r_images, relevant_candidates_dict,
                                success_X, verb):
    random.shuffle(items_with_existing_tup)
    for cand_data in items_with_existing_tup:
        if cand_data['img_name'] in r_images:
            continue
        if diff_key != 'verb' and (diff_key not in cand_data or ann_diff_key not in cand_data[diff_key]):
            # raise Exception(f"BUG")
            continue
        elif diff_key == 'verb' and cand_data['verb'] != ann_diff_key:
            # raise Exception(f"BUG")
            continue

        cand_frame_cpy = deepcopy(cand_data)
        annotations_cpy = deepcopy(annotations)
        annotations_cpy['verb'] = verb

        clash_with_d, cand_sim_with_input, cand_sim_with_solution = is_cand_clashes_with_D(annotations_cpy, can_be_similar_to_sol,
                                                                             cand_frame_cpy, diff_key, frames_D,
                                                                             got_tuple, verb_D)
        global clash_with_d_counter
        if clash_with_d:
            clash_with_d_counter['true'] += 1
            continue
        clash_with_d_counter['false'] += 1
        # if len(relevant_candidates_dict) < MAX_NUM_CANDIDATES or cand_sim_with_input > max_sim:
        if cand_sim_with_input >= max_sim:
            max_sim = cand_sim_with_input
            cand_frame_cpy = deepcopy(cand_data)
            cand_frame_cpy['cand_sim_with_input'] = cand_sim_with_input
            cand_frame_cpy['cand_sim_with_sol'] = cand_sim_with_solution
            if cand_frame_cpy['img_name'] in relevant_candidates_dict:
                if cand_frame_cpy['cand_sim_with_input'] > relevant_candidates_dict[cand_frame_cpy['img_name']][
                    'cand_sim_with_input']:
                    relevant_candidates_dict[cand_frame_cpy['img_name']] = cand_frame_cpy
            else:
                relevant_candidates_dict[cand_frame_cpy['img_name']] = cand_frame_cpy
        # if got_tuple is False and len(relevant_candidates_dict) >= MAX_NUM_CANDIDATES:
        #     break
    relevant_candidates_dict_list = list(relevant_candidates_dict.values())
    relevant_candidates_sorted = sorted(relevant_candidates_dict_list, key=lambda x: x['cand_sim_with_input'],
                                        reverse=True)
    if len(relevant_candidates_sorted) >= 1: # min candidates
        success_X = True
        sim_to_X_lst = relevant_candidates_sorted[:MAX_NUM_CANDIDATES]
        if len(set([x['img_name'] for x in sim_to_X_lst])) < len(sim_to_X_lst):
            raise Exception(f"ERROR")
        global cand_sim_with_input_lst
        cand_sim_with_input_lst += [x['cand_sim_with_input'] for x in sim_to_X_lst]
        if debug_plot_candidates:
            plot_distractors(img, annotations, verb, diff_key, sim_to_X_lst)
    else:
        sim_to_X_lst = []
    return sim_to_X_lst, success_X


def is_cand_clashes_with_D(annotations_cpy, can_be_similar_to_sol, cand_frame_cpy, diff_key, frames_D, got_tuple,
                           verb_D):
    del cand_frame_cpy[diff_key]
    del cand_frame_cpy['img_name']
    if diff_key in annotations_cpy:
        del annotations_cpy[diff_key]

    clash_with_d = False
    for d_annotations in frames_D:
        d_annotations_cpy = deepcopy(d_annotations)
        d_annotations_cpy['verb'] = verb_D
        if diff_key in d_annotations_cpy:
            del d_annotations_cpy[diff_key]

        cand_sim_with_input = get_dict_sim(cand_frame_cpy, annotations_cpy, eliminate_place=True)
        # if SPLIT == 'train' and got_tuple is False:
        if got_tuple is False:
            if cand_sim_with_input > 0:
                # print(f"is single, cand_sim_with_input: {cand_sim_with_input}")
                print(cand_frame_cpy)
                print(annotations_cpy)
                global cases_with_pos_sim
                cases_with_pos_sim += 1
                # raise Exception
                # return None, None, False
                clash_with_d = True
        cand_sim_with_solution = get_dict_sim(cand_frame_cpy, d_annotations_cpy, eliminate_place=True)

        if can_be_similar_to_sol is False and cand_sim_with_solution > 0:
            # continue
            clash_with_d = True
        elif can_be_similar_to_sol is True and cand_sim_with_solution == 1:
            global cases_with_filtered_solution
            cases_with_filtered_solution += 1
            # continue
            clash_with_d = True
    return clash_with_d, cand_sim_with_input, cand_sim_with_solution


def get_items_by_tuples_heuristic(ann_diff_key, annotations, diff_key, distractors_cache_by_keys,
                                  r_images, verb):
    annotations = deepcopy(annotations)
    annotations['verb'] = verb
    keys_that_are_not_diff_key = [k for k in annotations.keys() if k != diff_key]
    items_with_existing_tup_singles = distractors_cache_by_keys[diff_key][ann_diff_key]
    concat_items_with_existing_tup_tuples = []
    got_tuple = False
    for k in keys_that_are_not_diff_key:
        key_pair_key_option_1 = "_".join((k, diff_key))
        key1_option_1 = key_pair_key_option_1.split("_")[0]
        key2_option_1 = key_pair_key_option_1.split("_")[1]
        key_pair_val_op1 = "_".join([annotations[key1_option_1], annotations[key2_option_1]])

        key_pair_key_option_2 = "_".join((diff_key, k))
        key1_option_2 = key_pair_key_option_2.split("_")[0]
        key2_option_2 = key_pair_key_option_2.split("_")[1]
        key_pair_val_op2 = "_".join([annotations[key1_option_2], annotations[key2_option_2]])

        if (key_pair_key_option_1 in distractors_cache_by_keys and key_pair_val_op1 in distractors_cache_by_keys[key_pair_key_option_1]) or (key_pair_key_option_2 in distractors_cache_by_keys and key_pair_val_op2 in distractors_cache_by_keys[
            key_pair_key_option_2]):
            if key_pair_key_option_1 in distractors_cache_by_keys and  key_pair_val_op1 in distractors_cache_by_keys[key_pair_key_option_1]:
                key_pair_key = key_pair_key_option_1
                key_pair_val = key_pair_val_op1
            elif key_pair_key_option_2 in distractors_cache_by_keys and key_pair_val_op2 in distractors_cache_by_keys[key_pair_key_option_2]:
                key_pair_key = key_pair_key_option_2
                key_pair_val = key_pair_val_op2
            items_with_existing_tup = distractors_cache_by_keys[key_pair_key][key_pair_val]
            if len(items_with_existing_tup) > 0:
                concat_items_with_existing_tup_tuples += items_with_existing_tup
            # print(f"Tuples, {len(items_with_existing_tup)}")
    global iteration_lists_len
    iteration_lists_len['single'].append(len(items_with_existing_tup_singles))
    iteration_lists_len['tuple'].append(len(concat_items_with_existing_tup_tuples))
    global choose_tuple
    concat_items_with_existing_tup_tuples_not_intersected = [x for x in concat_items_with_existing_tup_tuples if x['img_name'] not in r_images]
    if len(concat_items_with_existing_tup_tuples_not_intersected) >= 1:
        choose_tuple['true'] += 1
        got_tuple = True
    else:
        choose_tuple['false'] += 1
    return items_with_existing_tup_singles, concat_items_with_existing_tup_tuples_not_intersected, got_tuple


if __name__ == '__main__':
    for split_file_name in split_to_files[SPLIT]:
        print(f"Extracting sim distractors {split_file_name}")
        main(split_file_name)

    print("Done")
    # global didnt_find_verb
    print(counter_found_match)