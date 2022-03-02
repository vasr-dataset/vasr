import itertools
import json
import json
import os
import pickle
import random
import time
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from dataset.utils.PairFilter import PairsFilter

from dataset.config import imsitu_path, AB_matches_filtered_path, ABCD_analogies_sampled_path, SPLIT, \
    columns_to_serialize, analogies_plots_path, PARAMS_FOR_SPLIT, AB_matches_dict, swig_path
from dataset.utils.visualization import visualize_analogy

counter_CD_replaces_AB = {'true_new': 0, 'false_new': 0, 'true_old': 0, 'false_old': 0}


def filter_cd_pairs_by_clip(CD_matches_for_AB_pair):
    for match in CD_matches_for_AB_pair:
        vl_feats = match['vl_feats_bbox'] if type(match['vl_feats_bbox']) == dict else match['vl_feats_full_img']
        weighted_srl_score, _, _ = calculate_weighted_srl_score(vl_feats)
        match['weighted_srl_score'] = weighted_srl_score
    sorted_CD_matches_for_AB_pair = sorted(CD_matches_for_AB_pair, key=lambda x: x['weighted_srl_score'], reverse=True)
    top_CD_matches_for_AB_pair = sorted_CD_matches_for_AB_pair[:PARAMS_FOR_SPLIT[SPLIT]['MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM']]
    return top_CD_matches_for_AB_pair


def calculate_weighted_srl_score(vl_feats):
    score_best_srl_for_each_img = (vl_feats['mesh_data_AB']['most_suitable_logit_first'] + vl_feats['mesh_data_BA'][
        'most_suitable_logit_first']) / 2
    score_conditioned_on_both_images = (vl_feats['probs_A_img_logits_per_AB_class_round'][0] +
                                        vl_feats['probs_B_img_logits_per_AB_class_round'][1]) / 2
    score_conditioned_on_both_images_normalized = score_conditioned_on_both_images * 10
    weighted_srl_score = score_best_srl_for_each_img + score_conditioned_on_both_images_normalized
    return weighted_srl_score, score_best_srl_for_each_img, score_conditioned_on_both_images_normalized


def main():
    data_split = json.load(open(os.path.join(swig_path, f"{SPLIT}.json")))

    instances_to_changes_dict = pickle.load(open(AB_matches_dict, 'rb'))

    all_ABCD_matches = []
    print(f"Loaded {len(instances_to_changes_dict)} AB pairs")
    items_with_no_combinations = 0
    CD_matches_for_AB_pair_nums = []
    top_common_images = defaultdict(list)

    pairs_filter = PairsFilter()

    total_idx = 0

    for change_idx, (change_key, change_list) in tqdm(enumerate(instances_to_changes_dict.items()), total=len(instances_to_changes_dict), desc="Iterating instances_to_changes_dict"):
        if change_idx % 1000 == 0:
            print(f"total matches: {len(all_ABCD_matches)}, group: {change_idx}/{len(instances_to_changes_dict)}, items_with_no_combinations: {items_with_no_combinations}, CD_matches_for_AB_pair_nums - average: {np.mean(CD_matches_for_AB_pair_nums)}, items with > 10: {len([x for x in CD_matches_for_AB_pair_nums if x > 10])}")

        if len(change_list) <= 1:
            items_with_no_combinations += 1
            continue

        appearences_for_images = defaultdict(int)

        for ab_idx, ab_data in enumerate(change_list):
            CD_matches_for_AB_pair = []
            random.shuffle(change_list)

            for cd_idx, cd_data in enumerate(change_list):
                # if SPLIT == 'test' and ab_idx >= cd_idx: # Not allowing duplicate pairs AB->CD => CD->AB
                #     continue
                total_idx += 1
                # if total_idx % 10000 == 0 and total_idx > 100:
                #     global counter_times
                #     print(total_idx, {k: np.mean(v) for k,v in counter_times.items()})
                #     print(total_idx, {k: np.mean(v[-500:]) for k,v in counter_times.items()})
                bool_replacable = img_CD_can_replace_AB(ab_data, cd_data, data_split, appearences_for_images, pairs_filter, ab_idx, cd_idx, total_idx)
                # bool_replacable = True
                if bool_replacable:
                    CD_matches_for_AB_pair.append(cd_data)
                    if len(CD_matches_for_AB_pair) > PARAMS_FOR_SPLIT[SPLIT]['MAX_CLIP_CD_FILTER']:
                        break
            CD_matches_for_AB_pair_nums.append(len(CD_matches_for_AB_pair))
            if len(CD_matches_for_AB_pair) > 0:
                if len(CD_matches_for_AB_pair) > PARAMS_FOR_SPLIT[SPLIT]['MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM']:
                    CD_matches_for_AB_pair_best_cd_pairs = filter_cd_pairs_by_clip(CD_matches_for_AB_pair)
                else:
                    CD_matches_for_AB_pair_best_cd_pairs = CD_matches_for_AB_pair
                add_matches_to_analogies_list(ab_data, CD_matches_for_AB_pair_best_cd_pairs, all_ABCD_matches, appearences_for_images)
                # print(f"different images: {len(appearences_for_images)}")
                # print(Counter(appearences_for_images).most_common(5))
                top_common_d = dict([(i, x[1]) for i, x in enumerate(Counter(appearences_for_images).most_common(5))])
                for k, v in top_common_d.items():
                    top_common_images[k].append(v)

    print(f"Dumping final total matches {len(all_ABCD_matches)}, created from {len(list(instances_to_changes_dict.keys()))} AB pairs, ."
          f"\n")

    each_change_support = [len(v) for k,v in instances_to_changes_dict.items()]
    print(f"each_change_support: {np.mean(each_change_support)}")

    print(f"Repeating images:{ {k:np.mean(v) for k,v in top_common_images.items()} }")
    print(f"items_with_no_combinations: {items_with_no_combinations}")

    print(f"counter_last_cond: {counter_last_cond}")
    print(f"counter_CD_replaces_AB: {counter_CD_replaces_AB}")
    all_ABCD_matches_df = pd.DataFrame(all_ABCD_matches)
    for c in columns_to_serialize:
        if c in all_ABCD_matches_df.columns:
            all_ABCD_matches_df[c] = all_ABCD_matches_df[c].apply(lambda x: json.dumps(x))
    print("Value Counts")
    print(all_ABCD_matches_df['different_key'].value_counts())

    agent_num = all_ABCD_matches_df['different_key'].value_counts().loc['agent']
    gbdk = all_ABCD_matches_df.groupby("different_key")
    all_ABCD_matches_df_sampled = pd.DataFrame()
    for group_name, group_df in gbdk:
        if group_name == 'verb':
            verb_desired_size = agent_num * 2
            if verb_desired_size < len(group_df):
                group_df = group_df.sample(verb_desired_size)
        if len(group_df) > 20:
            all_ABCD_matches_df_sampled = pd.concat([all_ABCD_matches_df_sampled, group_df])
    print(f"After sample, total of {len(all_ABCD_matches_df_sampled)}:")
    print(all_ABCD_matches_df_sampled['different_key'].value_counts())
    all_ABCD_matches_df_sampled_no_dups = all_ABCD_matches_df_sampled.drop_duplicates(subset=['A_img','B_img','C_img','D_img'])
    all_ABCD_matches_df_sampled_no_dups.to_csv(ABCD_analogies_sampled_path, index=False)
    print(F"*** After no dups {len(all_ABCD_matches_df_sampled_no_dups)} ***")
    print(f"Dumped analogies to path {ABCD_analogies_sampled_path}")
    print("Done")


def add_matches_to_analogies_list(ab_data, CD_matches_for_AB_pair, all_ABCD_matches, appearences_for_images):
    if len(CD_matches_for_AB_pair) > 1:
        chosen_CD_matches_for_AB_pair = sample_relevant_cds(CD_matches_for_AB_pair, ab_data)
    else:
        chosen_CD_matches_for_AB_pair = CD_matches_for_AB_pair

    for CD_match in chosen_CD_matches_for_AB_pair:
        assert ab_data['diff_item_A'] == CD_match['diff_item_A']
        assert ab_data['diff_item_B'] == CD_match['diff_item_B']
        assert ab_data['different_key'] == CD_match['different_key']
        original_keys_ab = {k: ab_data[k] for k in ab_data.keys() if 'original' in k}
        original_keys_cd = {k.replace("_A_","_C_").replace("_B_","_D_"): CD_match[k] for k in CD_match.keys() if 'original' in k}
        vl_feats_ab = {k + "_AB": ab_data[k] for k in ab_data.keys() if 'vl_feats' in k}
        vl_feats_cd = {k + "_CD": CD_match[k] for k in CD_match.keys() if 'vl_feats' in k}
        ABCD_match = {'A_img': ab_data['A_img'], 'B_img': ab_data['B_img'], 'A_verb': ab_data['A_verb'], 'B_verb': ab_data['B_verb'], 'diff_item_A': ab_data['diff_item_A'], 'diff_item_B': ab_data['diff_item_B'], 'diff_item_A_str_first': ab_data['diff_item_A_str_first'], 'diff_item_B_str_first': ab_data['diff_item_B_str_first'], 'A_annotations': ab_data['A_data']['A'], 'A_annotations_str': ab_data['A_data']['A_str'], 'B_annotations': ab_data['B_data']['B'], 'B_annotations_str': ab_data['B_data']['B_str'],
                      'C_img': CD_match['A_img'], 'D_img': CD_match['B_img'], 'C_verb': CD_match['A_verb'], 'D_verb': CD_match['B_verb'], 'C_annotations': CD_match['A_data']['A'], 'C_annotations_str': CD_match['A_data']['A_str'], 'D_annotations': CD_match['B_data']['B'], 'D_annotations_str': CD_match['B_data']['B_str'], 'different_key': ab_data['different_key'],
                      'A_bounding_box': ab_data['A_data']['A_bounding_box'], 'B_bounding_box': ab_data['B_data']['B_bounding_box'],
                      'C_bounding_box': CD_match['A_data']['A_bounding_box'], 'D_bounding_box': CD_match['B_data']['B_bounding_box'],
                      **original_keys_ab, **original_keys_cd, **vl_feats_ab, **vl_feats_cd}
        all_ABCD_matches.append(ABCD_match)
        appearences_for_images[ABCD_match['A_img']] += 1
        appearences_for_images[ABCD_match['B_img']] += 1
        appearences_for_images[ABCD_match['C_img']] += 1
        appearences_for_images[ABCD_match['D_img']] += 1
        out_p = os.path.join(analogies_plots_path, f'{len(all_ABCD_matches)}.png')
        # visualize_analogy(ABCD_match, out_p)
        # print(out_p)


def sample_relevant_cds(CD_matches_for_AB_pair, ab_data):
    """ Now, we want to take the CD with most distant verb ! """
    MAX_CDS_MATCHES_FOR_AB = PARAMS_FOR_SPLIT[SPLIT]['MAX_CDS_MATCHES_FOR_AB']
    if ab_data['different_key'] == 'verb':
        chosen_CD_matches_for_AB_pair = random.sample(CD_matches_for_AB_pair, MAX_CDS_MATCHES_FOR_AB) if len(
            CD_matches_for_AB_pair) > MAX_CDS_MATCHES_FOR_AB else CD_matches_for_AB_pair
    else:
        # for cd_data in CD_matches_for_AB_pair:
        #     cd_data['verb_similarity_to_AB'] = get_verbs_similarity_for_row(ab_data, cd_data)
        # CD_matches_for_AB_pair_sorted = sorted(CD_matches_for_AB_pair, key=lambda d: d['verb_similarity_to_AB'])

        chosen_CD_matches_for_AB_pair = random.sample(CD_matches_for_AB_pair, MAX_CDS_MATCHES_FOR_AB) if len(
            CD_matches_for_AB_pair) > MAX_CDS_MATCHES_FOR_AB else CD_matches_for_AB_pair
    return chosen_CD_matches_for_AB_pair

counter_last_cond = {'in': 0, 'success': 0, 'b4': 0}
counter_times = {'till_a_is_diff': [], 'a_is_diff': []}


def all_a_c_keys_are_different_except_diff_key(ab_data, cd_data):
    global counter_last_cond
    counter_last_cond['in'] += 1

    A_dict = ab_data['A_data']['A']
    A_dict['verb'] = ab_data['A_verb']

    C_dict = cd_data['A_data']['A']
    C_dict['verb'] = cd_data['A_verb']

    assert ab_data['different_key'] == cd_data['different_key']

    A_dict_cpy = deepcopy(A_dict)
    C_dict_cpy = deepcopy(C_dict)

    del A_dict_cpy[ab_data['different_key']]
    del C_dict_cpy[ab_data['different_key']]
    x, y = A_dict_cpy, C_dict_cpy
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    if len(shared_items) == 0:
    # if len(shared_items) <= 1:
        counter_last_cond['success'] += 1
        return True
    return False



def img_CD_can_replace_AB(ab_data, cd_data, data_split, appearences_for_images, pairs_filter, ab_idx, cd_idx, total_idx):
    """ When searching for CD match, we want C that has the same item as A (or D that has same item as B)
    It includes several conditions:
    1. The image is different.
    2. They have same different item. If the different item is the agent, then A's agent should be as C's agent.
    3. Additional condition - we want the verbs to be *distant* (not similar)
    4. New constraint - C frames can't contain the changed key from A->B.

    diff_data is actually the existing AB. r is the row fow CD candidates.
    """
    start = time.time()
    if ab_data['different_key'] != 'verb':
        if ab_data['A_verb'] == cd_data['A_verb']:
            return False

    all_images = {ab_data['A_img'], ab_data['B_img'], cd_data['A_img'], cd_data['B_img']}
    if len(all_images) < 4:
        return False

    current_occs = [appearences_for_images[img] for img in all_images]
    if max(current_occs) > PARAMS_FOR_SPLIT[SPLIT]['MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR']:
        return False

    if cd_data['different_key'] != 'verb':
        C_frames_at_changed_key = [x[cd_data['different_key']] for x in data_split[cd_data['A_img']]['frames']]
        if ab_data['diff_item_B'] in C_frames_at_changed_key:
            return False

    """ different_key - is the only one identical between A and C
        The other keys should be different by pairs filter """
    global counter_last_cond
    counter_last_cond['b4'] += 1

    # new_bool = A_is_different_than_C_in_the_other_keys_by_equality(ab_data, cd_data)
    new_bool = all_a_c_keys_are_different_except_diff_key(ab_data, cd_data)
    global counter_CD_replaces_AB

    if new_bool:
        counter_CD_replaces_AB['true_new'] += 1
    else:
        counter_CD_replaces_AB['false_new'] += 1

    if not new_bool:
        return False

    return True


def A_is_different_than_C_in_the_other_keys(ab_data, cd_data, pairs_filter):
    global counter_last_cond
    counter_last_cond['in'] += 1
    try:
        agent_pair_tuple, place_pair_tuple, verb_pair_tuple = get_agent_place_verb_tuple(ab_data, cd_data)
    except:
        return False
    if agent_pair_tuple is None:
        return True
    if ab_data['different_key'] == 'verb':
        if not pairs_filter.is_legit_object_change(agent_pair_tuple):
            return False
        if not pairs_filter.is_legit_place_change(place_pair_tuple):
            return False
    elif ab_data['different_key'] == 'place':
        if not pairs_filter.is_legit_object_change(agent_pair_tuple):
            return False
        if not pairs_filter.is_legit_verb_change(verb_pair_tuple):
            return False
    elif ab_data['different_key'] == 'agent':
        if not pairs_filter.is_legit_place_change(place_pair_tuple):
            return False
        if not pairs_filter.is_legit_verb_change(verb_pair_tuple):
            return False
    counter_last_cond['success'] += 1
    return True


def get_agent_place_verb_tuple(ab_data, cd_data):
    A_data = ab_data['A_data']
    C_data = cd_data['A_data']
    if 'agent' not in A_data['A'] or 'agent' not in C_data['A']:
        print(f"No agent!!!")
        print(A_data['A'])
        print(C_data['A'])
        return None, None, None
    A_agent = A_data['A']['agent']
    A_agent_str = A_data['A_str']['agent'][0]
    A_place = A_data['A']['place']
    A_place_str = A_data['A_str']['place'][0] if A_data['A_str']['place'] is not None else None
    C_agent = C_data['A']['agent']
    C_agent_str = C_data['A_str']['agent'][0]
    C_place = C_data['A']['place']
    C_place_str = C_data['A_str']['place'][0] if C_data['A_str']['place'] is not None else None
    A_verb = A_data['A_verb']
    C_verb = C_data['A_verb']
    agent_pair_tuple = (A_agent_str, C_agent_str, A_agent, C_agent)
    place_pair_tuple = (A_place_str, C_place_str, A_place, C_place)
    verb_pair_tuple = (A_verb, C_verb, A_verb, C_verb)
    return agent_pair_tuple, place_pair_tuple, verb_pair_tuple


def A_is_different_than_C_in_the_other_keys_by_existing_pairs(ab_data, cd_data, pairs_for_key_dict_agent_place_verb):
    global counter_last_cond
    counter_last_cond['in'] += 1
    try:
        agent_pair_tuple, place_pair_tuple, verb_pair_tuple = get_agent_place_verb_tuple(ab_data, cd_data)
    except:
        return False
    if ab_data['different_key'] == 'verb':
        if not agent_pair_tuple in pairs_for_key_dict_agent_place_verb['agent']:
            return False
        # if not place_pair_tuple in pairs_for_key_dict_agent_place_verb['place']:
        #     return False
    elif ab_data['different_key'] == 'place':
        if not agent_pair_tuple in pairs_for_key_dict_agent_place_verb['agent']:
            return False
        if not verb_pair_tuple in pairs_for_key_dict_agent_place_verb['verb']:
            return False
    elif ab_data['different_key'] == 'agent':
        # if not place_pair_tuple in pairs_for_key_dict_agent_place_verb['place']:
        #     return False
        if not verb_pair_tuple in pairs_for_key_dict_agent_place_verb['verb']:
            return False
    counter_last_cond['success'] += 1
    return True

def get_verbs_similarity_for_row(ab_data, cd_data):
    A_verb, B_verb = ab_data['A_verb'], ab_data['B_verb']
    C_verb, D_verb = cd_data['A_verb'], cd_data['B_verb']
    if A_verb == C_verb:
        max_sim_metrics = {'similiarity_wup': 1.0}
    else:
        max_sim_metrics = get_verbs_similarity(A_verb, C_verb)
    return max_sim_metrics['similiarity_wup']

def A_is_different_than_C_in_the_other_keys_by_equality(ab_data, cd_data):
    global counter_last_cond
    counter_last_cond['in'] += 1
    try:
        agent_pair_tuple, place_pair_tuple, verb_pair_tuple = get_agent_place_verb_tuple(ab_data, cd_data)
    except:
        return False
    if ab_data['different_key'] == 'verb':
        if not agent_pair_tuple or len(set(agent_pair_tuple)) == 2:
            return False
        if not place_pair_tuple or len(set(place_pair_tuple)) == 2:
            return False
    elif ab_data['different_key'] == 'place':
        if not agent_pair_tuple or len(set(agent_pair_tuple)) == 2:
            return False
        if not verb_pair_tuple or len(set(verb_pair_tuple)) == 1:
            return False
    elif ab_data['different_key'] == 'agent':
        if not place_pair_tuple or len(set(place_pair_tuple)) == 2:
            return False
        if not verb_pair_tuple or len(set(verb_pair_tuple)) == 1:
            return False
    else:
        if not agent_pair_tuple or len(set(agent_pair_tuple)) == 2:
            return False
    counter_last_cond['success'] += 1
    return True

def get_verbs_similarity(A_verb, C_verb):
    A_verb_synsets = [x for x in wn.synsets(A_verb) if x.pos() == 'v']
    C_verb_synsets = [x for x in wn.synsets(C_verb) if x.pos() == 'v']
    sims_metrics = defaultdict(list)
    for A_syn in A_verb_synsets:
        for C_syn in C_verb_synsets:
            similiarity_wup = round(A_syn.wup_similarity(C_syn), 3)
            shortest_path = A_syn.shortest_path_distance(C_syn)
            sims_metrics['similiarity_wup'].append(similiarity_wup)
            if shortest_path:
                sims_metrics['shortest_path'].append(shortest_path)
    max_sim_metrics = {k: max(v) if k != 'shortest_path' else min(v) for k, v in sims_metrics.items()}
    if 'similiarity_wup' not in max_sim_metrics:
        return {'similiarity_wup': 1.0}
    return max_sim_metrics


if __name__ == '__main__':
    main()