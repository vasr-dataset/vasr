import inspect
import os
from collections import Counter, defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import json
from nltk.corpus import wordnet as wn

from dataset.utils.PairFilter import PairsFilter, empty_clusters, final_reject
from dataset.config import imsitu_path, imsitu_images_path, AB_matches_path, \
    columns_to_serialize, \
    AB_matches_no_dups_path, SPLIT, AB_matches_filtered_dev_pairs_path, \
    AB_matches_filtered_textual, swig_path

from dataset.utils.visualization import visualize_pair

different_data_top_items = ['agent', 'item', 'verb', 'tool', 'destination', 'object',
                         'source', 'target', 'victim', 'food', 'vehicle', 'coagent', 'surface',
                         'substance', 'agentpart', 'instrument', 'container', 'contact',
                         'obstacle', 'goalitem', 'addressee', 'sliceditem', 'brancher', 'start',
                         'path', 'goods', 'student', 'focus', 'decomposer']
different_keys_to_filter = different_data_top_items
DEV_ANALOGIES_PAIRS = False
print(f"DEV_ANALOGIES: {DEV_ANALOGIES_PAIRS}, handing NO-DUPS file.")
imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
nouns = imsitu["nouns"]

def main():
    imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
    nouns = imsitu["nouns"]
    verbs = imsitu['verbs']
    if DEV_ANALOGIES_PAIRS:
        df = get_df_no_dups()
    else:
        df = read_full_df()

    df['diff_item_A_str_first'] = df['diff_item_A_str'].apply(lambda l: l[0] if type(l) == list else l)
    df['diff_item_B_str_first'] = df['diff_item_B_str'].apply(lambda l: l[0] if type(l) == list else l)

    df_filtered_before_filter = deepcopy(df)
    data_split = json.load(open(os.path.join(swig_path, f"{SPLIT}.json")))
    df_filtered = df
    df_filtered['frames_with_single_diff'] = df_filtered.apply(lambda r: get_a_b_diffs_set_str(r, data_split), axis=1)
    df_filtered['num_frames_with_single_diff'] = df_filtered['frames_with_single_diff'].apply(lambda x: len(x))
    print(df_filtered['num_frames_with_single_diff'].value_counts())
    df_filtered = df_filtered[df_filtered['num_frames_with_single_diff'] <= 2]  # >91% of the data
    # plot_pairs(df_filtered_before_filter, df_filtered)
    print(f"*** multiple AB changes started with {len(df_filtered_before_filter)} and received {len(df_filtered)}, which is {round(len(df_filtered)/len(df_filtered_before_filter) * 100, 2)}")
    df = df_filtered

    # specific_df = df.query('diff_item_A_str_first == "jockey" and diff_item_B_str_first == "horseman"').iloc[0]
    # print_most_common_different_keys(specific_df)

    len_start_df = len(df)
    print(df['different_key'].value_counts())
    df = df[df['different_key'].isin(different_keys_to_filter)]
    len_df_different_key = len(df)
    df = df[df['keys'].apply(lambda keys: 'coagent' not in keys)]
    len_df_without_co_agent = len(df)
    df = df[df.apply(lambda r: not words_lists_intersect(r['diff_item_A_str'], r['diff_item_B_str'], r['different_key']), axis=1)]
    len_df_without_intersecting_words_list = len(df)
    df_length_stats = {'len_start_df': len_start_df, 'len_df_different_key': len_df_different_key,
                       'len_df_without_co_agent': len_df_without_co_agent, 'len_df_without_intersecting_words_list': len_df_without_intersecting_words_list}
    print(df_length_stats)

    # df[df['diff_item_A_str_first'] == 'leopard']['A_data'].apply(lambda x: x['A']['agent'])
    # offset = str(wn_x.offset()).zfill(8) + '-' + wn_x.pos()
    pairs_for_key_dict = create_legit_pairs_by_rules(df, nouns, verbs)

    df_filtered = filter_by_legit_pairs_and_sample(df, pairs_for_key_dict)

    if not DEV_ANALOGIES_PAIRS:
        dump_filtered_AB_pairs(df_filtered)



def plot_pairs(df_filtered_before_bbox, df_filtered_before_bbox_after_taking_only_few_cases):
    the_filtered_data_really_bad = df_filtered_before_bbox[df_filtered_before_bbox['num_frames_with_single_diff'] > 6]
    the_filtered_data_zero_changes_sample = df_filtered_before_bbox[df_filtered_before_bbox['num_frames_with_single_diff'] == 0].sample(10)
    the_filtered_data_maybe_ambiguious = df_filtered_before_bbox[
        df_filtered_before_bbox['num_frames_with_single_diff'] == 3]
    the_filtered_data_really_bad_sample = the_filtered_data_really_bad.sample(5)
    the_filtered_data_maybe_ambiguious_sample = the_filtered_data_maybe_ambiguious.sample(5)

    print(f"Zero Changes")
    for _, p in the_filtered_data_zero_changes_sample.sample(5).iterrows():
        print(f"frames_with_single_diff: {len(p['frames_with_single_diff'])}")
        print(p['frames_with_single_diff'])
        visualize_pair(p, plot_annotations=False)

    print(f"Data Kept")
    for _, p in df_filtered_before_bbox_after_taking_only_few_cases.sample(5).iterrows():
        print(f"frames_with_single_diff: {len(p['frames_with_single_diff'])}")
        print(p['frames_with_single_diff'])
        visualize_pair(p, plot_annotations=False)

    print(f"\n\nREALLY BAD")
    for _, p in the_filtered_data_really_bad_sample.iterrows():
        print(p['frames_with_single_diff'])
        visualize_pair(p, plot_annotations=False)
    print(f"\n\nMaybe Ambiguious")
    for _, p in the_filtered_data_maybe_ambiguious_sample.iterrows():
        print(p['frames_with_single_diff'])
        visualize_pair(p, plot_annotations=False)


def plot_by_query(r):
    # r = df.query('diff_item_A_str_first == "glass"').iloc[0][['A_img', 'B_img', 'diff_item_A_str_first', 'diff_item_B_str_first']]
    img_A = cv2.imread(os.path.join(imsitu_images_path, r['A_img']))[:,:,::-1]
    img_B = cv2.imread(os.path.join(imsitu_images_path, r['B_img']))[:,:,::-1]
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img_A)
    axs[0].set_title(r['diff_item_A_str_first'])
    axs[1].imshow(img_B)
    axs[1].set_title(r['diff_item_B_str_first'])
    plt.show()

def read_full_df():
    # print(f"--- Taking head 10K ---")
    # df = pd.read_csv(AB_matches_path)
    # df = df.sample(10000)
    df = pd.read_csv(AB_matches_path)
    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.loads)
    return df


def get_df_no_dups():
    if os.path.exists(AB_matches_no_dups_path):
        df = pd.read_csv(AB_matches_no_dups_path)
        for c in columns_to_serialize:
            df[c] = df[c].apply(json.loads)
    else:
        print(f"no_dups not exists, creating")
        df = read_full_df()
        df = df.drop_duplicates(subset=['different_key', 'diff_item_A', 'diff_item_B'])
        for c in columns_to_serialize:
            if c in df.columns:
                df[c] = df[c].apply(json.dumps)
        df.to_csv(AB_matches_no_dups_path, index=False)
    return df


def create_legit_pairs_by_rules(df, nouns, verbs):
    pairs_filter = PairsFilter()

    def sort_pair(x):
        return tuple(list(sorted((tuple((x[0], x[1]))))) + list(sorted(tuple((x[2], x[3])))))

    pairs_for_key_dict = {}
    for k in different_keys_to_filter:
        df_different_key = df[df['different_key'] == k]
        print(f'For k {k}, len(df): {len(df_different_key)}')
        # changed_pairs_counter_for_key = Counter([tuple(sorted(tuple(x))) for x in df_different_key[relevant_keys].values])
        #### ALL KEYS TO OBSERVE  - changed_pairs_counter_for_key.keys()
        changed_pairs_counter_wn_for_key = Counter([tuple(sort_pair(tuple(x))) for x in df_different_key[pairs_filter.keys].values])
        changed_pairs_wn_for_key = [x[0] for x in changed_pairs_counter_wn_for_key.most_common()]
        legit_pairs_for_k = []
        for t_idx, t in enumerate(changed_pairs_wn_for_key):
            # if t_idx > 100:
            #     continue
            # if t_idx == 469:
            #     print("H")
            is_legit_k_chagnge = pairs_filter.is_legit_k_chagnge(k, t)
            if DEV_ANALOGIES_PAIRS:
                print(f"t_idx: {t_idx}/{len(changed_pairs_wn_for_key)}, {t[:2], is_legit_k_chagnge}")
                # x, y = t[:2]
                # x_v = verbs[x]
                # y_v = verbs[y]
                # print(f"t_idx: {t_idx}/{len(changed_pairs_wn_for_key)}, {t[:2], is_legit_k_chagnge}\nx_v: {x_v}, y_v: {y_v}\n\n")

        # print()
            if is_legit_k_chagnge:
                # if man->monkey, then monkey->man
                t_reverse = (t[1], t[0], t[3], t[2])
                legit_pairs_for_k.append(t)
                legit_pairs_for_k.append(t_reverse)

        # pairs_for_key_dict[k] = [x[0] for x in changed_pairs_counter_for_key.most_common(whitelist_num_changes)]
        print(f"*** for key {k}, started with {len(changed_pairs_wn_for_key) * 2}, filtered to {len(legit_pairs_for_k)}")
        initial_pairs_support = sum({k:v for k,v in changed_pairs_counter_wn_for_key.items()}.values())
        legit_pairs_support = sum({k:v for k,v in changed_pairs_counter_wn_for_key.items() if k in legit_pairs_for_k}.values())
        pairs_stats = {'initial_pairs': len(changed_pairs_wn_for_key) * 2, 'legit_pairs': len(legit_pairs_for_k),
                       'initial_pairs_support': initial_pairs_support, 'legit_pairs_support': legit_pairs_support}
        print((k, pairs_stats))
        pairs_for_key_dict[k] = legit_pairs_for_k

    # pprint(pairs_for_key_dict)
    print(f'filtered pairs stats')
    print({k: len(v) for k,v in pairs_for_key_dict.items()})

    print(f"Empty clusters top 100:")
    print(Counter(empty_clusters).most_common(100))

    print(f"Final reject top 100:")
    final_reject_strs = Counter([(x['diff_item_A_str_first'], x['diff_item_B_str_first']) for x in final_reject])
    print(final_reject_strs.most_common(100))

    pairs_filter.print_filter_stats()

    return pairs_for_key_dict

def take_only_filtered_agents_places_and_verbs(r, pairs_for_key_dict_agent_place_verb):
    return True

def filter_by_legit_pairs_and_sample(df, pairs_for_key_dict):
    print(f"filter_by_legit_pairs_and_sample...")
    df_filtered_relevant_keys = df[df['different_key'].isin(different_keys_to_filter)]
    df_filtered_relevant_keys['change_triplet'] = df_filtered_relevant_keys[['different_key', 'diff_item_A_str_first', 'diff_item_B_str_first']].apply(lambda r: tuple(r), axis=1)

    change_triplets = []
    for diff_key, diff_key_pairs in pairs_for_key_dict.items():
        for pair in diff_key_pairs:
            change_triplets.append((diff_key, *pair[:2]))
    change_triplets_set = set(change_triplets)
    # assert len(change_triplets) == len(change_triplets_set)

    df_examples_for_poc = df_filtered_relevant_keys[df_filtered_relevant_keys['change_triplet'].isin(change_triplets_set)]
    print(f"Filtered by pairs. Got {len(df_examples_for_poc)} from {len(df_filtered_relevant_keys)}")

    # df_examples_for_poc_orig = pd.DataFrame()
    # for diff_key, diff_key_pairs in pairs_for_key_dict.items():
    #     for pair in diff_key_pairs:
    #         df_with_changed_pair = df_filtered_relevant_keys.query(
    #             f'diff_item_A_str_first == "{pair[0]}" and diff_item_B_str_first == "{pair[1]}"')
    #         """ The sample should include the most varied items of agent and place."""
    #
    #         # sampled_examples_for_changed_pair = sample_examples_for_pair_by_random_sample(MAX_EXAMPLES_FROM_EACH_AB_PAIR, df_with_changed_pair)
    #         # sampled_examples_for_changed_pair = sample_examples_for_pair_by_maximizing_agent_place_verb_pairs(df_with_changed_pair, diff_key)
    #         df_examples_for_poc_orig = pd.concat([df_examples_for_poc_orig, df_with_changed_pair])

    print(f'Different keys value counts')
    print(df_examples_for_poc['different_key'].value_counts())

    return df_examples_for_poc


def sample_examples_for_pair_by_maximizing_agent_place_verb_pairs(relevant_df, diff_key):
    pairs_occ_reversed = relevant_df[['A_agent', 'A_verb']].value_counts().iloc[::-1]
    if diff_key in ['agent', 'verb']:
        pairs_occ_reversed_threshold = pairs_occ_reversed.median()
    else:
        pairs_occ_reversed_threshold = pairs_occ_reversed.quantile(0.95)
    df_examples_for_agent_place_verb_pairs = pd.DataFrame()
    for agent_place_pair, agent_place_verb_pair_occs in pairs_occ_reversed.iteritems():
        sample_from_pair = int(min(pairs_occ_reversed_threshold, agent_place_verb_pair_occs))
        relevant_df_agent_place_verb = relevant_df.query(
            f'A_agent == "{agent_place_pair[0]}" and A_place == "{agent_place_pair[1]}" and A_verb == "{agent_place_pair[2]}"')
        relevant_df_agent_place_verb_sample = relevant_df_agent_place_verb.sample(sample_from_pair)
        df_examples_for_agent_place_verb_pairs = pd.concat(
            [df_examples_for_agent_place_verb_pairs, relevant_df_agent_place_verb_sample])
    return df_examples_for_agent_place_verb_pairs


def sample_examples_for_pair_by_random_sample(MAX_EXAMPLES_FROM_EACH_AB_PAIR,
                                  relevant_df):
    if len(relevant_df) > MAX_EXAMPLES_FROM_EACH_AB_PAIR:
        relevant_df_sampled = relevant_df.sample(MAX_EXAMPLES_FROM_EACH_AB_PAIR)
    else:
        relevant_df_sampled = relevant_df
    return relevant_df_sampled

def dump_filtered_AB_pairs(df_filtered):
    for c in columns_to_serialize:
        if c in df_filtered.columns:
            df_filtered[c] = df_filtered[c].apply(json.dumps)
    if DEV_ANALOGIES_PAIRS:
        df_filtered.to_csv(AB_matches_filtered_dev_pairs_path, index=False)
    else:
        df_filtered.to_csv(AB_matches_filtered_textual, index=False)
    print(f"Dumped df at length {len(df_filtered)} to {AB_matches_filtered_textual}")
    print("Done")


def print_most_common_different_keys(df):
    # total_sum = sum(df['different_key'].value_counts().values)
    # different_key_pct = df['different_key'].value_counts().apply(lambda x: str(round(x / total_sum * 100, 3)) + "%")
    NUM_MOST_COMMON = 30
    for different_key in df['different_key'].value_counts().keys()[:NUM_MOST_COMMON]:
        if different_key == 'verb':
            different_key_list_A = list(df[df['different_key'] == different_key]['diff_item_A_str'].values)
            different_key_list_B = list(df[df['different_key'] == different_key]['diff_item_B_str'].values)
        else:
            different_key_list_A = list(
                df[df['different_key'] == different_key]['diff_item_A_str'].apply(lambda l: l[0]).values)
            different_key_list_B = list(
                df[df['different_key'] == different_key]['diff_item_B_str'].apply(lambda l: l[0]).values)
        different_key_counter = Counter(different_key_list_A + different_key_list_B)
        print(different_key)
        print(different_key_counter.most_common(NUM_MOST_COMMON))

def words_lists_intersect(l1, l2, diff_key):
    if diff_key == 'verb':
        return False
    if len(set(l1).intersection(set(l2))) > 0:
        return True
    for w1 in l1:
        if any(w2 in w1 for w2 in l2 if len(w2.split(" ")) > 1):
            return True
    for w2 in l2:
        if any(w1 in w2 for w1 in l1 if len(w1.split(" ")) > 1):
            return True
    return False

def get_a_b_diffs_set_str(r, data_split):
    image_a, image_b = r['A_img'], r['B_img']
    a_b_frames_with_single_diffs = []
    for a_frame in data_split[image_a]['frames']:
        a_frame.update({'verb': data_split[image_a]['verb']})
        for b_frame in data_split[image_b]['frames']:
            b_frame.update({'verb': data_split[image_b]['verb']})
            shared_key_but_not_val = {k: b_frame[k] for k in b_frame if k in a_frame and b_frame[k] != a_frame[k]}
            if len(shared_key_but_not_val) == 1:
                shared_key_but_not_val_key = list(shared_key_but_not_val.keys())[0]
                if shared_key_but_not_val_key != 'place':
                    k = shared_key_but_not_val_key
                    if shared_key_but_not_val_key != 'verb':
                        if b_frame[shared_key_but_not_val_key] == '':
                            v2 = ''
                        else:
                            v2 = nouns[b_frame[shared_key_but_not_val_key]]['gloss'][0]
                        if a_frame[shared_key_but_not_val_key] == '':
                            v1 = ''
                        else:
                            v1 = nouns[a_frame[shared_key_but_not_val_key]]['gloss'][0]
                    else:
                        v1, v2 = a_frame[shared_key_but_not_val_key], b_frame[shared_key_but_not_val_key]
                    # a_b_frames_with_single_diffs.append((a_frame, b_frame))
                    a_b_frames_with_single_diffs.append((k, v1, v2))
    return set(a_b_frames_with_single_diffs)

def check_frames_diff(first_frame, second_frame):
    diff = []
    a_b_diff = dict(set(first_frame.items()) - set(second_frame.items()))
    b_a_diff = dict(set(second_frame.items()) - set(first_frame.items()))
    # Checking that the diff of A B and B A is same
    for key in (a_b_diff.keys() & b_a_diff.keys()):
        diff.append({'feature': key, 'first_frame_diff': a_b_diff[key], 'second_frame_diff': b_a_diff[key]})
    return diff

if __name__ == '__main__':
    main()