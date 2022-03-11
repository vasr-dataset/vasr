import json
import os
from collections import Counter
from copy import deepcopy

import pandas as pd

from utils.PairFilter import PairsFilter, empty_clusters, final_reject
from utils.utils import imsitu_path, AB_matches_path, columns_to_serialize, SPLIT, AB_matches_filtered_textual

imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
nouns = imsitu["nouns"]

def main():
    df = read_full_df()

    df = initial_filters(df)

    pairs_for_key_dict = create_legit_pairs_by_rules(df)

    df_filtered = filter_by_legit_pairs_and_sample(df, pairs_for_key_dict)

    dump_filtered_AB_pairs(df_filtered)


def initial_filters(df):
    df_filtered_before_filter = deepcopy(df)
    data_split = json.load(open(os.path.join(imsitu_path, f"{SPLIT}.json")))
    df_filtered = df
    df_filtered['frames_with_single_diff'] = df_filtered.apply(lambda r: get_a_b_diffs_set_str(r, data_split), axis=1)
    df_filtered['num_frames_with_single_diff'] = df_filtered['frames_with_single_diff'].apply(lambda x: len(x))
    print(df_filtered['num_frames_with_single_diff'].value_counts())
    df_filtered = df_filtered[df_filtered['num_frames_with_single_diff'] <= 2]  # >91% of the data
    print(
        f"*** multiple AB changes started with {len(df_filtered_before_filter)} and received {len(df_filtered)}, which is {round(len(df_filtered) / len(df_filtered_before_filter) * 100, 2)}")
    df = df_filtered
    different_keys_to_filter = set(df['different_key'].values)
    different_keys_to_filter = [x for x in different_keys_to_filter if x not in ['place', 'source', 'destination']]
    len_start_df = len(df)
    print(df['different_key'].value_counts())
    df = df[df['different_key'].isin(different_keys_to_filter)]
    len_df_different_key = len(df)
    df = df[df['keys'].apply(lambda keys: 'coagent' not in keys)]
    len_df_without_co_agent = len(df)
    df = df[
        df.apply(lambda r: not words_lists_intersect(r['diff_item_A_str'], r['diff_item_B_str'], r['different_key']),
                 axis=1)]
    len_df_without_intersecting_words_list = len(df)
    df_length_stats = {'len_start_df': len_start_df, 'len_df_different_key': len_df_different_key,
                       'len_df_without_co_agent': len_df_without_co_agent,
                       'len_df_without_intersecting_words_list': len_df_without_intersecting_words_list}
    print(df_length_stats)
    return df


def read_full_df():
    df = pd.read_csv(AB_matches_path)
    for c in columns_to_serialize:
        if c in df.columns:
            df[c] = df[c].apply(json.loads)

    df['diff_item_A_str_first'] = df['diff_item_A_str'].apply(lambda l: l[0] if type(l) == list else l)
    df['diff_item_B_str_first'] = df['diff_item_B_str'].apply(lambda l: l[0] if type(l) == list else l)

    return df


def create_legit_pairs_by_rules(df):
    pairs_filter = PairsFilter()

    def sort_pair(x):
        return tuple(list(sorted((tuple((x[0], x[1]))))) + list(sorted(tuple((x[2], x[3])))))

    different_keys_to_filter = set(df['different_key'].values)
    pairs_for_key_dict = {}
    for k in different_keys_to_filter:
        df_different_key = df[df['different_key'] == k]
        print(f'For k {k}, len(df): {len(df_different_key)}')
        changed_pairs_counter_wn_for_key = Counter([tuple(sort_pair(tuple(x))) for x in df_different_key[pairs_filter.keys].values])
        changed_pairs_wn_for_key = [x[0] for x in changed_pairs_counter_wn_for_key.most_common()]
        legit_pairs_for_k = []
        for t_idx, t in enumerate(changed_pairs_wn_for_key):
            is_legit_k_chagnge = pairs_filter.is_legit_k_chagnge(k, t)

            if is_legit_k_chagnge:
                # if man->monkey, then monkey->man
                t_reverse = (t[1], t[0], t[3], t[2])
                legit_pairs_for_k.append(t)
                legit_pairs_for_k.append(t_reverse)

        print(f"*** for key {k}, started with {len(changed_pairs_wn_for_key) * 2}, filtered to {len(legit_pairs_for_k)}")
        initial_pairs_support = sum({k:v for k,v in changed_pairs_counter_wn_for_key.items()}.values())
        legit_pairs_support = sum({k:v for k,v in changed_pairs_counter_wn_for_key.items() if k in legit_pairs_for_k}.values())
        pairs_stats = {'initial_pairs': len(changed_pairs_wn_for_key) * 2, 'legit_pairs': len(legit_pairs_for_k),
                       'initial_pairs_support': initial_pairs_support, 'legit_pairs_support': legit_pairs_support}
        print((k, pairs_stats))
        pairs_for_key_dict[k] = legit_pairs_for_k

    print(f'filtered pairs stats')
    print({k: len(v) for k,v in pairs_for_key_dict.items()})

    print(f"Empty clusters top 100:")
    print(Counter(empty_clusters).most_common(100))

    print(f"Final reject top 100:")
    final_reject_strs = Counter([(x['diff_item_A_str_first'], x['diff_item_B_str_first']) for x in final_reject])
    print(final_reject_strs.most_common(100))

    pairs_filter.print_filter_stats()

    return pairs_for_key_dict


def filter_by_legit_pairs_and_sample(df, pairs_for_key_dict):
    print(f"filter_by_legit_pairs_and_sample...")
    different_keys_to_filter = set(df['different_key'].values)
    df_filtered_relevant_keys = df[df['different_key'].isin(different_keys_to_filter)]
    df_filtered_relevant_keys['change_triplet'] = df_filtered_relevant_keys[['different_key', 'diff_item_A_str_first', 'diff_item_B_str_first']].apply(lambda r: tuple(r), axis=1)

    change_triplets = []
    for diff_key, diff_key_pairs in pairs_for_key_dict.items():
        for pair in diff_key_pairs:
            change_triplets.append((diff_key, *pair[:2]))
    change_triplets_set = set(change_triplets)

    df_examples_for_poc = df_filtered_relevant_keys[df_filtered_relevant_keys['change_triplet'].isin(change_triplets_set)]
    print(f"Filtered by pairs. Got {len(df_examples_for_poc)} from {len(df_filtered_relevant_keys)}")

    print(f'Different keys value counts')
    print(df_examples_for_poc['different_key'].value_counts())

    return df_examples_for_poc


def dump_filtered_AB_pairs(df_filtered):
    for c in columns_to_serialize:
        if c in df_filtered.columns:
            df_filtered[c] = df_filtered[c].apply(json.dumps)
    df_filtered.to_csv(AB_matches_filtered_textual, index=False)
    print(f"Dumped df at length {len(df_filtered)} to {AB_matches_filtered_textual}")
    print("Done")



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
                    a_b_frames_with_single_diffs.append((k, v1, v2))
    return set(a_b_frames_with_single_diffs)


if __name__ == '__main__':
    main()