import json
import numpy as np
import os
import pickle
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from G_find_CD_pairs import calculate_weighted_srl_score
from dataset.config import ABCD_analogies_sampled_path, data_path, columns_to_serialize, test_plots_path
from dataset.utils.visualization import visualize_analogy


def main():
    train_df = pd.read_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_train.csv'))
    testdev_df = pd.read_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_testdev.csv'))
    testdev_with_extracted_feats_path = os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_testdev_extracted_feats.csv')
    print(f"Feats: {testdev_with_extracted_feats_path}")

    print(f"Loaded splits: train: {len(train_df)}, testdev_df: {len(testdev_df)}")

    extract_testdev_jaccard_and_atoms_feats(train_df, testdev_df)
    test_df_gold_before_annotation, dev_df_gold_before_annotation, dev_df_silver, train_df_gold_before_annotation = create_silver_and_gold_splits(testdev_df, train_df)

    dev_df_silver.to_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_dev.csv'),index=False)
    dev_df_gold_before_annotation.to_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_dev_gold_before_annotation.csv'), index=False)
    test_df_gold_before_annotation.to_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_test_gold_before_annotation.csv'),index=False)

    train_df.to_csv(os.path.join(data_path, 'ABCD_matches', f'all_ABCD_matches_rule_based_sampled_train_full.csv'), index=False)

    print(f"Dumped train & dev: train_df: {len(train_df)}, train_df_ood: {len(train_df)}")

    print("Done")


def pair_not_in_test(r, all_test_pairs):
    return (r['diff_item_A'], r['diff_item_B']) not in all_test_pairs and (r['diff_item_B'], r['diff_item_A']) not in all_test_pairs


def extract_testdev_jaccard_and_atoms_feats(train_df, testdev_df):
    json_loads(testdev_df)
    json_loads(train_df)

    atoms_counter, atoms_counter_as_diff_key = calculate_train_atoms_occurences(train_df)

    testdev_df = calc_jaccard_and_atoms_occs(testdev_df, atoms_counter, atoms_counter_as_diff_key)

    json_dumps(testdev_df)


def calculate_train_atoms_occurences(train_df):
    atoms_counter = defaultdict(int)
    atoms_counter_as_diff_key = defaultdict(int)
    for idx, (r_idx, r) in tqdm(enumerate(train_df.iterrows()), total=len(train_df), desc='calculate_train_atoms_occurences'):
        for img_key in ['A_annotations_str', 'B_annotations_str', 'C_annotations_str', 'D_annotations_str']:
            img_key_items = list(r[img_key].items())
            img_key_items.append(('verb', [r[f'{img_key[0]}_verb']]))
            for k, v in img_key_items:
                if type(v) == list:
                    v = v[0]

                if k == r['different_key']:
                    updated_v = r['diff_item_A_str_first'] if img_key[0] in ['A', 'C'] else r['diff_item_B_str_first']
                    atoms_counter_as_diff_key[updated_v] += 1
                else:
                    updated_v = v
                atoms_counter[updated_v] += 1
    return atoms_counter, atoms_counter_as_diff_key


def calc_jaccard_and_atoms_occs(testdev_df, atoms_counter, atoms_counter_as_diff_key):
    all_analogy_difficulty_scores = []
    for idx, (r_idx, r) in tqdm(enumerate(testdev_df.iterrows()), total=len(testdev_df), desc='calc_jaccard_and_atoms_occs'):
        # if idx < 48780:
        #     continue
        keys_jaccard, values_jaccard = calculate_AC_jaccard_distance(r)

        mean_diff_key_occs, max_diff_key_occs = calculate_test_sample_train_occ(atoms_counter, atoms_counter_as_diff_key, r)

        vl_feats_AB = r['vl_feats_bbox_AB'] if type(r['vl_feats_bbox_AB']) == dict else r['vl_feats_full_img_AB']
        score_srl_weighted_AB, score_best_srl_for_each_img_AB, score_conditioned_on_both_images_normalized_AB = calculate_weighted_srl_score(vl_feats_AB)
        vl_feats_CD = r['vl_feats_bbox_CD'] if type(r['vl_feats_bbox_CD']) == dict else r['vl_feats_full_img_CD']
        score_srl_weighted_CD, score_best_srl_for_each_img_CD, score_conditioned_on_both_images_normalized_CD = calculate_weighted_srl_score(vl_feats_CD)
        analogy_difficulty_score = {'keys_jaccard': keys_jaccard, 'values_jaccard': values_jaccard, 'mean_diff_key_occs': mean_diff_key_occs, 'max_diff_key_occs': max_diff_key_occs,
                         'score_srl_weighted_AB': score_srl_weighted_AB, 'score_best_srl_for_each_img_AB': score_best_srl_for_each_img_AB, 'score_conditioned_on_both_images_normalized_AB': score_conditioned_on_both_images_normalized_AB,
                         'score_srl_weighted_CD': score_srl_weighted_CD, 'score_best_srl_for_each_img_CD': score_best_srl_for_each_img_CD, 'score_conditioned_on_both_images_normalized_CD': score_conditioned_on_both_images_normalized_CD
                         }
        all_analogy_difficulty_scores.append(analogy_difficulty_score)

    testdev_df['analogy_difficulty_score'] = all_analogy_difficulty_scores
    return testdev_df


def calculate_test_sample_train_occ(atoms_counter,
                                    atoms_counter_as_diff_key, r):
    atoms_occs_dict = {}
    atoms_occs_diff_key_dict = {}
    for img_key in ['A_annotations_str', 'B_annotations_str', 'C_annotations_str', 'D_annotations_str']:
        img_key_items = list(r[img_key].items())
        img_key_items.append(('verb', [r[f'{img_key[0]}_verb']]))
        for k, v in img_key_items:
            if type(v) == list:
                v = v[0]
            if k == r['different_key']:
                updated_v = r['diff_item_A_str_first'] if img_key[0] in ['A', 'C'] else r['diff_item_B_str_first']
            else:
                updated_v = v

            atom_occ = atoms_counter[updated_v] if updated_v in atoms_counter else 0
            atoms_occs_dict[updated_v] = atom_occ
            if k == r['different_key']:
                atom_occ_dif_key = atoms_counter_as_diff_key[updated_v] if updated_v in atoms_counter_as_diff_key else 0
                atoms_occs_diff_key_dict[updated_v] = atom_occ_dif_key
    diff_key_occs = list(atoms_occs_diff_key_dict.values())
    mean_diff_key_occs, max_diff_key_occs = float(np.mean(diff_key_occs)), max(diff_key_occs)
    return mean_diff_key_occs, max_diff_key_occs


def json_dumps(testdev_df):
    for c in columns_to_serialize:
        if c in testdev_df.columns:
            testdev_df[c] = testdev_df[c].apply(json.dumps)


def json_loads(df):
    for c in columns_to_serialize:
        if c in df.columns:
            if c in ['vl_feats_bbox_AB', 'vl_feats_bbox_CD']:
                df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                df[c] = df[c].apply(json.loads)


def calculate_AC_jaccard_distance(r):
    A_keys = set(r['A_annotations'].keys())
    C_keys = set(r['C_annotations'].keys())
    AC_union = A_keys.union(C_keys)
    AC_intersect = A_keys.intersection(C_keys)
    keys_jaccard = round(len(AC_intersect) / len(AC_union), 2)
    A_annotations_in_intersection = {k: v for k, v in r['A_annotations'].items() if k in AC_intersect}
    C_annotations_in_intersection = {k: v for k, v in r['C_annotations'].items() if k in AC_intersect}
    x, y = A_annotations_in_intersection, C_annotations_in_intersection
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    num_shared_values = len(shared_items)
    values_jaccard = round(num_shared_values / len(AC_union), 2)
    return keys_jaccard, values_jaccard


def create_silver_and_gold_splits(testdev_df, train_df_silver):
    json_loads(testdev_df)

    feats_df = pd.DataFrame.from_records(testdev_df['analogy_difficulty_score'].values)
    testdev_feats_df = pd.concat([testdev_df, feats_df], axis=1)
    testdev_sorted_df = testdev_feats_df.sort_values(['keys_jaccard', 'values_jaccard', 'mean_diff_key_occs', 'max_diff_key_occs', 'score_srl_weighted_AB', 'score_srl_weighted_CD'], ascending=[True, True, True, True, False, False])

    test_df = build_test(testdev_sorted_df)

    NOISE_LEVEL = 0.4
    dev_df_ood_all_desired_size = 3000
    desired_dev_gold_size_after_annotation = 100
    train_df_ood_gold_desired_size = 1000

    dev_df_gold_before_annotation, dev_df_silver, test_df_percentages = build_dev(NOISE_LEVEL,
                                                                                  desired_dev_gold_size_after_annotation,
                                                                                  test_df, testdev_df)

    train_df_gold_before_annotation = build_train(NOISE_LEVEL, dev_df_silver, test_df_percentages,
                                                  train_df_ood_gold_desired_size, train_df_silver)


    print(test_df['different_key'].value_counts())
    relevant_test_keys = list(test_df['different_key'].value_counts().keys())
    test_gold_vc = test_df['different_key'].value_counts()
    test_gold_vc.name = 'Test'

    train_silver_vc = train_df_silver[train_df_silver['different_key'].isin(relevant_test_keys)]['different_key'].value_counts()
    train_silver_vc.name = 'Train'
    dev_silver_vc = dev_df_silver['different_key'].value_counts()
    dev_silver_vc.name = 'Dev'
    silver = pd.DataFrame(pd.concat([train_silver_vc, dev_silver_vc, test_gold_vc],axis=1))

    train_gold_vc = train_df_gold_before_annotation[train_df_gold_before_annotation['different_key'].isin(relevant_test_keys)]['different_key'].value_counts()
    train_gold_vc.name = 'Train'
    dev_gold_vc = dev_df_gold_before_annotation['different_key'].value_counts()
    dev_gold_vc.name = 'Dev'
    gold = pd.DataFrame(pd.concat([train_gold_vc, dev_gold_vc, test_gold_vc],axis=1))

    print(f"Done creating test")

    """
    
        # print(f"Plotting first 100")
    # for idx, (r_idx, r) in enumerate(test_df.sample(100).iterrows()):
    #     out_p = os.path.join(test_plots_path, f'{idx}_aa.png')
    #     visualize_analogy(r, out_p=out_p, plot_annotations=False, return_fig=False, hide_answer=True)
    #     out_p = os.path.join(test_plots_path, f'{idx}_full.png')
    #     visualize_analogy(r, out_p=out_p, plot_annotations=True, return_fig=False, hide_answer=False)

    testdev_feats_df_low_jaccard_sim = testdev_feats_df.sort_values(['keys_jaccard', 'values_jaccard'], ascending=[True, True])
    testdev_feats_df_low_train_atoms_occ = testdev_feats_df.sort_values(['mean_diff_key_occs', 'max_diff_key_occs'], ascending=[True, True])
    testdev_feats_df_best_srl = testdev_feats_df.sort_values(['score_srl_weighted_AB', 'score_srl_weighted_CD'], ascending=[False, False])
    # first_analogy, last_analogy, middle_analogy = get_3_analogies(testdev_feats_df_low_jaccard_sim)
    # first_analogy, last_analogy, middle_analogy = get_3_analogies(testdev_feats_df_low_train_atoms_occ)
    # first_analogy, last_analogy, middle_analogy = get_3_analogies(testdev_feats_df_best_srl)
    # show_analogy(first_analogy)
    # show_analogy(middle_analogy)
    # show_analogy(last_analogy)
    # show_analogy(first_analogy, plot_annotations=True)
    # show_analogy(middle_analogy, plot_annotations=True)
    # show_analogy(last_analogy, plot_annotations=True)
    """


def build_train(NOISE_LEVEL, dev_df_silver, test_df_percentages, train_df_ood_gold_desired_size, train_df_silver):
    required_train_gold_silver_annotations = int(train_df_ood_gold_desired_size / (1 - NOISE_LEVEL))
    train_df_items_to_sample = test_df_percentages.apply(lambda x: int(x * required_train_gold_silver_annotations) + 1)
    train_df_items = []
    for diff_key, diff_key_items_to_sample in train_df_items_to_sample.items():
        train_df_silver_diff_key = train_df_silver[dev_df_silver['different_key'] == diff_key]
        if len(train_df_silver_diff_key) > diff_key_items_to_sample:
            train_df_silver_diff_key = train_df_silver_diff_key.sample(diff_key_items_to_sample)
        train_df_items.append(train_df_silver_diff_key)
    train_df_gold = pd.concat(train_df_items)
    desired_train_gold_size_after_annotation = 1000
    required_train_silver_annotation = int(desired_train_gold_size_after_annotation / (1 - NOISE_LEVEL))
    train_df_gold_before_annotation = train_df_silver.sample(required_train_silver_annotation)
    return train_df_gold_before_annotation


def build_dev(NOISE_LEVEL, desired_dev_gold_size_after_annotation, test_df, testdev_df):
    all_test_images = set(
        [item for sublist in test_df[['A_img', 'B_img', 'C_img', 'D_img']].values for item in sublist])
    dev_df_no_test_images = testdev_df[testdev_df.apply(
        lambda r: all(x not in all_test_images for x in [r['A_img'], r['B_img'], r['C_img'], r['D_img']]), axis=1)]
    all_test_pairs = set([tuple(x) for x in test_df[['diff_item_A', 'diff_item_B']].values])
    dev_df_ood_all = dev_df_no_test_images[
        dev_df_no_test_images.apply(lambda r: pair_not_in_test(r, all_test_pairs), axis=1)]
    test_df_percentages = test_df['different_key'].value_counts().apply(lambda x: x / len(test_df))
    dev_df_items_to_sample = test_df_percentages.apply(lambda x: int(x * desired_dev_gold_size_after_annotation) + 1)
    dev_df_items = []
    for diff_key, diff_key_items_to_sample in dev_df_items_to_sample.items():
        dev_df_ood_all_diff_key = dev_df_ood_all[dev_df_ood_all['different_key'] == diff_key]
        if len(dev_df_ood_all_diff_key) > diff_key_items_to_sample:
            dev_df_ood_all_diff_key = dev_df_ood_all_diff_key.sample(diff_key_items_to_sample)
        dev_df_items.append(dev_df_ood_all_diff_key)
    dev_df_silver = pd.concat(dev_df_items)
    required_dev_silver_annotation = int(desired_dev_gold_size_after_annotation / (1 - NOISE_LEVEL))
    dev_df_gold_before_annotation = dev_df_silver.sample(required_dev_silver_annotation)
    return dev_df_gold_before_annotation, dev_df_silver, test_df_percentages


def build_test(testdev_sorted_df):
    current_test_AB_images = []
    current_test_items = []
    total_gold = 3100  # 750 agent, 750 verb, rest (1500) other keys
    total_verb_agent = int(total_gold / 2)
    budget_verb = total_verb_agent / 2
    budget_agent = total_verb_agent / 2
    total_gold_without_verb = total_gold - budget_verb
    interesting_keys = ['agent', 'verb', 'tool', 'item', 'victim', 'vehicle']
    testdev_sorted_df_relevant_keys = testdev_sorted_df[testdev_sorted_df['different_key'].isin(interesting_keys)]
    for idx, (r_idx, r) in enumerate(testdev_sorted_df_relevant_keys.iterrows()):
        if len(current_test_items) > total_gold_without_verb and r[
            'different_key'] != 'verb':  # break when reaching desired size
            continue
        if len(current_test_items) > total_gold:
            print(f"Breaking at idx: {idx} out of {len(testdev_sorted_df)}")
            break
        if (r['C_img'], r['D_img']) in current_test_AB_images:  # not repeating images
            continue
        if r['different_key'] == 'verb':  # not too many verbs
            if budget_verb <= 0:
                continue
            budget_verb -= 1
        if r['different_key'] == 'agent':  # not too many agents
            if budget_agent <= 0:
                continue
            budget_agent -= 1
        current_test_items.append(r)
        current_test_AB_images.append((r['A_img'], r['B_img']))
    test_df = pd.DataFrame(current_test_items)
    return test_df


def get_3_analogies(df_to_plot):
    first_analogy = df_to_plot.iloc[0]
    middle_analogy = df_to_plot.iloc[int(len(df_to_plot) / 2)]
    last_analogy = df_to_plot.iloc[-1]
    return first_analogy, last_analogy, middle_analogy


def show_analogy(first_analogy, plot_annotations=False):
    analogy = first_analogy
    fig_analogies = visualize_analogy(analogy, plot_annotations=plot_annotations, return_fig=True, hide_answer=False)
    fig_analogies.show()


if __name__ == '__main__':
    main()

