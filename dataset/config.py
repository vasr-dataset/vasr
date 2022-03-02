import io
from copy import deepcopy

import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import random
import os
import numpy as np
from PIL import Image

SPLIT = 'test'  # train, dev, test, testdev
PARAMS_FOR_SPLIT = {'train': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'dev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'testdev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'test': {'MAX_CDS_MATCHES_FOR_AB': 1, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 1, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 4, 'MAX_CLIP_CD_FILTER': 100},}
NUM_CANDIDATES = 4
FONT_SIZE = 16
# BBOX_PCT_THRESHOLD = 0.01
BBOX_PCT_THRESHOLD = 0.02

print(f"*** #### SPLIT: {SPLIT}, PARAMS_FOR_SPLIT: {PARAMS_FOR_SPLIT[SPLIT]} ### ***")

imsitu_path = r'C:\devel\image_analogies\imSitu'
swig_path = r'C:\devel\swig\SWiG_jsons'
swig_images_path = r'C:\devel\swig\images\images_512'

distractors_cache_path = os.path.join(imsitu_path, 'analogies_outputs', 'data', 'aggregate', f"distractors_cache_{SPLIT}.pickle")
aggregate_dir = os.path.join(imsitu_path, 'analogies_outputs', 'data', 'aggregate')
distractors_cache_by_keys_path = os.path.join(aggregate_dir, f"distractors_cache_by_keys_{SPLIT}.pickle")
imsitu_images_path = os.path.join(imsitu_path, 'resized_256')
data_path = os.path.join(imsitu_path, 'analogies_outputs', 'data')
plots_path = os.path.join(imsitu_path, 'analogies_outputs', 'plots')
AB_matches_dir = os.path.join(data_path, 'AB_matches')
AB_matches_path = os.path.join(AB_matches_dir, f'all_AB_matches_{SPLIT}.csv')
AB_matches_no_dups_path = os.path.join(AB_matches_dir, f'all_AB_matches_{SPLIT}_no_dups.csv')
AB_legit_pairs_path = os.path.join(AB_matches_dir, f'AB_pairs_{SPLIT}.pickle')
AB_legit_pairs_no_dups_path = os.path.join(AB_matches_dir, f'AB_pairs_no_dups_{SPLIT}.pickle')
AB_matches_filtered_textual = os.path.join(AB_matches_dir, f'all_AB_matches_filtered_rule_based_filtered_textual_{SPLIT}.csv')
AB_matches_filtered_visual = os.path.join(AB_matches_dir, f'all_AB_matches_filtered_rule_based_filtered_visual_{SPLIT}.csv')

AB_matches_vision_and_language_feats_path = os.path.join(AB_matches_dir, f'all_AB_matches_vision_and_language_feats_{SPLIT}.csv')
AB_matches_objects_no_bbox_feats_path = os.path.join(AB_matches_dir, f'all_AB_matches_objects_no_bbox_vision_and_language_feats_{SPLIT}.csv')
AB_matches_vision_and_language_feats_to_filter = os.path.join(AB_matches_dir, f'all_AB_matches_vision_and_language_feats_to_filter_{SPLIT}.csv')
AB_matches_vision_and_language_feats_to_keep = os.path.join(AB_matches_dir, f'all_AB_matches_vision_and_language_feats_to_keep_{SPLIT}.csv')

AB_matches_filtered_path = os.path.join(AB_matches_dir, f'all_AB_matches_filtered_rule_based_filtered_{SPLIT}.csv')
AB_matches_filtered_dev_pairs_path = os.path.join(AB_matches_dir, f'all_AB_matches_filtered_rule_based_filtered_{SPLIT}_dev_pairs.csv')
AB_matches_dict = os.path.join(AB_matches_dir, f'all_AB_matches_dict_{SPLIT}.pickle')
# AB_matches_filtered_path = os.path.join(data_path, f'all_AB_matches_filtered_rule_based_filtered_AB_NUM_SAMPLE_{NUM_EXAMPLES_FROM_EACH_PAIR}_{SPLIT}.csv')

ABCD_matches_dir = os.path.join(data_path, 'ABCD_matches')

# ABCD_analogies_path = os.path.join(data_path, f'all_ABCD_matches_rule_based_{SPLIT}.csv')
ABCD_analogies_sampled_path = os.path.join(ABCD_matches_dir, f'all_ABCD_matches_rule_based_sampled_{SPLIT}.csv')
ABCD_analogies_with_random_candidates_path = os.path.join(data_path, 'split_random', f'analogies_random_candidates_{SPLIT}_final.csv')
distractors_split_path = os.path.join(data_path, 'split_distractors')
random_split_path = os.path.join(data_path, 'split_random')
ABCD_analogies_with_distractors_path = os.path.join(data_path, 'split_distractors', f'analogies_distractors_{SPLIT}_final.csv')
ab_matches_plots_dir = os.path.join(plots_path, 'AB_matches')
abcd_matches_plots_dir = os.path.join(plots_path, 'ABCD_matches')
ab_matches_plots_path = os.path.join(plots_path, 'AB_matches',  f'AB_matches_rule_based_{SPLIT}')
analogies_plots_path = os.path.join(plots_path, 'ABCD_matches', f'ABCD_matches_rule_based_{SPLIT}')
test_plots_path = os.path.join(plots_path, 'test_data')

analogies_distractors_plots_path = os.path.join(plots_path, 'ABCD_matches', f'ABCD_matches_rule_based_distractors_{SPLIT}')
analogies_distractors_plots_path_turk = os.path.join(plots_path, 'ABCD_matches', f'ABCD_matches_rule_based_distractors_{SPLIT}_mturk.csv')
for d in [data_path, aggregate_dir, distractors_split_path, random_split_path, AB_matches_dir, ABCD_matches_dir, plots_path, ab_matches_plots_dir, abcd_matches_plots_dir, ab_matches_plots_path, analogies_plots_path, analogies_distractors_plots_path, test_plots_path]:
    if not os.path.exists(d):
        os.mkdir(d)

all_train_files = ['all_ABCD_matches_rule_based_sampled_train_full.csv', 'all_ABCD_matches_rule_based_sampled_train_ood.csv', 'all_ABCD_matches_rule_based_sampled_train_full_at_size_of_ood.csv']
all_dev_files = ['all_ABCD_matches_rule_based_sampled_dev.csv']
all_test_files = ['all_ABCD_matches_rule_based_sampled_test.csv']
split_to_files = {'train': all_train_files, 'dev': all_dev_files, 'test': all_test_files}

# columns_to_serialize = {'C_annotations', 'A_str', 'keys', 'A_annotations', 'A_annotations_str', 'B_annotations_str', 'D_annotations_str', 'D_annotations', 'vl_feats_bbox', 'C_annotations_str', 'B_bounding_box', 'A_bounding_box', 'B_data', 'diff_item_A_str', 'B_str', 'vl_feats_full_img', 'A_data', 'B_annotations', 'diff_item_B_str', 'distractors', 'distractors_data', 'distractors_data_and_clip_features', 'B_distractors_data', 'C_distractors_data', 'C_bounding_box', 'D_bounding_box'}
columns_to_serialize = {'C_annotations', 'A_str', 'keys', 'A_annotations', 'A_annotations_str', 'B_annotations_str', 'D_annotations_str', 'D_annotations', 'vl_feats_bbox', 'C_annotations_str', 'B_bounding_box', 'A_bounding_box', 'B_data', 'diff_item_A_str', 'B_str', 'vl_feats_full_img', 'A_data', 'B_annotations', 'diff_item_B_str', 'distractors', 'distractors_data', 'distractors_data_and_clip_features', 'B_distractors_data', 'C_distractors_data',
                        'diff_item_A_str_original', 'diff_item_B_str_original', 'diff_item_C_str_original', 'diff_item_D_str_original',
                        'analogy_difficulty_score', 'vl_feats_bbox_AB', 'vl_feats_full_img_AB', 'vl_feats_bbox_CD', 'vl_feats_full_img_CD'}
VIS_ENLARGE_FACTOR = 2
FINAL_COLS_TRAIN = ['A_img', 'B_img', 'A_verb', 'B_verb', 'diff_item_A', 'diff_item_B',
                    'diff_item_A_str_first', 'diff_item_B_str_first', 'A_annotations',
                    'A_annotations_str', 'B_annotations', 'B_annotations_str', 'C_img',
                    'D_img', 'C_verb', 'D_verb', 'C_annotations', 'C_annotations_str',
                    'D_annotations', 'D_annotations_str', 'different_key', 'distractors']

BAD_IMAGES = ['smashing_96.jpg', 'boarding_155.jpg', 'knocking_90.jpg', 'kissing_171.jpg', 'curtsying_206.jpg', 'adjusting_246.jpg', 'decomposing_205.jpg', 'imitating_110.jpg', 'adjusting_3.jpg', 'planting_128.jpg', 'brawling_61.jpg', 'barbecuing_6.jpg', 'unloading_84.jpg', 'ascending_46.jpg', 'tripping_245.jpg', 'subduing_39.jpg', 'typing_51.jpg', 'assembling_73.jpg', 'applauding_119.jpg', 'catching_208.jpg', 'scratching_156.jpg', 'crowning_51.jpg', 'spearing_35.jpg', 'pouting_219.jpg', 'saluting_260.jpg', 'filming_66.jpg', 'lifting_267.jpg', 'admiring_130.jpg', 'assembling_310.jpg', 'peeing_138.jpg', 'bulldozing_115.jpg', 'yanking_236.jpg', 'clapping_189.jpg', 'fueling_67.jpg', 'colliding_210.jpg']


def get_difference(AB_match_dict, A_data, B_data, str_fmt=False, all_str=False):
    different_key = AB_match_dict['B_data']['different_key']
    if different_key == 'verb':
        diff_item_A = A_data['A_verb']
        diff_item_B = B_data['B_verb']
    else:
        if str_fmt:
            diff_item_A = A_data['A_str'][different_key]
            diff_item_B = B_data['B_str'][different_key]
            if not all_str:
                diff_item_A = diff_item_A[0]
                diff_item_B = diff_item_B[0]
        else:
            diff_item_A = A_data['A'][different_key]
            diff_item_B = B_data['B'][different_key]
    return diff_item_A, diff_item_B, different_key


def get_dict_sim(x, y, eliminate_place=False):
    if eliminate_place:
        x = deepcopy(x)
        y = deepcopy(y)
        if 'place' in x:
            del x['place']
        if 'place' in y:
            del y['place']
    xy_keys = set(x.keys()).union(y.keys())
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    num_shared_values = len(shared_items)
    num_shared_values_divided_by_number_of_keys = round(num_shared_values / len(xy_keys), 2)
    return num_shared_values_divided_by_number_of_keys
