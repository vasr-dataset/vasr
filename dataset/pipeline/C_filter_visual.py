import json
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2
import pandas as pd

from dataset.utils.visualization import visualize_pair

imsitu_path = '/data/users/yonatab/analogies/imSitu'
imsitu = json.load(open(os.path.join(imsitu_path, "imsitu_space.json")))
nouns = imsitu["nouns"]
from dataset.config import AB_matches_filtered_textual, AB_matches_filtered_visual, swig_images_path, swig_path, \
    SPLIT, BBOX_PCT_THRESHOLD


def main():

    data_split = json.load(open(os.path.join(swig_path, f"{SPLIT}.json")))
    df_filtered_before_bbox = pd.read_csv(AB_matches_filtered_textual)

    for c in ['A_data', 'B_data']:
        df_filtered_before_bbox[c] = df_filtered_before_bbox[c].apply(lambda x: json.loads(x))

    df_filtered_before_bbox['A_img_height'] = df_filtered_before_bbox['A_img'].apply(lambda x: data_split[x]['height'])
    df_filtered_before_bbox['A_img_width'] = df_filtered_before_bbox['A_img'].apply(lambda x: data_split[x]['width'])
    df_filtered_before_bbox['A_img_size'] = df_filtered_before_bbox.apply(lambda r: r['A_img_height'] * r['A_img_width'], axis=1)
    df_filtered_before_bbox['B_img_height'] = df_filtered_before_bbox['B_img'].apply(lambda x: data_split[x]['height'])
    df_filtered_before_bbox['B_img_width'] = df_filtered_before_bbox['B_img'].apply(lambda x: data_split[x]['width'])
    df_filtered_before_bbox['B_img_size'] = df_filtered_before_bbox.apply(lambda r: r['B_img_height'] * r['B_img_width'], axis=1)
    df_filtered_before_bbox['diff_item_A_str_first_bbox'] = df_filtered_before_bbox.apply(lambda r: get_bbox_of_diff_item(r['A_data']['A_bounding_box'], r['diff_item_A_str_first'], r['different_key']), axis=1)
    df_filtered_before_bbox['diff_item_B_str_first_bbox'] = df_filtered_before_bbox.apply(lambda r: get_bbox_of_diff_item(r['B_data']['B_bounding_box'], r['diff_item_B_str_first'], r['different_key']), axis=1)

    # Format: [591, 202, 655, 382] - (x1, y1, x2, y2)
    # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    df_filtered_before_bbox['diff_item_A_str_first_bbox_proportion'] = df_filtered_before_bbox.apply(lambda r: calculate_size_proportions(r['diff_item_A_str_first_bbox'], r['A_img_size']), axis=1)
    df_filtered_before_bbox['diff_item_B_str_first_bbox_proportion'] = df_filtered_before_bbox.apply(lambda r: calculate_size_proportions(r['diff_item_B_str_first_bbox'], r['B_img_size']), axis=1)

    # plot_debug_bbox_filter()

    df_filtered = df_filtered_before_bbox[df_filtered_before_bbox.apply(lambda r: r_is_above_thresh_or_none(r),axis=1)]

    for c in ['A_data', 'B_data']:
        df_filtered[c] = df_filtered[c].apply(lambda x: json.dumps(x))

    df_filtered.to_csv(AB_matches_filtered_visual)
    print(f"Filtered from {len(df_filtered_before_bbox)} to {len(df_filtered)}, wrote to {AB_matches_filtered_visual}")
    filtered_amount = str(round((len(df_filtered_before_bbox) - len(df_filtered)) / len(df_filtered) * 100, 3)) + "%"
    print(f"filtered_amount: {filtered_amount}")


# def plot_debug_bbox_filter():
    # df_filtered_before_bbox_not_na_A = df_filtered_before_bbox[~df_filtered_before_bbox['diff_item_A_str_first_bbox_proportion'].isna()]
    # df_filtered_before_bbox_not_na_A_below_out = df_filtered_before_bbox_not_na_A[df_filtered_before_bbox_not_na_A['diff_item_A_str_first_bbox_proportion'].apply(lambda x: x <= 0.01)]
    # for idx, (r_idx, r) in enumerate(df_filtered_before_bbox_not_na_A_below_out[['diff_item_A_str_first', 'diff_item_A_str_first_bbox', 'diff_item_A_str_first_bbox_proportion', 'A_img']].sample(20).iterrows()):
    #     if idx > 10:
    #         break
    #     img_p = os.path.join(swig_images_path, r['A_img'])
    #     img = cv2.imread(img_p)
    #     x1, y1, x2, y2 = list(r['diff_item_A_str_first_bbox'])
    #     img_rect = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    #     plt.suptitle((r['diff_item_A_str_first'], r['diff_item_A_str_first_bbox_proportion']))
    #     plt.imshow(img_rect[:, :, ::-1])
    #     plt.show()


def get_a_b_diffs(r, data_split):
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
    return len(a_b_frames_with_single_diffs)


def get_bbox_of_diff_item(bbox_dict, diff_item, different_key):
    if different_key in bbox_dict and bbox_dict[different_key] != [-1, -1, -1, -1]:
        return bbox_dict[different_key]
    return None

def calculate_size_proportions(bbox, img_size):
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    obj_size = width * height
    return round(obj_size / img_size, 2)

def r_is_above_thresh_or_none(r):
    if r['diff_item_A_str_first_bbox_proportion'] is None or r['diff_item_B_str_first_bbox_proportion'] is None or np.isnan(r['diff_item_A_str_first_bbox_proportion']) or np.isnan(r['diff_item_B_str_first_bbox_proportion']):
        return True
    if r['diff_item_A_str_first_bbox_proportion'] > BBOX_PCT_THRESHOLD and r['diff_item_B_str_first_bbox_proportion'] > BBOX_PCT_THRESHOLD:
        return True
    return False


if __name__ == '__main__':
    main()