import json
import numpy as np
import os

import pandas as pd

from dataset.config import imsitu_path

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


    df_filtered_before_bbox['diff_item_A_str_first_bbox_proportion'] = df_filtered_before_bbox.apply(lambda r: calculate_size_proportions(r['diff_item_A_str_first_bbox'], r['A_img_size']), axis=1)
    df_filtered_before_bbox['diff_item_B_str_first_bbox_proportion'] = df_filtered_before_bbox.apply(lambda r: calculate_size_proportions(r['diff_item_B_str_first_bbox'], r['B_img_size']), axis=1)


    df_filtered = df_filtered_before_bbox[df_filtered_before_bbox.apply(lambda r: r_is_above_thresh_or_none(r),axis=1)]

    for c in ['A_data', 'B_data']:
        df_filtered[c] = df_filtered[c].apply(lambda x: json.dumps(x))

    df_filtered.to_csv(AB_matches_filtered_visual)
    print(f"Filtered from {len(df_filtered_before_bbox)} to {len(df_filtered)}, wrote to {AB_matches_filtered_visual}")
    filtered_amount = str(round((len(df_filtered_before_bbox) - len(df_filtered)) / len(df_filtered) * 100, 3)) + "%"
    print(f"filtered_amount: {filtered_amount}")


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