import json
import pickle

import pandas as pd
from tqdm import tqdm

from dataset.config import AB_matches_filtered_path, columns_to_serialize, AB_matches_dict


def main():
    all_AB_matches_df = load_data()

    instances_to_changes_dict = {}

    gbo = all_AB_matches_df.groupby(['different_key', 'diff_item_A', 'diff_item_B'])

    for name, group_df in tqdm(gbo, total=len(gbo), desc='Iterating groups...'):

        group_list = group_df.to_dict('records')
        instances_to_changes_dict[name] = group_list

    pickle.dump(instances_to_changes_dict, open(AB_matches_dict, 'wb'))
    print(f"Dumped {len(instances_to_changes_dict)} items to {AB_matches_dict}, # pairs: {len(gbo)}")

    print("Done")


def load_data():
    all_AB_matches_df = pd.read_csv(AB_matches_filtered_path)
    for c in columns_to_serialize:
        if c in all_AB_matches_df.columns:
            if c in ['vl_feats_bbox', 'vl_feats_full_img']:
                all_AB_matches_df[c] = all_AB_matches_df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
            else:
                all_AB_matches_df[c] = all_AB_matches_df[c].apply(json.loads)
    unnamed_cols = [x for x in all_AB_matches_df.columns if "Unnamed:" in x]
    all_AB_matches_df.drop(columns=unnamed_cols, inplace=True)
    return all_AB_matches_df


if __name__ == '__main__':
    main()