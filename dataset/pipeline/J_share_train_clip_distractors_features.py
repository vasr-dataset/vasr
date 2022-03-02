import json
import pandas as pd
import os

from dataset.config import distractors_split_path, columns_to_serialize



def main():
    all_clip_features_path = os.path.join(distractors_split_path, 'analogies_train_full_distractors_with_clip_features.csv')
    all_clip_features = pd.read_csv(all_clip_features_path)

    train_ood_before_clip_features_path = os.path.join(distractors_split_path, 'analogies_train_ood_distractors_before_clip_filter.csv')
    train_ood_before_clip_features_path_out = os.path.join(distractors_split_path, 'analogies_train_ood_distractors_with_clip_features.csv')
    train_ood_before_clip_features = pd.read_csv(train_ood_before_clip_features_path)

    train_full_at_size_oof_before_clip_features_path = os.path.join(distractors_split_path, 'analogies_train_full_at_size_of_ood_distractors_before_clip_filter.csv')
    train_full_at_size_oof_before_clip_features_path_out = os.path.join(distractors_split_path, 'analogies_train_full_at_size_of_ood_distractors_with_clip_features.csv')
    train_full_at_size_oof_before_clip_features = pd.read_csv(train_full_at_size_oof_before_clip_features_path)

    print(f"Read 3 dataframes, 1 with all CLIP features ({len(all_clip_features)}), and two without ({len(train_ood_before_clip_features)}, {len(train_full_at_size_oof_before_clip_features)}), we need to add the clip features to it")

    all_clip_features.rename(columns={"B_distractors_data": "B_distractors_data_clip_features", "C_distractors_data": "C_distractors_data_clip_features"}, inplace=True)

    all_columns_except_dictrators = [c for c in list(train_ood_before_clip_features.columns) if "distractors_data" not in c]

    train_ood_before_clip_features_merged = pd.merge(all_clip_features, train_ood_before_clip_features, on=all_columns_except_dictrators)
    train_full_at_size_oof_before_clip_features_merged = pd.merge(all_clip_features, train_full_at_size_oof_before_clip_features, on=all_columns_except_dictrators)

    for c in ['B_distractors_data', 'C_distractors_data']:
        train_ood_before_clip_features_merged[c] = train_ood_before_clip_features_merged[f'{c}_clip_features']
        train_full_at_size_oof_before_clip_features_merged[c] = train_full_at_size_oof_before_clip_features_merged[f'{c}_clip_features']

    cols_to_drop = [c for c in list(train_ood_before_clip_features_merged.columns) + list(train_full_at_size_oof_before_clip_features_merged.columns) if "Unnamed" in c]
    train_ood_before_clip_features_merged.drop(columns=cols_to_drop, inplace=True)
    train_full_at_size_oof_before_clip_features_merged.drop(columns=cols_to_drop, inplace=True)

    train_ood_before_clip_features_merged['B_distractors_data'] = train_ood_before_clip_features_merged['B_distractors_data_clip_features']
    train_ood_before_clip_features_merged['C_distractors_data'] = train_ood_before_clip_features_merged['C_distractors_data_clip_features']

    train_full_at_size_oof_before_clip_features_merged['B_distractors_data'] = train_full_at_size_oof_before_clip_features_merged['B_distractors_data_clip_features']
    train_full_at_size_oof_before_clip_features_merged['C_distractors_data'] = train_full_at_size_oof_before_clip_features_merged['C_distractors_data_clip_features']

    serialize_cols_load(all_clip_features, train_full_at_size_oof_before_clip_features, train_ood_before_clip_features)

    serialize_cols_dump(all_clip_features, train_full_at_size_oof_before_clip_features, train_ood_before_clip_features)

    print(f"Dumping df at len {len(train_ood_before_clip_features_merged)} to {train_ood_before_clip_features_path_out}")
    train_ood_before_clip_features_merged.to_csv(train_ood_before_clip_features_path_out)

    print(f"Dumping df at len {len(train_full_at_size_oof_before_clip_features_merged)} to {train_full_at_size_oof_before_clip_features_path_out}")
    train_full_at_size_oof_before_clip_features_merged.to_csv(train_full_at_size_oof_before_clip_features_path_out)

    print(f"Columns")
    print(train_ood_before_clip_features.columns)
    print(train_full_at_size_oof_before_clip_features.columns)

    print("Done")


def serialize_cols_dump(all_clip_features, train_full_at_size_oof_before_clip_features, train_ood_before_clip_features):
    for c in columns_to_serialize:
        for df in [all_clip_features, train_ood_before_clip_features, train_full_at_size_oof_before_clip_features]:
            if c in df.columns:
                df[c] = df[c].apply(json.dumps)


def serialize_cols_load(all_clip_features, train_full_at_size_oof_before_clip_features, train_ood_before_clip_features):
    for c in columns_to_serialize:
        for df in [all_clip_features, train_ood_before_clip_features, train_full_at_size_oof_before_clip_features]:
            if c in df.columns:
                if c in ['B_distractors_data', 'C_distractors_data']:
                    df[c] = df[c].apply(lambda x: json.loads(str(x).replace('nan', 'NaN')))
                else:
                    df[c] = df[c].apply(json.loads)
    print(f"Finished serialization")


if __name__ == '__main__':
    main()