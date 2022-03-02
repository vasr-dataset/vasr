import os
import pandas as pd
from dataset.config import data_path

final_blacklist = [['hand', 'finger'], ['arm', 'hand']]

def main():
    remove_blacklist()
    make_random_and_distractor_equal()
    print("Done")


def make_random_and_distractor_equal():
    for split in ['train', 'dev', 'test']:
        distractors_csv_path = os.path.join(data_path, 'split_distractors', f"analogies_distractors_{split}_final.csv")
        random_csv_path = os.path.join(data_path, 'split_random', f"analogies_random_candidates_{split}_final.csv")
        distractors_df = pd.read_csv(distractors_csv_path)
        random_df = pd.read_csv(random_csv_path)
        if len(distractors_df) > len(random_df):
            distractors_df = distractors_df.sample(len(random_df))
            distractors_df.to_csv(distractors_csv_path, index=False)
        elif len(distractors_df) < len(random_df):
            random_df = random_df.sample(len(distractors_df))
            random_df.to_csv(random_csv_path, index=False)
        print(f"split: {split}, random_df: {len(random_df)}, distractors_df: {len(distractors_df)}")


def remove_blacklist():
    distractors_dir_path = os.path.join(data_path, 'split_distractors')
    random_dir_path = os.path.join(data_path, 'split_random')
    distractors_files_final = [f for f in os.listdir(distractors_dir_path) if 'final' in f and f.endswith(".csv")]
    random_files_final = [f for f in os.listdir(random_dir_path) if 'final' in f and f.endswith(".csv")]
    sizes = filter_csv(distractors_dir_path, distractors_files_final, distractors=True)
    filter_csv(random_dir_path, random_files_final, distractors=False, sizes=sizes)


def filter_csv(dir_path, files, distractors, sizes=None):
    sizes = {}
    for f in files:
        csv_p = os.path.join(dir_path, f)
        df = pd.read_csv(csv_p)
        df_before_len = len(df)
        cols_to_drop = [c for c in df.columns if "Unnamed" in c]
        df.drop(columns=cols_to_drop, inplace=True)
        for pair in final_blacklist:
            df = df.query(f'diff_item_A_str_first != "{pair[0]}" and diff_item_B_str_first != "{pair[1]}"')
            df = df.query(f'diff_item_A_str_first != "{pair[1]}" and diff_item_B_str_first != "{pair[0]}"')
        if len(df) < df_before_len:
            df.to_csv(csv_p, index=False)
            print(f"Filtered {df_before_len}->{len(df)} to: {csv_p}")
        sizes[f] = len(df)
    print(sizes)
    return sizes


if __name__ == '__main__':
    main()