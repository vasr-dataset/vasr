import shutil
from tqdm import tqdm

from dataset.config import distractors_split_path, data_path
import pandas as pd
import os

def main():
    feats_dir = os.path.join(distractors_split_path, 'feats_dir')
    if not os.path.exists(feats_dir):
        os.mkdir(feats_dir)
    available_splits = list(set([x.split('_features')[0] for x in os.listdir(distractors_split_path) if "_out_" in x]))
    print(f"available_splits: {available_splits}")
    for split in available_splits:
        print(f"Working on SPLIT: {split}")
        split_files = sorted([x for x in os.listdir(distractors_split_path) if split in x and "_out_" in x])
        print(f"split_files: {len(split_files)}, split_files: {split_files}")
        df = pd.DataFrame()
        for f in tqdm(split_files):
            src = os.path.join(distractors_split_path, f)
            dst = os.path.join(feats_dir, f)
            shutil.move(src, dst)
            curr_df = pd.read_csv(os.path.join(feats_dir, f))
            df = pd.concat([df, curr_df])
        out_name = f'{split}_features.csv'
        out_path = os.path.join(distractors_split_path, out_name)
        print(f"Dumping df at length {len(df)} to {out_path}")
        df.to_csv(out_path)
    print("Done")

if __name__ == '__main__':
    main()