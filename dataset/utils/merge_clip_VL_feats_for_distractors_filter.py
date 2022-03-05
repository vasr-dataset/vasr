import shutil
from tqdm import tqdm

from dataset.config import distractors_split_path, data_path
import pandas as pd
import os

def main():
    split = 'dev'
    feats_dir = data_path
    if split == 'train':
        distractors_files_feats = sorted([x for x in os.listdir(distractors_split_path) if 'analogies_train_full_distractors_with_clip_features' in x])
    else:
        distractors_files_feats = sorted([x for x in os.listdir(os.path.join(distractors_split_path,'feats_dir')) if 'analogies_dev_distractors_with_clip_features' in x])
    feats_dir = os.path.join(distractors_split_path, 'feats_dir')
    if not os.path.exists(feats_dir):
        os.mkdir(feats_dir)
    print(distractors_files_feats)
    df = pd.DataFrame()
    for f in tqdm(distractors_files_feats):
        src = os.path.join(distractors_split_path, f)
        dst = os.path.join(feats_dir, f)
        shutil.copyfile(src, dst)
        if split == 'train':
            curr_df = pd.read_csv(os.path.join(distractors_split_path, f))
        else:
            curr_df = pd.read_csv(os.path.join(feats_dir, f))
        df = pd.concat([df, curr_df])
    out_name = f'analogies_{split}_full_distractors_with_clip_features.csv'
    out_path = os.path.join(distractors_split_path, out_name)
    print(f"Dumping df at length {len(df)} to {out_path}")
    df.to_csv(out_path)

if __name__ == '__main__':
    main()