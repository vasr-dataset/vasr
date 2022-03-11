import shutil

from tqdm import tqdm

from utils import AB_matches_dir, SPLIT
import pandas as pd
import os

def main():
    ab_files_feats = sorted([x for x in os.listdir(AB_matches_dir) if '_indices' in x and SPLIT in x and 'all_AB_matches_vision_and_language_feats' in x])
    out_name = f'all_AB_matches_vision_and_language_feats_{SPLIT}.csv'
    merge_and_dump(ab_files_feats, out_name)

    ab_files_feats = sorted([x for x in os.listdir(AB_matches_dir) if '_indices' in x and SPLIT in x and 'all_AB_matches_objects_no_bbox_vision_and_language_feats' in x])
    out_name = f'all_AB_matches_objects_no_bbox_vision_and_language_feats_{SPLIT}.csv'
    merge_and_dump(ab_files_feats, out_name)


def merge_and_dump(ab_files_feats, out_name):
    feats_dir = os.path.join(AB_matches_dir, 'feats_dir')
    if not os.path.exists(feats_dir):
        os.mkdir(feats_dir)
    print(ab_files_feats)
    df = pd.DataFrame()
    for f in tqdm(ab_files_feats):
        src = os.path.join(AB_matches_dir, f)
        curr_df = pd.read_csv(os.path.join(AB_matches_dir, f))
        dst = os.path.join(feats_dir, f)
        shutil.move(src, dst)
        df = pd.concat([df, curr_df])
    out_path = os.path.join(AB_matches_dir, out_name)
    df.to_csv(out_path)
    print(f"Dumped df at length {len(df)} to {out_path}")


if __name__ == '__main__':
    main()