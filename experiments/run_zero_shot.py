# ------------------------------Imports--------------------------------

import os
import os.path
import pandas as pd
import timm
import torch
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
import argparse
from collections import defaultdict
from experiments.config import ZEROSHOT_RESULTS_PATH, TEST_GOLD_PATH, TEST_RANDOM_PATH, MODELS_MAP
from experiments.models.zero_shot import ZeroShot

# ------------------------------Constants--------------------------------

IMG_NAMES = ['A', 'B', 'C', 'D']
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------Arguments--------------------------------


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit', type=str, required=True, help=f'options {MODELS_MAP.keys()}')
    parser.add_argument('--result_suffix', default="", required=False, help='suffix to add to results name')
    parser.add_argument('--split', default='random', help='options : ["random","difficult]')
    args = parser.parse_args()

    return args


# ------------------------------Code--------------------------------

def run_trained_scores(model, row):
    """
    Parameters
    ----------
    model : ZeroShot model
    row : pandas DataFrame row contains information on the inputs images and the candidates
    Returns
    -------
    (dict)  where the key is the image file name of a candidate and the value is  cosine of (candidate,C+(B-A)

    """
    all_image_features = {k: model.preprocess(row[f'{k}_img']) for k in IMG_NAMES}
    all_image_features['candidates'] = [model.preprocess(candidate) for candidate in row.candidates]
    return model.get_analogies_scores(all_image_features, row.candidates)


def load_model(args):
    """
    Parameters
    ----------
    args : (argparse.Namespace) arguments

    Returns
    -------
    ZeroShot model defined by the given arguments

    """
    model_name = args.model
    if model_name not in MODELS_MAP:
        raise Exception(f"Unknown model {model_name}")

    model = timm.create_model(MODELS_MAP[model_name], pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    model = model.to(device)
    zero_shot_model = ZeroShot(model, args.model, core_model_preprocess_func=transform)
    return zero_shot_model


def run_test(args, df):
    """

    Parameters
    ----------
    args : (argparse.Namespace) arguments
    df :(DataFrame) contains information on the inputs images and the candidates

    """
    preds = defaultdict(list)
    best_scores = defaultdict(list)
    num_success = defaultdict(int)
    model = load_model(args)

    model_name = f'analogies_{args.model}'

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Iterating"):
        row.candidates = [row.D_img] + eval(row.candidates)
        scores = run_trained_scores(model, row)
        pred_img = max(scores, key=scores.get)
        preds[model_name].append(pred_img)
        best_scores[model_name].append(scores[pred_img])
        num_success[model_name] += pred_img == row.D_img

    models_accuracy = {}
    for model_name in preds:
        df[f'model_{model_name}_preds'] = pd.Series(preds[model_name], index=df.index)
        df[f'model_{model_name}_scores'] = pd.Series(best_scores[model_name], index=df.index)
        model_acc = len(df.query(f'model_{model_name}_preds == D_img'))
        models_accuracy[model_name] = model_acc

    if not os.path.exists(ZEROSHOT_RESULTS_PATH):
        os.makedirs(ZEROSHOT_RESULTS_PATH)

    print('Accuracy')
    print({k: round(v / len(df) * 100, 1) for k, v in num_success.items()})

    # Store results
    out_p = os.path.join(ZEROSHOT_RESULTS_PATH, f"results_{args.model}_{args.split}")
    if args.result_suffix != '':
        out_p += "_" + args.result_suffix
    out_p += ".csv"
    print(f'Dumping df {len(df)} to {out_p}')
    df.to_csv(out_p)


def get_test_set(args):
    """
    Parameters
    ----------
    args : (argparse.Namespace) arguments

    Returns
    -------
    The test set as DataFrame
    """

    if args.split == 'random':
        test_gold = pd.read_csv(TEST_RANDOM_PATH)
        test_gold['label'] = test_gold['D_img']
        print(f"read test_gold random: {len(test_gold)}")
    else:
        test_gold = pd.read_csv(TEST_GOLD_PATH)
        test_gold['label'] = test_gold['workers_most_common_answer']

    for c in ['distractors', 'random_candidates']:
        if c in test_gold.columns:
            test_gold['candidates'] = test_gold[c]

    print(f"Read GOLD test: {len(test_gold)}")
    return test_gold


def main():
    args = get_args()
    print(args)
    df = get_test_set(args)
    run_test(args, df)


if __name__ == '__main__':
    main()
