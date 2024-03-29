# ------------------------------Imports--------------------------------

import torch
import numpy as np
import os
import pandas as pd
import pickle
from models_config import TRAIN, TEST, DEV, SPLIT_PATH

# ------------------------------Code--------------------------------

def calculate_accuracy(out_prob, y):
    prob = torch.softmax(out_prob, dim=1)
    out_np = prob.detach().cpu().numpy()
    labels_np = y.detach().cpu().numpy()
    accuracy = (np.argmax(out_np, 1) == labels_np).mean()
    predictions = [float(x) for x in np.argmax(out_np, 1)]
    labels = [float(x) for x in labels_np]
    return accuracy, predictions, labels


def save_model(model_dir_path, epoch, model, dev_accuracy_list):
    out_p = os.path.join(model_dir_path, f"epoch_{epoch}.pth")
    print(f"Saving model path to... {out_p}")
    torch.save(model.state_dict(), out_p)
    if dev_accuracy_list[-1] == max(dev_accuracy_list):
        out_p = os.path.join(model_dir_path, f"epoch_BEST.pth")
        print(f"Saving BEST model path to... {out_p}")
        print(dev_accuracy_list)
        torch.save(model.state_dict(), out_p)


def dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch):
    test_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['test'])}
    test_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_test_accuracy)}
    test_info = pd.concat(
        [pd.Series(test_losses_mean, name='test loss'), pd.Series(test_accuracy_mean, name='test accuracy')], axis=1)
    out_p = os.path.join(model_dir_path, f'epoch_{epoch}_test')
    all_losses_out_p = out_p + '_all_losses_test.pickle'
    out_p_test_df = out_p + "_test_df.csv"
    out_p += ".csv"
    test_info.to_csv(out_p)
    test_df.to_csv(out_p_test_df)
    all_losses_and_acc_d = {'all_losses': all_losses, 'all_test_accuracy': all_test_accuracy}
    with open(all_losses_out_p, 'wb') as f:
        pickle.dump(all_losses_and_acc_d, f)
    print(f'Dumping losses {len(test_info)} to {all_losses_out_p}')
    print(test_info)
    print(f'Dumping df {len(test_info)} to {out_p}, and {len(test_df)} to {out_p_test_df}')


def dump_train_info(model_dir_path, all_losses, all_dev_accuracy, epoch):
    train_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['train'])}
    dev_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['dev'])}
    dev_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_dev_accuracy)}
    train_info = pd.concat(
        [pd.Series(train_losses_mean, name='train loss'), pd.Series(dev_losses_mean, name='dev loss'),
         pd.Series(dev_accuracy_mean, name='dev accuracy')], axis=1)
    out_p = os.path.join(model_dir_path, f'epoch_{epoch}')

    all_losses_out_p = out_p + '_all_losses.pickle'
    out_p += ".csv"
    train_info.to_csv(out_p)
    dev_loss_list = list(train_info['dev loss'].values)
    dev_accuracy_list = list(train_info['dev accuracy'].values)
    print(f"*** dev loss ***")
    print(dev_loss_list)
    print(f"*** dev accuracy ***")
    print(dev_accuracy_list)
    all_losses_and_acc_d = {'all_losses': all_losses, 'all_dev_accuracy': all_dev_accuracy}
    with open(all_losses_out_p, 'wb') as f:
        pickle.dump(all_losses_and_acc_d, f)
    print(f'Dumping losses {len(train_info)} to {all_losses_out_p}')
    print(train_info)
    print(f'Dumping df {len(train_info)} to {out_p}')
    return dev_accuracy_list


def get_split(args):
    split = {}

    dir_path = os.path.join(SPLIT_PATH, f'split_{args.split}')

    print(f"dir_path: {dir_path}")
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".csv")]

    print(f"Files splits: {files}")

    for file in files:

        file_path = os.path.join(dir_path, file)

        if args.test_model and TEST in file:
            split[TEST] = pd.read_csv(file_path)
            setattr(args, TEST, file_path)
        else:

            if TRAIN in file:
                train_df = pd.read_csv(file_path)
                split[TRAIN] = train_df
                setattr(args, TRAIN, file_path)

            elif DEV in file:
                split[DEV] = pd.read_csv(file_path)
                setattr(args, DEV, file_path)

            elif TEST in file:
                test_df = pd.read_csv(file_path)
                split[TEST] = test_df
                setattr(args, TEST, file_path)

    for k, v in split.items():

        for c in ['distractors', 'random_candidates']:
            if c in split[k].columns:
                split[k]['candidates'] = split[k][c]

    return split
