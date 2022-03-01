# ------------------------------Imports--------------------------------
import random
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utills import get_split
import argparse
import os
from utills import save_model, dump_test_info, dump_train_info, calculate_accuracy
from config import TRAIN, DEV, TRAIN_RESULTS_PATH, TEST, FEW_SHOT_DATA_SAMPLES, MODELS_MAP
from models.backend import BackendModel
from models.trainable import BaselineModel

# ------------------------------Constants--------------------------------

device_ids = [0, 1, 2, 3]
model_description_options = {

    # (A,B,C) --> D
    'concatenation',
    # (C+(B-A)) --> D
    'arithmetics'
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------Arguments--------------------------------


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr', help='learning rate', default=0.001, type=float)
    parser.add_argument('-bz', '--batch_size', default=128, type=int)
    parser.add_argument('--model_description', default="similar_to_b", help=f'options: {model_description_options}', type=str)
    parser.add_argument('-ne', '--n_epochs', default=3, type=int)
    parser.add_argument('-s', '--split', default='random', help='Path to save the results as csv')
    parser.add_argument('-rs', '--result_suffix', default="", required=False, help='suffix to add to results name')

    parser.add_argument('--model_backend_type', default='vit', help=f"{set(MODELS_MAP.keys())}", required=False)

    parser.add_argument("--debug", action='store_const', default=False, const=True)
    parser.add_argument("--multi_gpu", action='store_const', default=False, const=True)
    parser.add_argument("--test_model", action='store_const', default=False, const=True)
    parser.add_argument('--load_epoch', default=2)
    parser.add_argument("--few_shot_experiments", action='store_const', help='few-shot experiments', default=False, const=True)
    parser.add_argument("--train_on_silver", action='store_const', default=False, const=True)
    parser.add_argument("--cheap_model", action='store_const', default=False, const=True)
    parser.add_argument("--small_test", action='store_const', default=False, const=True)
    args = parser.parse_args()

    if args.multi_gpu:
        print(f"Multiplying batch_size by # GPUs: {len(device_ids)}")
        initial_batch_size = args.batch_size
        args.batch_size *= len(device_ids)
        print(f"initial_batch_size: {initial_batch_size}, new batch_size: {args.batch_size}")
    return args


# ------------------------------DataLoader--------------------------------

class Loader(Dataset):
    def __init__(self, data, backend_model):
        self.data = data
        self.backend_model = backend_model

    def __getitem__(self, index):
        row = self.data.iloc[index]
        label = torch.FloatTensor([row.label])

        label = label.to(device)
        images_to_load = {'A': row.A_img, 'B': row.B_img, 'C': row.C_img}
        # input_imgs = {k: self.backend_model.load_and_process_img(v) for k, v in images_to_load.items()}
        input_imgs = list(images_to_load.values())

        return input_imgs, row.option, label

    # def __getitem__(self, index):
    #     row = self.data.iloc[index]
    #
    #     candidates = eval(row.candidates) + [row.D_img]
    #     random.shuffle(candidates)
    #     candidates = np.array(candidates)
    #     label = torch.from_numpy(np.where(candidates == row.D_img)[0])
    #     label = label.to(device)
    #     images_to_load = {'A': row.A_img, 'B': row.B_img, 'C': row.C_img}
    #
    #     # input_imgs = {k: self.backend_model.load_and_process_img(v) for k, v in images_to_load.items()}
    #     # candidate_imgs = [self.backend_model.load_and_process_img(x) for x in candidates]
    #     input_imgs = list(images_to_load.values())
    #     candidate_imgs = list(candidates)
    #
    #     return input_imgs, candidate_imgs, label

    def __len__(self):
        return len(self.data)


# ------------------------------Code--------------------------------

def get_experiment_dir(args):
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)

    model_dir_path = os.path.join(TRAIN_RESULTS_PATH,
                                  f"model_{args.model_description}_backend_{args.model_backend_type}_{args.backend_version.replace('/', '-')}_{args.split}")

    if args.few_shot_experiments:
        model_dir_path += f"_few_shot_{args.few_shot_items}_items"

    if args.train_on_silver:
        model_dir_path += f"_train_on_silver"

    if args.debug:
        model_dir_path += "_DEBUG"
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    json.dump(args.__dict__, open(os.path.join(model_dir_path, 'args.json'), 'w'))
    return model_dir_path


def test(backend_model, baseline_model, data, loss_fn):
    """
    Defines the parameters for the test loop and runs it

    Parameters
    ----------
    backend_model : BackendModel is the feature extraction model (e.g., VIT)
    baseline_model :(nn.Module) The baseline model
    data :(dict) contains the train, dev and test data
    loss_fn : Loss function

    """
    print('*** testing ***')
    test_dataset = Loader(data[TEST], backend_model)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    model_dir_path = get_experiment_dir(args)
    model_path = os.path.join(model_dir_path, f'epoch_{args.load_epoch}.pth')
    print(f"Loading model (epoch_{args.load_epoch}) from {model_path}")
    assert os.path.exists(model_path)
    baseline_model.load_state_dict(torch.load(model_path))
    baseline_model = baseline_model.eval()
    test_loop(args=args, model=baseline_model, test_loader=test_loader, loss_fn=loss_fn, test_df=data[TEST])


def train(backend_model, baseline_model, data, loss_fn):
    """
    Defines the parameters for the train loop and runs it

    Parameters
    ----------
    backend_model : BackendModel is the feature extraction model (e.g., VIT)
    baseline_model :(nn.Module) The baseline model
    data :(dict) contains the train, dev and test data
    loss_fn : Loss function

    """
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr, weight_decay=0.0001)
    train_dataset = Loader(data[TRAIN], backend_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    dev_dataset = Loader(data[DEV], backend_model)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    train_loop(args=args, model=baseline_model, optimizer=optimizer, train_loader=train_loader, dev_loader=dev_loader, loss_fn=loss_fn,
               n_epoch=args.n_epochs)


def train_loop(args, model, optimizer, train_loader, dev_loader, loss_fn, n_epoch):
    """

    Parameters
    ----------
    args : (argparse.Namespace) arguments
    model :(nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader :(DataLoader) for the train set
    dev_loader :(DataLoader) for the dev set
    loss_fn : Loss function
    n_epoch :(int) epoch number

    """
    all_losses = {TRAIN: [], DEV: []}
    all_dev_accuracy = []
    model_dir_path = get_experiment_dir(args)
    print(f"model_dir_path: {model_dir_path}")

    for epoch in tqdm(range(n_epoch)):
        epoch_train_losses = train_epoch(loss_fn, model, optimizer, train_loader, epoch)
        epoch_dev_losses, epoch_dev_accuracy, _, _ = test_epoch(loss_fn, model, dev_loader, epoch)
        all_losses[TRAIN].append(epoch_train_losses)
        all_losses[DEV].append(epoch_dev_losses)
        all_dev_accuracy.append(epoch_dev_accuracy)

        dev_accuracy_list = dump_train_info(args, model_dir_path, all_losses, all_dev_accuracy, epoch=epoch)
        save_model(model_dir_path, epoch, model, dev_accuracy_list)


def test_loop(args, model, test_loader, loss_fn, test_df):
    """
    Runs the test loop on the given test set
    Parameters
    ----------
    args :  (argparse.Namespace) arguments
    model :(nn.Module) The baseline model
    test_loader :(DataLoader)
    loss_fn : Loss function
    test_df : (DataFrame) contain information the test set

    """
    all_losses = {TEST: []}
    all_test_accuracy = []
    model_dir_path = get_experiment_dir(args)

    epoch_test_losses, epoch_test_accuracy, predictions, labels = test_epoch(loss_fn, model, test_loader, args.load_epoch)
    all_losses[TEST].append(epoch_test_losses)
    all_test_accuracy.append(epoch_test_accuracy)

    test_df['predictions'] = predictions
    test_df['labels'] = labels

    dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch=args.load_epoch)


nn.BCEWithLogitsLoss()


def train_epoch(loss_fn, model, optimizer, train_loader, epoch):
    """
    Runs training on a single epoch
    Parameters
    ----------
    loss_fn : Loss function
    model : (nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader : (DataLoader)
    epoch : (int) epoch number

    Returns
    -------
    The epoch losses

    """
    model.train()
    epoch_train_losses = []

    with tqdm(enumerate(train_loader), total=len(train_loader)) as epochs:
        epochs.set_description(f'Training epoch {epoch}, model {model.model_description}, split: {args.split}')

        for batch_idx, batch_data in epochs:

            # Forward pass
            input_img, options, label = batch_data
            out = model(input_img, options).squeeze()

            if args.debug:
                if batch_idx > 5:
                    break
            y = label.squeeze()
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_fn(out, y)
            epoch_train_losses.append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()

    return epoch_train_losses


def test_epoch(loss_fn, model, dev_loader, epoch):
    """
    Tests the model on a single epoch using the given dev_loader

    Parameters
    ----------
    loss_fn : Loss function
    model : (nn.Module) The baseline model
    dev_loader : (DataLoader)
    epoch : (int) epoch number

    Returns
    -------
    The epoch: losses,accuracy,model's prediction, labels

    """

    model.eval()
    epoch_dev_losses = []
    epoch_dev_accuracy = []
    all_predictions = []
    all_labels = []

    for batch_idx, batch_data in tqdm(enumerate(dev_loader), total=len(dev_loader), desc=f'Testing epoch {epoch}...'):

        with torch.no_grad():

            input_img, options, label = batch_data
            out = model(input_img, options).squeeze()
        y = label.squeeze()

        if args.debug:
            if batch_idx > 5:
                break
        loss = loss_fn(out, y)
        accuracy, predictions, labels = calculate_accuracy(out, y)
        epoch_dev_losses.append(loss.item())
        epoch_dev_accuracy.append(accuracy)
        all_predictions += predictions
        all_labels += labels

    return epoch_dev_losses, epoch_dev_accuracy, all_predictions, all_labels


def create_new_df(df, sufix):
    res = []
    for i, row in df.iterrows():
        candidates = eval(row.candidates) + [row.D_img]
        for cand in candidates:
            res.append({'A_img': row.A_img, 'B_img': row.B_img, 'C_img': row.C_img, 'D_img': row.D_img, 'option': cand,
                        'label': float(cand == row.D_img)})
    df = pd.DataFrame(res)
    df.to_csv(f'new_dis_{sufix}.csv')
    return df


def main():
    data = get_split(args)
    # data[TRAIN] = create_new_df(data[TRAIN],TRAIN)
    # data[DEV] = create_new_df(data[DEV],DEV)
    data[TRAIN] = pd.read_csv('new_dis_train.csv')
    data[DEV] = pd.read_csv('new_dis_dev.csv')

    backend_model = BackendModel(args)
    baseline_model = BaselineModel(backend_model, args)
    baseline_model = baseline_model.to(device)
    print(f"Checking baseline model cuda: {next(baseline_model.parameters()).is_cuda}")
    if args.multi_gpu:
        baseline_model = nn.DataParallel(baseline_model)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # pos_weights = np.array([4, 1])
    # weights = torch.as_tensor(pos_weights, dtype=torch.float)
    weights = [4.0]
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    if args.test_model:
        test(backend_model, baseline_model, data, loss_fn)
    else:
        train(backend_model, baseline_model, data, loss_fn)
        args.test_model = True
        test(backend_model, baseline_model, data, loss_fn)


if __name__ == '__main__':
    args = get_args()
    print(f"args")
    print(args)
    setattr(args, 'backend_version', MODELS_MAP[args.model_backend_type])

    if args.debug:
        print(f"*** DEBUG MODE ***")

    elif args.few_shot_experiments:
        test_all = args.test_model
        for few_shot_items in FEW_SHOT_DATA_SAMPLES:
            setattr(args, 'few_shot_items', few_shot_items)
            if not test_all:
                args.test_model = False
            main()
    else:
        main()
