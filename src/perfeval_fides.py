import argparse
import time
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numpy import argmax
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from captum.attr import GradientShap,DeepLift,DeepLiftShap,IntegratedGradients,NoiseTunnel

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score,f1_score, balanced_accuracy_score, accuracy_score, plot_roc_curve, roc_curve, precision_recall_curve

from . import data
from . import models
from . import os_layer
from . import utils


def main(args: argparse.Namespace, log: logging.Logger) -> None:

    raw_path: Path = Path(args.raw_path)
    raw_path_status: Optional[Path] = os_layer.create_dir_if_doesnt_exist(raw_path, log)
    if raw_path_status is None:
        msg: str = f"Something went wrong when creating {raw_path}. Aborting..."
        log.error(msg)
        raise EnvironmentError(msg)
    log.info("Dataset {}".format(args.dataset))


    if args.dataset == "COMPAS":
        X, y, Z = data.loadCOMPAS(raw_path / "compas.csv")
        y = y.values.ravel()

    elif args.dataset == "CENSUS":
        X, y, Z = data.load_census_data(raw_path / 'adult.data')

    elif args.dataset == "LAW":
        X, y, Z = data.load_lawschool_data()

    elif args.dataset == "CREDIT":
        X, y, Z = data.loadCredit(raw_path)

    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)

    log.info("Concatenating sensitive attributes to the main features")
    X = pd.concat([X, Z], axis=1)
    model = models.BinaryNet(X.shape[1])
    model = model.to(args.device)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    le = LabelEncoder()
    y = le.fit_transform(y)
    (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.3, random_state=1337)
    X_train, X_test, y_train, y_test, Z_train, Z_test = X_train.to_numpy(), X_test.to_numpy(), y_train, y_test, Z_train, Z_test

    traindata = Data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor))
    testdata = Data.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=64, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model = utils.train(args.epochs, model, trainloader, testloader, optimizer, args, log)

    # choose data for generating explanations
    (X_adv_train, X_adv_test, y_adv_train, y_adv_test, Z_adv_train, Z_adv_test)  = train_test_split(X_test, y_test, Z_test, test_size=0.5, random_state=1337)

    input_train = torch.from_numpy(X_adv_train).type(torch.FloatTensor)
    baseline_train = torch.mean(input_train,dim=0) # use the mean vector across all instances as the baseline instance
    baseline_train = baseline_train.repeat(input_train.size()[0], 1)

    input_test = torch.from_numpy(X_adv_test).type(torch.FloatTensor)
    baseline_test = torch.mean(input_test,dim=0) # use the mean vector across all instances as the baseline instance
    baseline_test = baseline_test.repeat(input_test.size()[0], 1)


    execution_time =[]
    for i in range(10):
        start = time.time()
        if args.explanations == "IntegratedGradients":
            ig = IntegratedGradients(model)
            attributions_train, delta_train = ig.attribute(input_train, baseline_train, target=0, return_convergence_delta=True)
            # attributions_test, delta_test = ig.attribute(input_test, baseline_test, target=0, return_convergence_delta=True)

        elif args.explanations == "DeepLift":
            dl = DeepLift(model)
            attributions_train, _ = dl.attribute(input_train, baseline_train, target=0, return_convergence_delta=True)
            # attributions_test, _ = dl.attribute(input_test, baseline_test, target=0, return_convergence_delta=True)

        elif args.explanations == "GradientShap":
            gs = GradientShap(model)
            baseline_dist_train = torch.randn(input_train.size()) * 0.001
            # baseline_dist_test = torch.randn(input_test.size()) * 0.001
            attributions_train, _ = gs.attribute(input_train, stdevs=0.09, n_samples=4, baselines=baseline_dist_train,target=0, return_convergence_delta=True)
            # attributions_test, _ = gs.attribute(input_test, stdevs=0.09, n_samples=4, baselines=baseline_dist_test,target=0, return_convergence_delta=True)

        elif args.explanations == "smoothgrad":
            ig = IntegratedGradients(model)
            nt = NoiseTunnel(ig)
            attributions_train, _ = nt.attribute(input_train, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,baselines=baseline_train, target=0, return_convergence_delta=True)
            # attributions_test, _ = nt.attribute(input_test, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,baselines=baseline_test, target=0, return_convergence_delta=True)

        else:
            msg_algo_error: str = "No such algorithm"
            log.error(msg_algo_error)
            raise EnvironmentError(msg_algo_error)

        end = time.time()
        execution_time.append(end-start)
    log.info("##############################################################################################")
    log.info("Mean: {}; Std. Dev. {}".format(np.array(execution_time).mean(),np.array(execution_time).std()))
    print("Mean: {}; Std. Dev. {}".format(np.array(execution_time).mean(),np.array(execution_time).std()))


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="data/", help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default="LFW", help='Options: LAW, CREDIT, COMPAS, CENSUS')
    parser.add_argument('--explanations', type=str, default="IntegratedGradients", help='Options: IntegratedGradients,smoothgrad,DeepLift,GradientShap')
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning Rate")
    parser.add_argument("--decay", type = float, default = 0, help = "Weight decay/L2 Regularization")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training data")
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument("--save", type = bool, default = True, help = "Save model")
    parser.add_argument("--epochs", type = int, default = 30, help = "Number of Model Training Iterations")

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="perfeval.log", filemode="w")
    log: logging.Logger = logging.getLogger("PerfEval")
    args = handle_args()
    main(args, log)
