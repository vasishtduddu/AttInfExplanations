import argparse
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from captum.attr import GradientShap,DeepLift,IntegratedGradients,NoiseTunnel

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

    if args.with_sattr == True:
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
    X_train, X_test, y_train, y_test, Z_train, Z_test = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.astype(int), y_test.astype(int), Z_train.astype(int), Z_test.astype(int)

    traindata = Data.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.LongTensor))
    testdata = Data.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor))
    trainloader = torch.utils.data.DataLoader(dataset=traindata, batch_size=64, shuffle=False)
    testloader = torch.utils.data.DataLoader(dataset=testdata, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    model = utils.train(args.epochs, model, trainloader, testloader, optimizer, args, log)

    # choose data for generating explanations
    (X_adv_train, X_adv_test, y_adv_train, y_adv_test, Z_adv_train, Z_adv_test)  = train_test_split(X_test, y_test, Z_test, test_size=0.5, random_state=1337)

    input_train = torch.from_numpy(X_adv_train).type(torch.FloatTensor)
    target_train = torch.from_numpy(y_adv_train)
    baseline_train = torch.mean(input_train,dim=0) # use the mean vector across all instances as the baseline instance
    baseline_train = baseline_train.repeat(input_train.size()[0], 1)

    input_test = torch.from_numpy(X_adv_test).type(torch.FloatTensor)
    target_test = torch.from_numpy(y_adv_test)
    baseline_test = torch.mean(input_test,dim=0) # use the mean vector across all instances as the baseline instance
    baseline_test = baseline_test.repeat(input_test.size()[0], 1)

    if args.explanations == "IntegratedGradients":
        ig = IntegratedGradients(model)
        attributions_train, delta_train = ig.attribute(input_train, baseline_train, target=target_train, return_convergence_delta=True)
        attributions_test, delta_test = ig.attribute(input_test, baseline_test, target=target_test, return_convergence_delta=True)


    elif args.explanations == "DeepLift":
        dl = DeepLift(model)
        attributions_train, delta_train = dl.attribute(input_train, baseline_train, target=target_train, return_convergence_delta=True)
        attributions_test, delta_test = dl.attribute(input_test, baseline_test, target=target_test, return_convergence_delta=True)

    elif args.explanations == "GradientShap":
        gs = GradientShap(model)
        baseline_dist_train = torch.randn(input_train.size()) * 0.001
        baseline_dist_test = torch.randn(input_test.size()) * 0.001
        attributions_train, delta_train = gs.attribute(input_train, stdevs=0.09, n_samples=4, baselines=baseline_dist_train,target=target_train, return_convergence_delta=True)
        delta_train = torch.mean(delta_train.reshape(input_train.shape[0], -1), dim=1)
        attributions_test, delta_test = gs.attribute(input_test, stdevs=0.09, n_samples=4, baselines=baseline_dist_test,target=target_test, return_convergence_delta=True)
        delta_test = torch.mean(delta_test.reshape(input_test.shape[0], -1), dim=1)

    elif args.explanations == "smoothgrad":
        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        attributions_train, delta_train = nt.attribute(input_train, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,baselines=baseline_train, target=target_train, return_convergence_delta=True)
        delta_train = torch.mean(delta_train.reshape(input_train.shape[0], -1), dim=1)
        attributions_test, delta_test = nt.attribute(input_test, nt_type='smoothgrad', stdevs=0.02, nt_samples=4,baselines=baseline_test, target=target_test, return_convergence_delta=True)
        delta_test = torch.mean(delta_test.reshape(input_test.shape[0], -1), dim=1)

    else:
        msg_algo_error: str = "No such algorithm"
        log.error(msg_algo_error)
        raise EnvironmentError(msg_algo_error)


    # attribute inference attack [explanations only, prediction+ explanations] with and without sensitive attribute in training dataset
    if args.attfeature == "expl":
        attributions_train = attributions_train.detach().numpy()
        delta_train = delta_train.numpy()
        delta_train = np.expand_dims(delta_train, axis=0)
        X_adv_train = np.concatenate((attributions_train, delta_train.T), axis=1)

        attributions_test = attributions_test.detach().numpy()
        delta_test = delta_test.numpy()
        delta_test = np.expand_dims(delta_test, axis=0)
        X_adv_test = np.concatenate((attributions_test, delta_test.T), axis=1)

        utils.attinfattack(X_adv_train, Z_adv_train, X_adv_test, Z_adv_test, args, log)

    elif args.attfeature == "both":
        model.eval()
        predictions_train = model(torch.from_numpy(X_adv_train).type(torch.FloatTensor))
        predictions_train = predictions_train.detach().numpy()
        attributions_train = attributions_train.detach().numpy()
        delta_train = delta_train.numpy()
        delta_train = np.expand_dims(delta_train, axis=0)
        conc_train = np.concatenate((attributions_train, predictions_train), axis=1)
        X_adv_train = np.concatenate((conc_train, delta_train.T), axis=1)

        predictions_test = model(torch.from_numpy(X_adv_test).type(torch.FloatTensor))
        predictions_test = predictions_test.detach().numpy()
        attributions_test = attributions_test.detach().numpy()
        delta_test = delta_test.numpy()
        delta_test = np.expand_dims(delta_test, axis=0)
        conc_test = np.concatenate((attributions_train, predictions_train), axis=1)
        X_adv_test = np.concatenate((conc_test, delta_test.T), axis=1)

        utils.attinfattack(X_adv_train, Z_adv_train, X_adv_test, Z_adv_test, args, log)

    else:
        msg_config_error: str = "No such configuration"
        log.error(msg_config_error)
        raise EnvironmentError(msg_config_error)


def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="data/", help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default="LFW", help='Options: LAW, CREDIT, COMPAS, CENSUS')
    parser.add_argument('--explanations', type=str, default="IntegratedGradients", help='Options: IntegratedGradients,smoothgrad,DeepLift,GradientShap')
    parser.add_argument('--with_sattr', type=bool, default=False, help='Includes the sensitive attributes to the main features for tabular data')
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning Rate")
    parser.add_argument("--decay", type = float, default = 0, help = "Weight decay/L2 Regularization")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training data")
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument("--save", type = bool, default = True, help = "Save model")
    parser.add_argument("--epochs", type = int, default = 30, help = "Number of Model Training Iterations")
    parser.add_argument('--attfeature', type=str, default="expl", help='Options: expl, both')

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="attack_attrinf.log", filemode="w")
    log: logging.Logger = logging.getLogger("AttributeInference")
    args = handle_args()
    main(args, log)
