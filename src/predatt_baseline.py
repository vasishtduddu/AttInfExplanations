import torch
import argparse
import logging
import numpy as np
from pathlib import Path
import torch.utils.data as Data
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, accuracy_score, roc_auc_score, plot_roc_curve

from . import data
from . import os_layer
from . import inference_attacks



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

    elif args.dataset == "CENSUS":
        X, y, Z = data.load_census_data(raw_path / 'adult.data')

    elif args.dataset == "LAW":
        X, y, Z = data.load_lawschool_data()
        # used random_state =100 for train_test_split

    elif args.dataset == "CREDIT":
        X, y, Z = data.loadCredit(raw_path)

    else:
        msg_dataset_error: str = "No such dataset"
        log.error(msg_dataset_error)
        raise EnvironmentError(msg_dataset_error)

    if args.with_sattr == True:
        log.info("Concatenating sensitive attributes to the main features")
        X = pd.concat([X, Z], axis=1)
    (X_train, X_test, y_train, y_test, Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.3, random_state=1337)

    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df),columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    X_adv_train, y_adv_train, Z_adv_train = X_test, y_test, Z_test
    X_adv_test, y_adv_test, Z_adv_test = X_train[:len(X_test)], y_train[:len(y_test)], Z_train[:len(Z_test)]

    clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(64,128,32,), verbose=2, max_iter=300,random_state=1337)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    log.info("Target Classifier Performance Evaluation on {} Dataset".format(args.dataset))
    log.info("Recall: {}".format(recall_score(y_test, y_pred)))
    log.info("Precision: {}".format(precision_score(y_test, y_pred)))
    log.info("Balanced Accuracy: {}".format(balanced_accuracy_score(y_test,y_pred)))
    log.info("Accuracy Score: {}".format(accuracy_score(y_test, y_pred)))


    utils.prediction_adversary(clf, X_adv_train, y_adv_train, Z_adv_train, X_adv_test, y_adv_test, Z_adv_test, args, log)



def handle_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument('--raw_path', type=str, default="data/", help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default="LFW", help='Options: LAW, CREDIT, LFW, COMPAS, CENSUS, MEPS')
    parser.add_argument('--attack', type=str, default="proposed", help='Options: proposed, yeom, basic')
    parser.add_argument('--with_sattr', type=bool, default=False, help='Includes the sensitive attributes to the main features for tabular data')
    parser.add_argument('--precrecall', type=bool, default=False, help='Use threshold to optimize for precision and recall instead of ')

    args: argparse.Namespace = parser.parse_args()
    return args



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="attack_attrinf.log", filemode="w")
    log: logging.Logger = logging.getLogger("AttributeInference:")
    args = handle_args()
    main(args, log)
