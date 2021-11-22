import os
import requests
import tempfile
import zipfile
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_lawschool_data():

    with tempfile.TemporaryDirectory() as temp_dir:
        response = requests.get("http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip")
        temp_file_name = os.path.join(temp_dir, "LSAC_SAS.zip")
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(response.content)
        with zipfile.ZipFile(temp_file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data = pd.read_sas(os.path.join(temp_dir, "lsac.sas7bdat"))

        # data contains 'sex', 'gender', and 'male' which are all identical except for the type;
        # map string representation of feature "sex" to 0 for Female and 1 for Male
        data = data.assign(gender=(data["gender"] == b"male") * 1)

        # filter out all records except the ones with the most common two races
        data = data.assign(white=(data["race1"] == b"white") * 1)
        data = data.assign(black=(data["race1"] == b"black") * 1)
        data = data[(data['white'] == 1) | (data['black'] == 1)]

        # encode dropout as 0/1
        data = data.assign(dropout=(data["Dropout"] == b"YES") * 1)
        data = data[(data['pass_bar'] == 1) | (data['pass_bar'] == 0)]

        # drop NaN records for features
        data = data[np.isfinite(data["lsat"]) & np.isfinite(data['ugpa'])]

        # Select relevant columns for machine learning.
        # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
        # TODO: consider using 'fam_inc', 'age', 'parttime', 'dropout'
        data = data[['white', 'black', 'gender', 'lsat', 'ugpa', 'pass_bar']]
        X = data[['lsat', 'ugpa']]
        y = data['pass_bar'].astype(int)
        Z_race = data['white'].apply(lambda x: 1 if x==1 else 0).rename('race')
        Z_sex = data['gender'].rename('sex')
        Z = pd.concat([Z_race, Z_sex], axis=1)
    return X, y, Z


def loadCredit(raw_path):
    df = pd.read_csv(raw_path / 'UCI_Credit_Card.csv')
    df = df.rename(columns={'PAY_0': 'PAY_1'})
    df['LIMIT_BAL'] = df['default.payment.next.month'] + np.random.normal(scale=0.5, size=df.shape[0])
    df.loc[df['SEX'] == 2, 'LIMIT_BAL'] = np.random.normal(scale=0.5, size=df[df['SEX'] == 2].shape[0])
    y = df['default.payment.next.month']
    df['AGE'] = (df['AGE'] < 40).astype(int)
    sex = df.SEX.values - 1
    race = df.AGE.values
    Z = pd.DataFrame({'race': race,'sex': sex})
    X = df.drop(["SEX","ID","default.payment.next.month"], 1)
    return X, y, Z


def loadCOMPAS(raw_path):
    data = pd.read_csv(raw_path)
    y = data.score_factor.values
    race=1-data.Black.values  # 0: black 1: white
    sex=1-data.Female.values  # 0: female 1: male
    y = pd.DataFrame({'Score': y})
    Z = pd.DataFrame({'race': race, 'sex': sex})
    X = data.drop(["score_factor","Black","Female"], 1)
    return X, y, Z


def load_census_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','martial_status', 'occupation', 'relationship', 'race', 'sex','capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names, na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs].assign(race=lambda df: (df['race'] == 'White').astype(int),sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
    # print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    # print(f"targets y: {y.shape} samples")
    # print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")
    return X, y, Z
