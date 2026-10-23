"""
Data loading and splitting for the UCI Adult (Census Income) dataset.

Dataset page: https://archive.ics.uci.edu/dataset/20/census+income
The raw files are read straight from the public UCI URL, so nothing is
stored locally and no local paths are hard-coded.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"

# Column names from the dataset's adult.names file (the raw files have no header).
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

# fnlwgt is a census sampling weight (how many people a row represents), not a
# property of the person; education is a 1-to-1 duplicate of education-num.
DROP_COLUMNS = ["fnlwgt", "education"]

TARGET = "income"


def load_adult():
    """Download adult.data + adult.test, merge them, and return (X, y).

    y is 1 for income >50K and 0 for <=50K.
    Missing values stay as the raw '?' string here; they are handled in
    NeuralNetwork.preprocess so that all cleaning lives in one place.
    """
    # skipinitialspace removes the space after every comma in the raw files.
    train = pd.read_csv(BASE_URL + "adult.data", header=None, names=COLUMNS,
                        skipinitialspace=True)
    # adult.test starts with a junk line ("|1x3 Cross validator").
    test = pd.read_csv(BASE_URL + "adult.test", header=None, names=COLUMNS,
                       skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True)

    # adult.test labels end in a period ("<=50K."); normalise both files.
    labels = df[TARGET].str.strip().str.rstrip(".")
    y = (labels == ">50K").astype(int).to_numpy()

    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    return X, y


def split_data(X, y, ratios=(0.70, 0.15, 0.15), seed=42):
    """Stratified train/val/test split.

    Done in two steps: first carve off the test set, then split the rest
    into train and validation. Stratifying both times keeps the class ratio
    (~24% >50K) the same in all three sets.
    """
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9, "ratios must sum to 1"

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_r, stratify=y, random_state=seed)
    # val's share of the remaining (train + val) portion
    val_share = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share, stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
