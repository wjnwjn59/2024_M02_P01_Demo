import os
import gdown
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from config.config import DatasetConfig

def download_dataset():
    id = '1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R'
    DATASET_DIR = 'models/naive_bayes/datasets'
    os.makedirs(DATASET_DIR, exist_ok=True)

    gdown.download(id=id, 
                   output=DatasetConfig.DATASET_PATH,
                   quiet=True,
                   fuzzy=True)

def split_dataset(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=DatasetConfig.VAL_SIZE,
        shuffle=DatasetConfig.IS_SHUFFLE,
        random_state=DatasetConfig.RANDOM_SEED
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=DatasetConfig.TEST_SIZE,
        shuffle=DatasetConfig.IS_SHUFFLE,
        random_state=DatasetConfig.RANDOM_SEED
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_df(csv_path):
    if not os.path.exists(csv_path):
        try:
            download_dataset()
        except:
            ERROR_MSG = 'Failed when attempting download the dataset. Please check the download process.'
            raise Exception(ERROR_MSG)
    df = pd.read_csv(csv_path)

    return df