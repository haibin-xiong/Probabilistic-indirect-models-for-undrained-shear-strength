import numpy as np
import pandas as pd
import math
import random
import os.path
import pathlib
from sklearn.model_selection import train_test_split

def split_data(data, variables):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        variables (list): List of column names for the variables in the dataset.
        train_ratio (float): Proportion of data to be used for training.
        valid_ratio (float): Proportion of data to be used for validation.
        test_ratio (float): Proportion of data to be used for testing.

    Returns:
        dict: A dictionary containing the training, validation, and test sets
              along with their sizes and the total number of data points.
    """

    dataset_valid = data[variables].notna().all(axis=1)
    selected = data.loc[dataset_valid, variables]
    # print(selected.shape)
    input_selected = selected.iloc[:, :-1].to_numpy(dtype=np.float32)
    target_selected = selected.iloc[:, -1].to_numpy(dtype=np.float32)

    def arcsigmoid(x):
        return np.log(x / (1 - x))

    for i in range(len(variables) - 2):
        input_selected[:,i] = np.arcsinh(input_selected[:,i])

    input_selected[:,-1] = arcsigmoid(input_selected[:,-1] / 1.2)
    target_selected = np.log(target_selected)

    X_train, X_test, y_train, y_test = train_test_split(input_selected, target_selected, test_size=0.1, random_state=42)

    return input_selected, X_train, X_test, target_selected, y_train, y_test
