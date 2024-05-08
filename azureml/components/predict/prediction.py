# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""This module will load mlflow model and do prediction."""

import argparse
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from mlflow.sklearn import load_model


def init():
    print("Environment variables start ****")
    for key, val in os.environ.items():
        print(key, val)
    print("Environment variables end ****")

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="ParallelRunStep Agent"
    )
    parser.add_argument("--model", type=str, default=0)
    args, _ = parser.parse_known_args()

    global clf_model

    clf_model = load_model(args.model)


def run(mini_batch):
    df_predictions = []
    for file_path in mini_batch:
        df_data = pd.read_csv(file_path)
        pred = clf_model.predict(df_data)
        df_data["predictions"] = pred
        print(f"file_path {file_path}")
        print(df_data.shape)
        print(df_data.columns)
        df_predictions.append(df_data)

    result_df = pd.concat(df_predictions,axis=0)
    print(len(df_predictions))
    print(result_df.shape)
    print(result_df.columns)
    azureml_columns = ','.join(result_df.columns.tolist())
    # print(f"azureml_columns {azureml_columns}")
    result = []
    result.append(azureml_columns)

    # print(result_df.head(2))

    # # Now we have to parse all values as strings, row by row, 
    # # adding a comma between each value
    for index, row in result_df.iterrows():
        azureml_row = ','.join(map(str, row))
        result.append(azureml_row)
    print(result[:5])
    return result
