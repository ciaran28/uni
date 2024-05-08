# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import glob
import mlflow
import pandas as pd
import logging


def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It is the path to the model folder

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    print(model_dir)
    model_path = os.path.join(model_dir, "model") 
    # model = joblib.load(model_path) 

    # Load the model, it's input types and output names
    model = mlflow.pyfunc.load_model(model_path)
    

def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")

    data = pd.concat(
        map(
            lambda fp: pd.read_csv(fp).assign(filename=os.path.basename(fp)), mini_batch
        )
    )
    print(data.shape)
    print(data.columns)

    new_df = data.drop(["filename"],axis=1)
    
    pred = model.predict(new_df)

    new_df["filename"] = data["filename"]
    data["predictions"] = pred
    print(data.shape)
    print(data.columns)


    azureml_columns = ','.join(data.columns.tolist())
    # print(f"azureml_columns {azureml_columns}")
    result = []
    result.append(azureml_columns)

    # print(result_df.head(2))

    # # Now we have to parse all values as strings, row by row, 
    # # adding a comma between each value
    for index, row in data.iterrows():
        azureml_row = ','.join(map(str, row))
        result.append(azureml_row)
    print(result[:5])
    return result

 
    
