# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import os
import time


from azureml.core import Run

import mlflow
import mlflow.sklearn

# Based on example:
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli
# which references
# https://github.com/Azure/azureml-examples/tree/main/cli/jobs/train/lightgbm/iris


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_info_path", type=str, help="Path to model info metadata file")
    parser.add_argument(
        "--model_name", type=str, help="Name of the registered model"
    )

    # parse args
    args = parser.parse_args()
    print("Path: " + args.model_info_path)
    # return args
    return args


def main(args):
    """
    Register Model Example
    """
    # Set Tracking URI
    current_experiment = Run.get_context().experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)

    #get run uri from  model info json 

    with open(os.path.join(args.model_info_path,"model_info.json"),"r") as f:
        model_info = json.load(f)


    run_uri = model_info["run_uri"]
    

    # Register the model with Model URI and Name of choice

 
    
    #MLflow model logged inside of a run and register it 
    mlflow.register_model(run_uri, args.model_name)
  


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
