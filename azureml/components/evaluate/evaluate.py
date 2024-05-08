import argparse
import os
from azureml.core import Run
import mlflow
import mlflow.sklearn
import mltable
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import numpy as np
import pandas as pd
import json



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument('--model_info_path', type = str, help = "model details path")
    parser.add_argument('--predictions_path', type = str, help = "test data predictions path")
    parser.add_argument('--model_path', type=str,help ='path to download artifacts')
    
        
    # parse args
    args = parser.parse_args()
    print(args.test_data)
    print("\n")
    print(args.model_info_path)
    print("\n")
    print(args.target_column_name)
    # return args
    print("\n")
    print(args.predictions_path)

    print("\n")
    print(args.model_path)
   
    return args


def main(args):
    current_experiment = Run.get_context().experiment
    print(current_experiment)
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)
    client = mlflow.tracking.MlflowClient()
    mlflow.start_run()
    # Read in data
    print("Reading data")
    test_df = pd.read_csv(args.test_data)

    print("Extracting X_test, y_test")
    print("all_data cols: {0}".format(test_df.columns))
    y_test = test_df[args.target_column_name]
    X_test= test_df.drop(labels=args.target_column_name, axis="columns")
    
    print("X_test cols: {0}".format(X_test.columns))
    print("X_test shape: {0}".format(X_test.shape))

    print("validate model")
    # Load model from model input path
    with open(os.path.join(args.model_info_path,"model_info.json"),"r") as f:
        model_info = json.load(f)


    run_id = model_info["run_id"]
    artifact_path = model_info["artifact_path"]

    file_path = mlflow.artifacts.download_artifacts(run_id=run_id,dst_path=args.model_path,artifact_path=artifact_path)
    print(os.listdir(file_path))

    model_file = os.path.join(args.model_path,artifact_path,"model.pkl")

    
    with open(model_file,"rb") as f:
        model = pickle.load(f)
   

    y_pred = model.predict(X_test)
    print("y_pred shape: {0}".format(y_pred.shape))
    
    
  # Calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred)
    mlflow.log_metric('precision', precision)

    
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Precision: {0}, Recall: {1}, F1 Score: {2}".format(precision,recall,f1))


   
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('F1 score', f1)
 
    test_df["predictions"] = y_pred
    output_data = test_df.to_csv(os.path.join(args.predictions_path,"predictions.csv"))



    mlflow.end_run()
    

    

# run script
if __name__ == "__main__":
    # add space in logs
    print("*" * 60)
    print("\n\n")

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")