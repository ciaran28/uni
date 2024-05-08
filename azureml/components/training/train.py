import argparse
import os
import shutil
import tempfile
from azureml.core import Run
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


import json

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--target_column_name", type=str, help="Name of target column")
    parser.add_argument('--model_info_path', type = str, help = "model info  path")
    parser.add_argument("--model_artifacts_path",type=str, help="mlflow artifact path", default="model")
      
    # parse args
    args = parser.parse_args()
    print(args.training_data)
    print("\n")
  
    print(args.target_column_name)
    # return args

    # return args
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
    mlflow.autolog()
    # Read in data
    
    print("Reading data")
    
    training_df = pd.read_csv(args.training_data)
    print("train data shape  : {0}, train data columns: {1}".format(training_df.shape,training_df.columns))

    y_train = training_df[args.target_column_name]
    X_train = training_df.drop(labels=args.target_column_name, axis="columns")
 
    print("X_train cols: {0}".format(X_train.columns))

    print("Training model")
    # The estimator can be changed to suit
    model =  RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X_train, y_train)


    # Log model in mlflow explicitly if autolog is not enabled
    #mlflow.sklearn.log_model(sk_model=model,artifact_path=args.model_artifacts_path,registered_model_name="mlflow_test_model")

    # get run details

    runs = mlflow.search_runs(experiment_names=[current_experiment.name],output_format="list",)
    last_run = runs[-1]
    print("Last run ID:", last_run.info)

    run_uri = "runs:/{0}//{1}".format(last_run.info.run_id,args.model_artifacts_path)

    model_info = {"experiment_id": last_run.info.experiment_id,"run_id": last_run.info.run_id,"artifact_path": args.model_artifacts_path,"run_uri":run_uri}

    with open(os.path.join(args.model_info_path,"model_info.json"), "w") as outfile: 
        json.dump(model_info,outfile)

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