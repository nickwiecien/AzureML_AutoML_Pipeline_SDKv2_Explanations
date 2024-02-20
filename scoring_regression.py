import argparse  
import datetime  
from pathlib import Path  
import yaml  
from mltable import load  
import time  
import mltable  
import random  
import pandas as pd  
from sklearn.model_selection import train_test_split
from azureml.core import Run, Model
import mlflow
import json
import shap
import os
  
def parse_args():  
    # setup arg parser  
    parser = argparse.ArgumentParser()  
  
    # add arguments  
    parser.add_argument("--training_data", type=str)  
    parser.add_argument("--scoring_data", type=str)  
    parser.add_argument("--model_name", type=str) 
    parser.add_argument("--target_column", type=str) 
    parser.add_argument("--scored_data", type=str) 
    
    # parse args  
    args = parser.parse_args()  
    print("args received ", args)  
  
    return args  
  
def main(args):  
    """  
    Preprocessing of training/test data  
    """  
    # Set Tracking URI
    current_run = Run.get_context()
    current_experiment = current_run.experiment
    tracking_uri = current_experiment.workspace.get_mlflow_tracking_uri()
    print("tracking_uri: {0}".format(tracking_uri))
    mlflow.set_tracking_uri(tracking_uri)
      
    data = pd.read_csv(args.training_data)
    
    try:
        data = data.drop(columns=[args.target_column])
    except Exception as e:
        pass
  
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42) 
    
    mlflow_model = Model(current_experiment.workspace, args.model_name)
    model = mlflow.pyfunc.load_model(mlflow_model.properties['mlflow.modelSourceUri'])
    
    scoring_data_files = [x for x in os.listdir(args.scoring_data) if '.csv' in x]
    
    scored_df = pd.DataFrame()
    
    for f in scoring_data_files:
        curr_df = pd.read_csv(os.path.join(args.scoring_data, f))
        scored_df = pd.concat([scored_df, curr_df])
        
    try:
        scored_df = scored_df.drop(columns=[args.target_column])
    except Exception as e:
        pass
    
    preds = model.predict(scored_df)
    
    os.makedirs(args.scoring_data, exist_ok=True)
    
    def model_predict(data_as_array):  
        # Convert the input to the format expected by the pyfunc model (pandas DataFrame in this case)  
        data_as_df = pd.DataFrame(data_as_array, columns=train_data.columns)  
        # Get predictions from the pyfunc model  
        return model.predict(data_as_df)  


    import shap  

    # Use a representative subset of your data as the background dataset  
    background_data = shap.sample(train_data, min(150, len(train_data)))  # For instance, 100 random samples  
    explainer = shap.KernelExplainer(model_predict, background_data)  
    shap_values = explainer.shap_values(scored_df)  
    
    shap_df = pd.DataFrame(shap_values, columns = [x + '_SHAP' for x in background_data.columns])
    
                                  
    scored_df[f'PREDICTED_{args.target_column}'] = preds
    for col in shap_df.columns:
         scored_df[col] = shap_df[col] 
    
    current_timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    scored_df.to_csv(args.scored_data + f'/scored_data_{current_timestamp}.csv', index=False)

# run script  
if __name__ == "__main__":  
    # parse args  
    args = parse_args()  
  
    # run main function  
    main(args)  