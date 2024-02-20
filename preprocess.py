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

  
def parse_args():  
    # setup arg parser  
    parser = argparse.ArgumentParser()  
  
    # add arguments  
    parser.add_argument("--raw_dataset", type=str)  
    parser.add_argument("--preprocessed_train_data", type=str)  
    parser.add_argument("--preprocessed_test_data", type=str)  
    # parse args  
    args = parser.parse_args()  
    print("args received ", args)  
    # return args  
    return args  
  
def main(args):  
    """  
    Preprocessing of training/test data  
    """  
      
    current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  
    
    print(args.raw_dataset)
    
    data = pd.read_csv(args.raw_dataset)
    
  
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42) 
  
  
    # write preprocessed train data in output path  
    train_data.to_csv(  
        args.preprocessed_train_data + "/train_data.csv",  
        index=False,  
        header=True,  
    )  
  
    # write preprocessed validation data in output path  
    test_data.to_csv(  
        args.preprocessed_test_data + "/test_data.csv",  
        index=False,  
        header=True,  
    )  
  
    # Write MLTable yaml file as well in output folder  
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed  
  
    # read and write MLModel yaml file for train data  
    yaml_str = """  
    paths:  
      - file: ./train_data.csv  
    transformations:  
      - read_delimited:  
          delimiter: ','  
          encoding: 'ascii'  
          empty_as_string: false  
    """  
    with open(args.preprocessed_train_data + "/MLTable", "w") as file:  
        file.write(yaml_str)  
  
    # read and write MLModel yaml file for validation data  
    yaml_str = """  
    paths:  
      - file: ./test_data.csv  
    transformations:  
      - read_delimited:  
          delimiter: ','  
          encoding: 'ascii'  
          empty_as_string: false  
    """  
    with open(args.preprocessed_test_data + "/MLTable", "w") as file:  
        file.write(yaml_str)  
  
  
# run script  
if __name__ == "__main__":  
    # parse args  
    args = parse_args()  
  
    # run main function  
    main(args)  