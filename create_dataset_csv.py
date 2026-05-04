import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import argparse

parser = argparse.ArgumentParser(description="Create dataset CSV for ImageNet")
parser.add_argument("train_or_val", choices=["train", "val"], help="Dataset split to process (train or val)")
args = parser.parse_args()

train_or_val = args.train_or_val
data = pd.read_csv(f"<your path to imagenet>/imagenet/LOC_{train_or_val}_solution.csv")

def get_label(val):
    return val.split(" ")[0] 

data["PredictionString"] = data["PredictionString"].apply(get_label)

data["Path"] = np.nan

def generate_paths(x):
    path = f"Data/CLS-LOC/{train_or_val}/"
    if train_or_val == "train":
        path += f"{x.split("_")[0]}/"
    path += f"{x}.JPEG"
    return path

data["Path"] = data["ImageId"].apply(generate_paths)

if train_or_val == "train":
    PredictionString_to_int = dict()
    curr_key = 0 
    for prediction_string in data["PredictionString"]:
        if prediction_string not in PredictionString_to_int:
            PredictionString_to_int[prediction_string] = curr_key
            curr_key += 1
    with open('data/label_encoder.pkl', 'wb') as f:
        pickle.dump(PredictionString_to_int, f)
else:
    with open('data/label_encoder.pkl', 'rb') as f:
        PredictionString_to_int = pickle.load(f)
 
curr_key = 0 # curr_key approach doesn't work for maintaining count 
def encode_prediction_string(prediction_string):
    return PredictionString_to_int[prediction_string]

data["label"] = np.nan
data["label"] = data["PredictionString"].apply(encode_prediction_string)


data.to_csv(f"data/{train_or_val}_dataset.csv")