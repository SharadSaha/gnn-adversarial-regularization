import os
import argparse
import numpy as np
import tensorflow as tf
from get_data import read_params,get_data


# get the data and save the raw data

def load_and_save(config_path):
    config = read_params(config_path)
    data = get_data(config_path)
    raw_data_path = os.path.join(config['load_data']['raw_dataset1'],config['load_data']['raw_dataset2'],'raw_data.npy')
    with open(raw_data_path, 'wb') as f:
        np.save(f, np.array(data))
    with open(raw_data_path, 'rb') as f:
        loaded_data = np.load(f,allow_pickle=True)
    print(loaded_data.shape)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)

