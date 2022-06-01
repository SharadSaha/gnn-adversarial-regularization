import os
import yaml
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    print(config)
    data_path = os.path.join(config['data_source']['dataset1'],config['data_source']['dataset2'],"*","*") +".png"
    print(data_path)
    data = []
    for img in glob.glob(data_path):
        label = int(img.split(os.path.sep)[2][-1])-1
        image = cv2.imread(img)
        data.append((image,label))

    data = np.array(data,dtype=object)
    print(data.shape)
    return data

def plot(data):
  image,label = data
  plt.imshow(tf.squeeze(image))
  plt.show()
  print("\nLabel: ",label)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
    plot(data[0])
