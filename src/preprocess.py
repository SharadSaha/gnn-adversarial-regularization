import os
import argparse
import numpy as np
import tensorflow as tf
from get_data import get_data,read_params

IMG_DIMENSION = [299,299]

# splitting train and test data 

def train_test_split(data,config_path):
    config = read_params(config_path)
    np.random.shuffle(data)
    n_images = len(data)
    train_size = int(config['split_data']['train_size'] * n_images)

    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data,test_data



# resizing the images to shape suitable for inceptionv3 model

def get_resized_image(image):
  image = image[:,:,:3]
  image = tf.image.resize(image, IMG_DIMENSION)
  return image


# overall image preprocessing

def process_images(data,config_path):
    train_data,test_data = train_test_split(data,config_path)
    for i in range(len(train_data)):
        train_data[i][0] = get_resized_image(train_data[i][0])

    for i in range(len(test_data)):
        test_data[i][0] = get_resized_image(test_data[i][0])

    X_train,Y_train = train_data[:,0],train_data[:,1]
    X_test,Y_test = test_data[:,0],test_data[:,1]

    X_train = np.array([np.array(val) for val in X_train])
    X_test = np.array([np.array(val) for val in X_test])
    Y_train = np.array([np.array(val) for val in Y_train])
    Y_test = np.array([np.array(val) for val in Y_test])

    processed_data = (X_train,Y_train),(X_test,Y_test)
    processed_data = np.array(processed_data,dtype=object)

    return processed_data


# save the processed data

def save_data(processed_data,config_path):
    config = read_params(config_path)
    processed_train_data_path = os.path.join(config['split_data']['train_path1'],config['split_data']['train_path2'],config['split_data']['train_path3'],'train_data.npy')
    processed_test_data_path = os.path.join(config['split_data']['test_path1'],config['split_data']['test_path2'],config['split_data']['test_path3'],'test_data.npy')

    with open(processed_train_data_path, 'wb') as f:
        np.save(f, np.array(processed_data[0]))
    with open(processed_test_data_path, 'wb') as f:
        np.save(f, np.array(processed_data[1]))

    with open(processed_train_data_path, 'rb') as f:
        loaded_data_1 = np.load(f,allow_pickle=True)
    with open(processed_test_data_path, 'rb') as f:
        loaded_data_2 = np.load(f,allow_pickle=True)
    print(loaded_data_1.shape,loaded_data_2.shape)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
    processed_data = process_images(data,config_path=parsed_args.config)
    save_data(processed_data,config_path=parsed_args.config)