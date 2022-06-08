import neural_structured_learning as nsl
from graph_reg_model import get_GNN_Model
import tensorflow_hub as hub
import os
import argparse
import numpy as np
import tensorflow as tf
from get_data import read_params
from get_graph_reg_data import HParams
from base_model import get_base_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from get_graph_reg_data import make_dataset

hp = HParams()

def train_base_model(X_train,Y_train,X_test,Y_test):
    base_model = get_base_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=40)

    history_ = base_model.fit(
        X_train,Y_train,
        validation_data=(X_test,Y_test),
        epochs=hp.train_epochs,
        callbacks=[callback])

    Y_pred_ = np.array([np.argmax(x) for x in base_model.predict(X_test)])

    print("Validation accuracy of base model: ",accuracy_score(Y_test,Y_pred_))
    base_model.save(os.path.join('saved_models','base_model'))
    return history_


def train_GNN_model(train_dataset,test_dataset,X_test,Y_test):
    gnn_model = get_GNN_Model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    history = gnn_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=hp.train_epochs,
        callbacks=[callback])

    gnn_model.save(os.path.join('saved_models','graph_reg_model'))

    Y_pred = np.array([np.argmax(x) for x in gnn_model.predict(X_test)])

    print("Validation accuracy of gnn model: ",accuracy_score(Y_test,Y_pred))
    return history


def plot_history(history,history_):
    plt.plot(history.history['val_accuracy'])
    plt.plot(history_.history['val_accuracy'])
    plt.legend(['With graph reg','Without graph reg'])
    plt.savefig(os.path.join('src','images','history.png'))
    plt.show()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)
    processed_train_data_path = os.path.join(config['split_data']['train_path1'],config['split_data']['train_path2'],config['split_data']['train_path3'],'train_data.npy')
    processed_test_data_path = os.path.join(config['split_data']['test_path1'],config['split_data']['test_path2'],config['split_data']['test_path3'],'test_data.npy')
    with open(processed_train_data_path, 'rb') as f:
        X_train,Y_train = np.load(f,allow_pickle=True)
    with open(processed_test_data_path, 'rb') as f:
        X_test,Y_test = np.load(f,allow_pickle=True)

    train_dataset = make_dataset('Graph/data/augmented_train_data.tfr', True)
    test_dataset = make_dataset('Graph/data/test_data.tfr')

    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

    history = train_GNN_model(train_dataset,test_dataset,X_test,Y_test)
    history_ = train_base_model(X_train,Y_train,X_test,Y_test)
    plot_history(history,history_)