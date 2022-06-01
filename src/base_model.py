
from email.mime import base
import tensorflow as tf
from get_graph_reg_data import HParams

hp = HParams()

def Model():  
  model = tf.keras.models.Sequential([
                                    tf.keras.layers.InputLayer(input_shape=(299, 299, 3),name='image'),
                                    tf.keras.layers.Conv2D(filters=8, kernel_size = 3, input_shape = [299,299,3], activation='relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                    tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation = 'relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                      tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=64, activation = 'relu'),
                                    tf.keras.layers.Dense(units = 4, activation='softmax')
  ])
  return model


def get_base_model():
    base_model = Model()
    base_model.compile(optimizer=hp.OPT,loss=hp.LOSS,metrics=hp.METRICS)

    return base_model