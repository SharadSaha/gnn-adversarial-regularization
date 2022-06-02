import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    return (x_train, y_train), (x_test, y_test)

def plot(image,label):
  plt.figure()
  plt.imshow(tf.squeeze(image))
  plt.savefig(os.path.join('src','images','perturbation.png'))
  print("label : ",label)
  plt.show()



def preprocessing(image):
  pixels = pd.Series(image.flatten())
  bw_pixels = pixels.apply(lambda x: 0 if x<128 else 255)
  bw_image = bw_pixels.values.reshape((28,28))
  bw_image = bw_image/255.0
  final_image = tf.expand_dims(bw_image, axis=2)
  return final_image


def process_images(x_train,x_test):
    X_train=[]
    X_test=[]
    for image in x_train:
        X_train.append(preprocessing(image))

    for image in x_test:
        X_test.append(preprocessing(image))

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    return X_train,X_test


# keras CNN model

def Model():
  model = tf.keras.models.Sequential([
                                    tf.keras.layers.InputLayer(input_shape=(28,28,1),name='image'),
                                    tf.keras.layers.Conv2D(filters=8, kernel_size = 3, input_shape = [28,28,1], activation='relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                    tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation = 'relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                      tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu',padding='same'),
                                    tf.keras.layers.MaxPooling2D(padding='same'),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=64, activation = 'relu'),
                                    tf.keras.layers.Dense(units = 10, activation='softmax')
  ])
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  return model



def train_model(X_train,y_train,X_test,y_test):
    model = Model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=7,callbacks=[callback])
    model.save('saved_models/fgsm_init_model')
    return model



# creating adversarial example using fsgm

def get_adversarial_pattern(image,label,model):
  Loss = tf.keras.losses.CategoricalCrossentropy()
  with tf.GradientTape() as tape:
    tape.watch(image)
    y_pred = model(image)
    loss = Loss(label,y_pred)
    gradient = tape.gradient(loss, image)
  signed_grad = tf.sign(gradient)
  return signed_grad



# find suitable image for adversarial attack

def find_image(model,X_test,y_test):
  y_pred = model(X_test)
  # y_pred[np.argmin(y_pred)]

  for i in range(len(y_pred)):
    pred = np.argmax(y_pred[i])
    confidence = y_pred[i][np.argmax(y_pred[i])]*100
    if confidence>60 and confidence<62 and pred==y_test[i]:
      print(i)


# visualizing the perturbations

def display_perturbation(i,X_test,y_test,model):
  index = i

  image = tf.convert_to_tensor(X_test[index])
  image = tf.reshape(image,(1,28,28,1))
  label = tf.one_hot(y_test[index],10)
  label = tf.reshape(label, (1,10))
  perturbations = get_adversarial_pattern(image, label,model)
  plot(perturbations[0] * 0.5 + 0.5,y_test[index])
  return perturbations,i,image,label


def display_image(image,label,desc,model):
  pred = model(image)
  confidence = pred[0][label]
  plt.figure()
  plt.imshow(tf.squeeze(image[0]*0.5+0.5))
  plt.title('{} \n {} : {:.2f}% Confidence'.format(desc,label, confidence*100))
  plt.savefig(os.path.join('src','images','adv {}.png'.format(desc)))
  plt.show()


# visualize the difference in confidence on addition of pertubations

def visualize_adv_attack(perturbations,index,image,y_test,model):
  epsilons = [0, 0.01, 0.04, 0.05,0.09,0.15]
  descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                  for eps in epsilons]

  for i, eps in enumerate(epsilons):
    adv_x = image + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_image(adv_x,y_test[index],descriptions[i],model)




if __name__ == "__main__":
  (x_train, y_train), (x_test, y_test) = load_mnist_data()
  X_train,X_test = process_images(x_train,x_test)
  # trained_model = train_model(X_train,y_train,X_test,y_test)
  trained_model = tf.keras.models.load_model('saved_models/fgsm_init_model')
  perturbations,index,image,label = display_perturbation(2185,X_test,y_test,trained_model)
  visualize_adv_attack(perturbations,index,image,y_test,trained_model)