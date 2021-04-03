#NEw autoencoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import random
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Classic Sparse Autoencoder model for NLPCA
class myAutoencoder(Model):
    def __init__(self, visible_dim=2, latent_dim=1, mapping_layers=6, activation='sigmoid'):
        super(myAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.visible_dim=visible_dim
        self.mapping_layers=mapping_layers
        
        #Encoder
        self.encoder = tf.keras.Sequential([
            #Flatten just in case
            layers.Flatten(),
            #First hidden layer
            layers.Dense(mapping_layers, activation=activation),
            #Bottleneck layer
            layers.Dense(latent_dim, activation='linear'),
            ])
        #Decoder
        self.decoder = tf.keras.Sequential([
            #Third hidden layer
            layers.Dense(mapping_layers, activation=activation),
            #Output layer
            layers.Dense(visible_dim, activation='linear'),
            #Put back into 28x28
            #layers.Reshape((28,28))
            ])
        self.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                     loss=tf.keras.losses.MeanSquaredError())
    def call(self, x):
        endoced = self.encoder(x)
        decoded = self.decoder(endoced)
        return decoded
    def eval(self, data):
        data2 = self.call(data)
        mse = tf.keras.losses.MeanSquaredError()
        return mse(data2,data)
        
    def train(self, data, epochs=1):
        self.fit(data,data, epochs=epochs, shuffle=True,
                 verbose=2, validation_data=(data,data))
        

#Sequential Autoencoder model for NLPCA
class sequentialAutoencoder():
    def __init__(self, visible_dim=2, latent_dim=1, mapping_layers=[6], activation='sigmoid'):
        self.coders = []
        for i in range(latent_dim):
            if type(mapping_layers) == type(1) :
                ml = mapping_layers
            else: ml = mapping_layers[i]    
            #Create list of autoencoders
            self.coders.append(myAutoencoder(visible_dim=visible_dim, latent_dim=1,
                                             mapping_layers=ml,
                                             activation=activation))

    def train(self, data, epochs=1):
        for ac in self.coders:
            ac.fit(data, data, epochs=epochs, shuffle=True, verbose=2,
                   validation_data=(data,data))
            data_prime = ac.call(data)
            data = data - data_prime
            
    def call(self, data):
        final_data = data - data #so it's all zeroes
        for ac in self.coders:
            data_prime = ac.call(data)
            final_data = final_data + data_prime
            data = data - data_prime
        return final_data

    def eval(self, data):
        data2 = self.call(data)
        mse = tf.keras.losses.MeanSquaredError()
        return mse(data2, data)

def lookatclothes():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255. # Converts uint8 to float32 between 0 and 1
    print (x_train.shape)
    print(type(x_train))
    print (x_test.shape)
    sqa = sequentialAutoencoder(visible_dim=784, latent_dim=2, mapping_layers=50)
    na = myAutoencoder(visible_dim=784, latent_dim=2, mapping_layers=50)
    sqa.train(x_train, epochs=10)

    na.train(x_train, epochs=10)
    print("SQA")
    print(sqa.eval(x_train))
    print("NA")
    print(na.eval(x_train))

    nrecoded = na.call(x_test).numpy()
    recoded = sqa.call(x_test).numpy()
    n = 10
    plt.figure(figsize=(30, 4))
    for i in range(n):
      # display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(recoded[i])
      plt.title("original")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    
      # display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(nrecoded[i])
      plt.title("reconstructed")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      # display reconstruction-sqanormal
      #ax = plt.subplot(2, n, i + 1 + n )
      #plt.imshow(recoded[i])
      #plt.title("reconstructed-sqa")
      #plt.gray()
      #ax.get_xaxis().set_visible(False)
      #ax.get_yaxis().set_visible(False)
    plt.show()

#lookatclothes()
