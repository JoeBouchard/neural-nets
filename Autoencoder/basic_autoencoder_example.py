# -*- coding: utf-8 -*-
"""
Basic autoencoder example from
https://www.tensorflow.org/tutorials/generative/autoencoder
Created on Sun Feb 14 12:36:28 2021

@author: dowd0002
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model



class Autoencoder(Model):
  def __init__(self, visible_dim=784, latent_dim=64):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.visible_dim=visible_dim

    self.encoder = tf.keras.Sequential([
      #Convert 28x28 to 1x784
      layers.Flatten(), 
      #Hidden layer 
      #Single layer with size latent_dim, activation function RELU
      layers.Dense(latent_dim, activation='relu'),
    ])
    
    self.decoder = tf.keras.Sequential([
      #Single layer with output size visible_dim
      layers.Dense(visible_dim, activation='sigmoid'),
      #Convert 1x784 to 28x28
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#Classic Sparse Autoencoder model for NLPCA
class myAutoencoder(Model):
    def __init__(self, visible_dim=2, latent_dim=1, mapping_layers=6):
        super(myAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.visible_dim=visible_dim
        self.mapping_layers=mapping_layers
        
        #Encoder
        self.encoder = tf.keras.Sequential([
            #Flatten just in case
            layers.Flatten(),
            #First hidden layer
            layers.Dense(mapping_layers, activation='sigmoid'),
            #Bottleneck layer
            layers.Dense(latent_dim, activation='sigmoid'),
            ])
        #Decoder
        self.decoder = tf.keras.Sequential([
            #Third hidden layer
            layers.Dense(mapping_layers, activation='sigmoid'),
            #Output layer
            layers.Dense(visible_dim, activation='sigmoid'),
            #Put back into 28x28
            layers.Reshape((28,28)),
            ])
        
    def call(self, x):
        endoced = self.encoder(x)
        decoded = self.decoder(endoced)
        return decoded

#NLPCA training function
def nlpca():
    #Create training data
    x_train = []
    for i in range(10000):
        theta = random.random() * 2 * math.pi
        y1 = 0.4*math.sin(theta) + 0.5
        y2 = 0.4*math.cos(theta) + 0.5
        
        x_train.append([y1,y2])
    
    x_train = tf.constant(x_train)
    print(x_train.shape)
    print(x_train)

    ac = myAutoencoder()
    ac.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1), loss=losses.MeanSquaredError())
    ac.fit(x_train, x_train, 
           epochs=100, shuffle=True, validation_data=(x_train, x_train))
    
    #Show some workings
    inner = ac.encoder(x_train).numpy()
    end = ac.decoder(inner).numpy()
    start = x_train.numpy()
    for i in range(10):
        print(start[i])
        print(end[i])
        print("\n")
    
        
    print(ac.encoder.summary())
    print(ac.decoder.summary())

#Clothes-training function
def lookatclothes():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255. # Converts uint8 to float32 between 0 and 1
    
    print (x_train.shape)
    print(type(x_train))
    print (x_test.shape)
    
    
    latent_dim = 64 
    
    autoencoder = myAutoencoder(visible_dim=784, latent_dim=64, mapping_layers=300)
    #autoencoder = Autoencoder(latent_dim=64)
    
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
      # display original
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_test[i])
      plt.title("original")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    
      # display reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(decoded_imgs[i])
      plt.title("reconstructed")
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
    plt.show()

#nlpca()
lookatclothes()
