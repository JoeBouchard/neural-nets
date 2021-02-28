# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:48:57 2021

@author: dowd0002
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import tensorflow_text as tft


#My autoencoder

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
            layers.Dense(mapping_layers, activation='swish'),
            #Bottleneck layer
            layers.Dense(latent_dim, activation='swish'),
            ])
        #Decoder
        self.decoder = tf.keras.Sequential([
            #Third hidden layer
            layers.Dense(mapping_layers, activation='swish'),
            #Output layer
            layers.Dense(visible_dim, activation='swish'),
            #dont Put back into 28x28
            #layers.Reshape((28,28)),
            ])
        
    def call(self, x):
        endoced = self.encoder(x)
        decoded = self.decoder(endoced)
        return decoded


#Function to train an autoencoder on some data
def train(ae, data, testData, epochs=1):
    print(data.shape)

    #Allow keyboard interrupt during training
    try:
        ae.fit(data,data, epochs=epochs, shuffle=True, verbose=2,
               #batch_size=1, 
               validation_data=(testData, testData))
    except KeyboardInterrupt:
        print("Skipping the rest of the training")
    return ae


#Function to turn a bytes file into a tensor
def fileToArray(file, chunkSize=500):
    array = [[]]
    for i in file.read():
        if len(array[-1]) < chunkSize:
            array[-1].append(i)
        else:
            array.append([i])
    #Add filler 0s if necessary
    while len(array[-1]) < chunkSize:
        array[-1].append(0)

    tensor = tf.convert_to_tensor(array)
    return tensor

#Function to write a tensor into an open file
def arrayToFile(tensor, file):
    array = tensor.numpy().tolist()
    fullList = []
    for i in array:
        for j in i:
            fullList.append(j)
    #Trim trailing 0s
    fullList = np.trim_zeros(fullList, 'b')
    filebytes = bytearray(fullList)
    file.write(filebytes)

#Test function for file conversion
def fileArrayTest():
    x = open("readme.txt", "br")
    a = fileToArray(x)
    y = open("readme3.txt", "wb")
    arrayToFile(a, y)
    x.close()
    y.close()
    

