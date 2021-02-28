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
            layers.Dense(visible_dim, activation='linear'),
            #dont Put back into 28x28
            #layers.Reshape((28,28)),
            ])
        
    def call(self, x):
        endoced = self.encoder(x)
        decoded = self.decoder(endoced)
        return decoded

def tokenizeUTF(sentence, size):

    toReturn = []
    for i in sentence:
        toReturn.append(ord(i))

    while len(toReturn) < size:
        toReturn.append(0)
    return toReturn

def tokenizeBin(sentence, size):
    toReturn = []
    for i in sentence:
        toReturn.append(int(i))

    while len(toReturn) < size:
        toReturn.append(0)
    return toReturn[0:500]

def detokenizeUTF(sentence):

    toReturn = ''
    for i in sentence:
        if i != 0:
            toReturn += chr(int(round(i, 1)))

    return toReturn


#Function to create a set of tensors from a text file of sentences
def tokenizeFile(filename, count=0, tokenizer='test_model.model'):
    dataSet = []
    #Prep the tokenizer
    model = open(tokenizer, mode='rb')
    tkr = tft.SentencepieceTokenizer(model=model.read())
    model.close()
    #Read the file
    file = open(filename, mode='rb')
    print("Splitting...")
    allLines = file.read().split(b"\r\nXFLORB\r\n")
    numLines = len(allLines)
    #print(allLines)
    #Tokenize
    counter = 0
    tokens = []
    for sent in allLines:
        if counter%1000 == 0:
            print(counter, len(allLines))
        counter+=1
        tokens.append(tokenizeBin(sent, 500))
    #print(tokens)
    print("Making tensor...")
    tokens = tf.convert_to_tensor(tokens)#.to_tensor(shape=[None, 100])#tkr.tokenize(allLines).to_tensor(shape=[None, 34])
    #print(tokens)
    print("Returning tensor...")
    return tokens

#Function to code-encode a line
def predict(line, autoencoder, tokenizer='test_model.model'):
    #Prep the tokenizer
    model = open(tokenizer, mode='rb')
    tkr = tft. SentencepieceTokenizer(model=model.read())
    model.close()
    tokens = tf.convert_to_tensor([tokenizeUTF(line, 500)])#.to_tensor(shape=[None, 100])#tkr.tokenize([line]).to_tensor(shape=[None, 34])
    #tokens = tokenizeFile('readme.txt')#[0].to_tensor(shape=[None, 34])

    #print("Input: " + str(tokens))
    c = autoencoder.encoder(tokens).numpy()
    t2 = autoencoder.decoder(c).numpy()
    #print("OPUtpt: " + str(t2))

    t3 = list(t2[0].astype(float))
    t4 = []
    for i in t3:
        if i > -0.5:
            t4.append(i)#int(round(i, 0)))
    #Detokenize
    x = detokenizeUTF(t4)#tkr.detokenize(t4).numpy()
    #Extract from tensor:
   # raw = x.split(b" \xe2\x81\x87")[0]
    #print(raw.decode('ascii'))
    print(x)


#Function to train an autoencoder on some data
def train(ae, data, testData):
    print(data.shape)

    #Allow keyboard interrupt during training
    try:
        ae.fit(data,data, epochs=300, shuffle=True, verbose=2,
               #batch_size=1, 
               validation_data=(testData, testData))
    except KeyboardInterrupt:
        print("Skipping the rest of the training")
        
    return ae

#Make data
#data = tokenizeFile(os.pardir + '/trainingdata/askscience.txt')
data = tokenizeFile('../trainingdata/lifeprotips.txt')
#make ae
ae = myAutoencoder(visible_dim=500, latent_dim=500, mapping_layers=500)
ae.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001), loss=losses.MeanSquaredError())

ae = train(ae, data, data)
print("Original: " + str(data))
encoded = ae.encoder(data).numpy()
print("Encoded: " + str(encoded))
decoded = ae.decoder(encoded).numpy()
print("Decoded: "+ str(decoded))


predict("hello world", ae)

while True:
    x = input()
    predict(x, ae)
    
    

    
