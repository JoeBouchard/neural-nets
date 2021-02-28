# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:57:45 2021

@author: dowd0002
"""

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
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
print(tf.config.list_physical_devices('GPU'))
import tensorflow_text as tft

model = open('test_model.model', mode='rb')

tokenizer = tft.SentencepieceTokenizer(model=model.read())
model.close()

x = tokenizer.tokenize(['hello world'])

print(x)
  
y = tokenizer.detokenize(x)

print(y[0].value)
