import tensorflow as tf
import tensorflow_text as tft
import os

print("Opening data...")
file = open(os.pardir+'/trainingdata/wikiText.txt')
raw = file.read().split('\nXFLORBS\n')
file.close()

print("Tokenizing data...")
tokenizer = tft.WhitespaceTokenizer()
tokens = tokenizer.tokenize(raw)
