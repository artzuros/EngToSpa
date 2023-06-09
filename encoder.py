import gc
import time 
import re
import unicodedata 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, return_state=True)
    def call(self, input_sequences, states):
        embed = self.embedding(input_sequences)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c
    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.hidden_dim]), tf.zeros([batch_size, self.hidden_dim]))