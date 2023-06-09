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

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError("'Provide either dot, general or concat as attention_func'")
        if attention_func == 'general':
            self.Wa = tf.keras.layers.Dense(rnn_size)
        elif attention_func == 'concat':
            self.Wa = tf.keras.layers.Dense(rnn_size, activation = 'tanh')
            self.va = tf.keras.layers.Dense(1)
    
    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True) # (batch_size, 1, max_len)
        elif self.attention_func == 'general':
            score = tf.matmul(decoder_output, self.Wa(encoder_output), transpose_b=True) # (batch_size, 1, max_len)
        elif self.attention_func == 'concat':
            decoder_output = tf.tile(decoder_output, [1, encoder_output.shape[1], 1]) # (batch_size, max_len, hidden_dim)
            score = self.va(self.Wa(tf.concat([decoder_output, encoder_output], axis=-1))) # (batch_size, max_len, 1)
            score = tf.transpose(score, perm=[0, 2, 1]) # (batch_size, 1, max_len)
        alignment = tf.keras.activations.softmax(score, axis=-1) # (batch_size, 1, max_len)
        context = tf.matmul(alignment, encoder_output) # (batch_size, 1, rnn_size)

        return context, alignment

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_func):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(hidden_dim, attention_func)
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(hidden_dim, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)
    
    def call(self, input_sequences, state, encoder_output):
        embed = self.embedding(input_sequences)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
        context, alignment = self.attention(lstm_out, encoder_output)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        lstm_out = self.wc(lstm_out)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment