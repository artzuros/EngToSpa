from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)

train_filename = 'spa.txt'

INPUT_COLUMN = 'input'
TARGET_COLUMN = 'target'
TARGET_FOR_INPUT = 'target_for_input'
NUM_SAMPLES = 50000
MAX_VOCAB_SIZE = 50000
EMBEDDING_DIM = 128
HIDDEN_DIM = 1024

BATCH_SIZE = 64
EPOCHS = 10

ATTENTION_FUNC = 'general'

from helper import unicode_to_ascii, preprocess_sentence

df = pd.read_csv(train_filename, sep='\t', header=None, names=[INPUT_COLUMN, TARGET_COLUMN], nrows=NUM_SAMPLES, usecols= [0,1])
input_data = df[INPUT_COLUMN].apply(lambda x : preprocess_sentence(x)).tolist()
target_data = df[TARGET_COLUMN].apply(lambda x : preprocess_sentence(x) + ' <eos> ').tolist()
target_input_data = df[TARGET_COLUMN].apply(lambda x : '<sos> ' + preprocess_sentence(x)).tolist()

tokenizer_inputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_inputs.fit_on_texts(input_data)
input_sequences = tokenizer_inputs.texts_to_sequences(input_data)
input_max_len = max(len(s) for s in input_sequences)

tokenizer_outputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_data)
tokenizer_outputs.fit_on_texts(target_input_data)
target_sequences = tokenizer_outputs.texts_to_sequences(target_data)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_input_data)
target_max_len = max(len(s) for s in target_sequences)

word2idx_inputs = tokenizer_inputs.word_index
word2idx_outputs = tokenizer_outputs.word_index
num_words_output = len(word2idx_outputs) + 1
num_words_inputs = len(word2idx_inputs) + 1
idx2word_inputs = {v:k for k, v in word2idx_inputs.items()}
idx2word_outputs = {v:k for k, v in word2idx_outputs.items()}

encoder_inputs = pad_sequences(input_sequences, maxlen=input_max_len, padding = 'post')
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=target_max_len, padding = 'post')
decoder_targets = pad_sequences(target_sequences, maxlen=target_max_len, padding = 'post')

from decoder import Decoder, LuongAttention
from encoder import Encoder

num_words_inputs = len(word2idx_inputs) + 1
num_words_outputs = len(word2idx_outputs) + 1
encoder = Encoder(num_words_inputs, EMBEDDING_DIM, HIDDEN_DIM)
decoder = Decoder(num_words_outputs, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_FUNC)

import os
optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
checkpoint_dir = './training_ckpt_seq2seq_att'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def predict_seq2seq_att(input_text):
    if input_text is None:
        input_text = input_data[np.random.choice(len(input_data))]
    print(input_text)
    input_seq = tokenizer_inputs.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=input_max_len, padding='post')
    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(input_seq), en_initial_states)
    de_input = tf.constant([[word2idx_outputs['<sos>']]])
    de_state_h, de_state_c = en_outputs[1:]
    
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(idx2word_outputs[de_input.numpy()[0][0]])
        alignments.append(alignment.numpy())

        if out_words[-1] == '<eos>' or len(out_words) >= 20:
            break
    print(' '.join(out_words))
    return np.array(alignments), input_text.split(' '), out_words

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prompt = data['prompt']
    alignments, source, response = predict_seq2seq_att(prompt)
    response = ' '.join(response)

    return jsonify({"response":response})

if __name__ == "__main__":
    app.run(debug = True)