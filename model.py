# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: train.py
# Python  : python3.6
# Time    : 18-11-23 14:41

from keras.layers import LSTM, Input, Dense, Bidirectional, Embedding, \
    Dropout, Concatenate, RepeatVector, Activation, Dot, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import keras.backend as K
from gensim.models import word2vec
from sklearn.utils import shuffle
from seq2seq_model import Encoder, Decoder
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import json
from keras.optimizers import Adam
from keras.models import load_model
with open('config.json', 'r') as f:
    config = json.load(f)

with open('data/ch_voc', 'r') as f:
    ch_voc = json.load(f)

with open('data/en_voc', 'r') as f:
    en_voc = json.load(f)

def gen_datasets():
    encode_input = np.load('data/encode_input.npy')
    decode_input = np.load('data/decode_input.npy')
    decode_output = np.load('data/decode_output.npy')
    encode_input = pad_sequences(encode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_input = pad_sequences(decode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = pad_sequences(decode_output, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = np.expand_dims(decode_output, 2)
    return encode_input, decode_input, decode_output

def get_word2vector():
    ch_w2v_matrix = np.zeros((len(ch_voc), 100)) # vector size 100
    ch_w2v = word2vec.Word2Vec.load('data/ch_word2vec.model').wv
    for w, i in ch_voc.items():
        try:
            ch_w2v_matrix[i] = ch_w2v.get_vector(w)
        except:
            pass
    en_w2v_matrix = np.zeros((len(en_voc), 100))  # vector size 100
    en_w2v = word2vec.Word2Vec.load('data/en_word2vec.model').wv
    for w, i in en_voc.items():
        try:
            en_w2v_matrix[i] = en_w2v.get_vector(w)
        except:
            pass
    return ch_w2v_matrix, en_w2v_matrix

ch_w2v_matrix, en_w2v_matrix = get_word2vector()

def train():
    encode_input, decode_input, decode_output = gen_datasets()
    encode_input, decode_input, decode_output = shuffle(encode_input, decode_input, decode_output)

    encoder_input = Input(shape=(config['max_num'],), name='encode_input')
    embedded_input = Embedding(config['en_voc_size'], 100, weights=[en_w2v_matrix], trainable=False,
                               name="embedded_layer")(encoder_input)
    encoder = Encoder(6, 100, 5, 20, mask_zero=True, name='encoder')
    encoder_output = encoder([embedded_input, embedded_input, embedded_input, encoder_input, encoder_input])

    # decoder
    decoder_input = Input(shape=(config['max_num'],), name='decode_input')
    embedded_input2 = Embedding(config['ch_voc_size'], 100, weights=[ch_w2v_matrix], trainable=False,
                                name="embedded_layer2")(decoder_input)
    decoder = Decoder(6, 100, 5, 20, mask_zero=True, name='decoder')
    decoder_output = decoder([embedded_input2, encoder_output, encoder_output, decoder_input, encoder_input])
    decoder_dense = Dense(config['ch_voc_size'], activation='softmax', name='dense_layer')
    decoder_output = decoder_dense(decoder_output)
    model = Model([encoder_input, decoder_input], decoder_output)

    opt = Adam(lr=0.01, decay=0.04, clipnorm=1.0)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.summary()
    tb = TensorBoard(log_dir='./tb_logs/1026', histogram_freq=0, write_graph=True, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint('./models/attention_seq2seq.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)
    try:
        model.fit([encode_input, decode_input], decode_output, validation_split=0.2, batch_size=128, epochs=10, callbacks=[tb, cp])
    except KeyboardInterrupt:
        model.save('attention_seq2seq')
    else:
        model.save('attention_seq2seq')

def predict():
    model = load_model('attention_seq2seq', {'Encoder': Encoder, 'Decoder': Decoder})
    return model

if __name__ == '__main__':
    train()

