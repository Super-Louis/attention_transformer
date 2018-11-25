# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: model.py
# Python  : python3.6
# Time    : 18-11-23 14:41
from layers import *
from keras.layers import LSTM, Input, Dense, Bidirectional, Embedding, \
    Dropout, Concatenate, RepeatVector, Activation, Dot, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import keras.backend as K
from gensim.models import word2vec
from sklearn.utils import shuffle
from seq2seq import Encoder, Decoder
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import json
from keras.optimizers import Adam
from keras.models import load_model
import nltk
from keras.objectives import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
import tensorflow as tf


with open('config.json', 'r') as f:
    config = json.load(f)

with open('data/ch_voc', 'r') as f:
    ch_voc = json.load(f)

with open('data/en_voc', 'r') as f:
    en_voc = json.load(f)

ch_voc_rev = {v:k for k, v in ch_voc.items()}
en_voc_rev = {v:k for k, v in en_voc.items()}

def gen_datasets():
    encode_input = np.load('data/encode_input.npy')
    decode_input = np.load('data/decode_input2.npy')
    decode_output = np.load('data/decode_output2.npy')
    encode_input = pad_sequences(encode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_input = pad_sequences(decode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = pad_sequences(decode_output, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = np.expand_dims(decode_output, 2)
    return encode_input, decode_input, decode_output

def get_word2vector():
    ch_w2v_matrix = np.zeros((len(ch_voc), 300)) # vector size 100
    ch_w2v = word2vec.Word2Vec.load('data/ch_word2vec2.model').wv
    for w, i in ch_voc.items():
        try:
            ch_w2v_matrix[i] = ch_w2v.get_vector(w)
        except:
            pass
    en_w2v_matrix = np.zeros((len(en_voc), 300))  # vector size 100
    en_w2v = word2vec.Word2Vec.load('data/en_word2vec2.model').wv
    for w, i in en_voc.items():
        try:
            en_w2v_matrix[i] = en_w2v.get_vector(w)
        except:
            pass
    return ch_w2v_matrix, en_w2v_matrix

ch_w2v_matrix, en_w2v_matrix = get_word2vector()

def mask_loss(y_true, y_pred):
    # y_true(batch_size, timestep, 1); y_pred(batch_size, timestep, voc_size)
    y_true = tf.cast(y_true, 'int32')
    # (batch_size, time_step)
    y_true = K.reshape(y_true, [-1, K.shape(y_pred)[1]])
    # (batch_size, time_step)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # (batch_size, time_step)
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    # loss*mask(batch_size, time_step); tf.reduce_sum(batch_size)
    loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
    # 1
    loss = K.mean(loss)
    return loss

def mask_accuracy(y_true, y_pred):
    # y_true(batch_size, timestep, 1); y_pred(batch_size, timestep, voc_size)
    y_true = K.reshape(y_true, [-1, K.shape(y_pred)[1]])
    mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
    corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
    corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
    return K.mean(corr)


def train():
    encode_input, decode_input, decode_output = gen_datasets()
    encode_input, decode_input, decode_output = shuffle(encode_input, decode_input, decode_output)

    encoder_input = Input(shape=(config['max_num'],), name='encode_input')
    embedded_input = Embedding(config['en_voc_size'], 300, weights=[en_w2v_matrix], trainable=False,
                               name="embedded_layer")(encoder_input)
    encoder = Encoder(6, 300, 6, 50, mask_zero=True, name='encoder')
    encoder_output = encoder([embedded_input, embedded_input, embedded_input, encoder_input, encoder_input])

    # decoder
    decoder_input = Input(shape=(config['max_num'],), name='decode_input')
    embedded_input2 = Embedding(config['ch_voc_size'], 300, weights=[ch_w2v_matrix], trainable=False,
                                name="embedded_layer2")(decoder_input)
    decoder = Decoder(6, 300, 6, 50, mask_zero=True, name='decoder')
    decoder_output = decoder([embedded_input2, encoder_output, encoder_output, decoder_input, encoder_input])
    decoder_dense = Dense(config['ch_voc_size'], activation='softmax', name='dense_layer')
    decoder_output = decoder_dense(decoder_output)
    label_smooth = LabelSmooth(0.2, config['ch_voc_size'])
    decoder_output = label_smooth(decoder_output)
    model = Model([encoder_input, decoder_input], decoder_output)

    opt = Adam(0.001, 0.9, 0.98, epsilon=1e-9)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    model.compile(optimizer=opt, loss=mask_loss, metrics=[mask_accuracy])

    model.summary()
    tb = TensorBoard(log_dir='./tb_logs/1125', histogram_freq=0, write_graph=True, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint('./models/attention_seq2seq.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)
    try:
        model.fit([encode_input, decode_input], decode_output, validation_split=0.2, batch_size=256, epochs=10, callbacks=[tb, cp])
    except KeyboardInterrupt:
        model.save('attention_seq2seq')
    else:
        model.save('attention_seq2seq')

def predict(inputs):
    model = load_model('attention_seq2seq', {
        'mask_loss': mask_loss,
        'mask_accuracy': mask_accuracy,
        'PositionEmbedding': PositionEmbedding,
        'Attention': Attention,
        'LayerNormalization': LayerNormalization,
        'PWFF': PWFF,
        'LabelSmooth': LabelSmooth
    })
    seq = inputs.strip().lower().replace(' - ', '-')
    words = nltk.word_tokenize(seq, )
    seqs = [en_voc.get(w, en_voc["<unk>"]) for w in words]
    encoder_inputs = pad_sequences([seqs], maxlen=config['max_num'], truncating='post', padding='post')
    # encoder_inputs = [[30, 16, 34, 248, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    decoder_inputs = [ch_voc['<start>']]
    decoder_inputs = pad_sequences([decoder_inputs], maxlen=config['max_num'], truncating='post', padding='post')
    # decoder_inputs = [[2, 8, 355, 43, 21, 229, 452, 14, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    outputs = ''
    # outputs = model.predict([encoder_inputs, decoder_inputs])
    # outputs = np.argmax(outputs, axis=-1)
    for i in range(config['max_num']-1):
        print(encoder_inputs, decoder_inputs)
        decoder_output = model.predict([encoder_inputs, decoder_inputs])
        print(decoder_output[0, i, 3])
        sampled_index = np.argmax(decoder_output[0, i, :])
        print(sampled_index)
        decoder_inputs[0, i+1] = sampled_index
        word = ch_voc_rev[sampled_index]
        outputs += word
        # if word == config['end_word']:
        #     break
    return outputs

if __name__ == '__main__':
    # print(predict("what is your first name?"))
    train()
