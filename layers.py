# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: attentions.py
# Python  : python3.6
# Time    : 18-11-23 11:36

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import ReLU, Lambda


class PositionEmbedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # inputs(batch_size, seq_len, model_size)
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(inputs.shape[-1])
        seq_len, model_size = K.shape(inputs)[1], K.shape(inputs)[2]
        pos = K.cast(K.expand_dims(K.arange(0, seq_len), -1), tf.float32)
        powers = K.expand_dims(K.pow(1000.0, tf.cast(2*tf.floor(K.arange(0, self.size)/2)/self.size, tf.float32)), 0)
        pe = pos*powers
        # the first time_step is all zeros
        sin = K.sin(pe[0::2])
        cos = K.cos(pe[1::2])
        pe = K.concatenate([sin, cos], axis=0)
        return inputs + pe

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

    def get_config(self):
        # this method is important for loading the model!
        # for more information, please refer to https://github.com/keras-team/keras/issues/7000
        config = {
            'size': self.size,
            'mode': self.mode,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, mask=False, selfmask_decoder=False, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head  # equals the model size
        self.mask = mask
        self.selfmask_decoder = selfmask_decoder
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape根据call中的输入自动得到
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        # todo: check if it's necessary
        # self.WO = self.add_weight(name='WO',
        #                           shape=(self.nb_head * self.size_per_head, self.output_dim),
        #                           initializer='glorot_uniform',
        #                           trainable=True)
        super(Attention, self).build(input_shape)

    def pad_mask(self, ini_input_q, ini_input_k):
        """method to create pad mask

        :param ini_input: initial input with the shape of [batch_size, seq_len]
        :return: mask
        """
        # ones(batch_size, seq_len_q, 1)
        ones = K.expand_dims(K.ones_like(ini_input_q, 'float32'), -1)
        # mask(batch_size, 1, seq_len_k)
        mask = K.expand_dims(K.cast(K.not_equal(ini_input_k, 0), 'float32'), 1)
        # mask(batch_size, seq_len_q, seq_len_k)
        mask = K.batch_dot(ones, mask, axes=[2, 1])
        return mask

    def sub_mask(self, ini_input):
        """method to create sub mask for self_attention in decoder layer

        :param ini_input: initial input with the shape of [batch_size, seq_len]
        :return: mask
        """
        # mask(batch_size, seq_len, seq_len)
        return K.cumsum(tf.eye(tf.shape(ini_input)[1], batch_shape=(tf.shape(ini_input)[0],)), axis=1)

    def mask_out(self, W, ini_input_q, ini_input_k, selfmask_decoder):
        """compute the output of W with the mask masked

        :param W: (nb_head*batch_size, seq_len_q, seq_len_k)
        :param ini_input_q: (batch_size, seq_len_q)
        :param ini_input_k: (batch_size, seq_len_k)
        :param selfmask_decoder: True if self_attention in decoder else False
        :return:
        """
        # mask(batch_size, seq_len_q, seq_len_k)
        mask = self.pad_mask(ini_input_q, ini_input_k)
        if selfmask_decoder:
            sub_mask = self.sub_mask(ini_input_k)
            mask = K.minimum(mask, sub_mask)
        # mask(nb_head*batch_size, seq_len_q, seq_len_k)
        mask = K.repeat_elements(mask, self.nb_head, axis=0)
        W = W - (1 - mask) * 1e10
        return W

    def dot_and_reshape(self, input, weight):
        """dot and reshape for the q,k,v

        :param input:
        :param weight:
        :return:
        """
        # do not use input.shape directly, or it will raise an error
        batch_size, seq_len = K.shape(input)[0], K.shape(input)[1]
        # output(batch_size, seq_len, nb_head*size_per_head)
        output = K.dot(input, weight)
        # output(batch_size, seq_len, nb_head, size_per_head)
        output = K.reshape(output, [batch_size, seq_len, self.nb_head, self.size_per_head])
        # output(nb_head, batch_size, seq_len, size_per_head)
        output = K.permute_dimensions(output, [2, 0, 1, 3])
        # output(nb_head*batch_size, seq_len, size_per_head)
        output = K.reshape(output, [-1, seq_len, self.size_per_head])
        return output

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.

        :param inputs: input tensor(s)
        :param kwargs: other params
        :return: output tensor
        """
        if self.mask:
            # Q_emb, K_emb, V_emb are the embedded inputs whit the shape of (batch_size, seq_len, emb_size)
            # Q_ini, K_ini are the initial inputs with the shape of (batch_size, seq_len)
            Q_emb, K_emb, V_emb, Q_ini, K_ini = inputs
        else:
            Q_emb, K_emb, V_emb = inputs[:3]
            Q_ini, K_ini = None, None
        # map Q_emb, K_emb, K_emb to the shape of (nb_head*batch_size, seq_len, size_per_head)
        Q_seq = Lambda(lambda x: self.dot_and_reshape(x[0], x[1]))([Q_emb, self.WQ])
        K_seq = Lambda(lambda x: self.dot_and_reshape(x[0], x[1]))([K_emb, self.WK]) # pay attention!!!
        V_seq = Lambda(lambda x: self.dot_and_reshape(x[0], x[1]))([V_emb, self.WV])
        # (nb_head*batch_size, seq_len_q, seq_len_k)
        W = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / self.size_per_head ** 0.5
        if self.mask:
            W = self.mask_out(W, Q_ini, K_ini, self.selfmask_decoder)
        # (nb_head*batch_size, seq_len_q, seq_len_k)
        W = K.softmax(W)
        # (nb_head*batch_size, seq_len_q, seq_len_k/v)*(nb_head*batch_size, seq_len_v, size_per_head) (2,1)
        # (nb_head*batch_size, seq_len_q, size_per_head)
        O_seq = K.batch_dot(W, V_seq, axes=[2, 1])
        # (nb_head, batch_size, seq_len_q, size_per_head)
        O_seq = K.reshape(O_seq, (self.nb_head, -1, O_seq.shape[1], O_seq.shape[2]))
        # (batch_size, seq_len_q, nb_head, head_size)
        O_seq = K.permute_dimensions(O_seq, (1, 2, 0, 3))
        # (batch_size, seq_len_q, nb_head*head_size)
        O_seq = K.reshape(O_seq, [K.shape(O_seq)[0], K.shape(O_seq)[1], -1])
        # (batch_size, seq_len_q, output_dim)
        # O_seq = K.dot(O_seq, self.WO)
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def get_config(self):
        # this method is important for loading the model!
        # for more information, please refer to https://github.com/keras-team/keras/issues/7000
        config = {
            'nb_head': self.nb_head,
            'size_per_head': self.size_per_head,
            'mask': self.mask,
            'selfmask_decoder': self.selfmask_decoder
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LayerNormalization(Layer):
    """normalization for each attention_sub layer and point_wise_forward sub_layer
    """

    def __init__(self, eps=1e-6, residual=True, **kwargs):
        self.eps = eps
        self.residual = residual
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.residual:  # 残差修正
            input_shape = input_shape[0]
        # scale and shift
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer='glorot_uniform', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer='glorot_uniform', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.residual:
            # residual connection which adds the output of the layer and the import of the layer
            inputs = inputs[0] + inputs[1]
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        if self.residual:
            input_shape = input_shape[0]
        return input_shape

    def get_config(self):
        # this method is important for loading the model!
        # for more information, please refer to https://github.com/keras-team/keras/issues/7000
        config = {
            'eps': self.eps,
            'residual': self.residual
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PWFF(Layer):
    """Position-wise Feed-Forward Networks
    """

    def __init__(self, **kwargs):
        super(PWFF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inner_w = self.add_weight(
            name='inner_w',
            shape=(input_shape[-1], 4 * input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.inner_b = self.add_weight(
            name='inner_b',
            shape=(4 * input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.outter_w = self.add_weight(
            name='outter_w',
            shape=(4 * input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.outter_b = self.add_weight(
            name='outter_b',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(PWFF, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner_output = K.bias_add(K.dot(inputs, self.inner_w), self.inner_b)
        inner_output = ReLU()(inner_output)
        output = K.bias_add(K.dot(inner_output, self.outter_w), self.outter_b)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class LabelSmooth(Layer):
    def __init__(self, e, K, **kwargs):
        self.e = e
        self.K = K
        super(LabelSmooth, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        # only takes effect in training phase
        def smooth_outputs():
            return (1 - self.e) * inputs + self.e / self.K

        return K.in_train_phase(smooth_outputs, inputs,
                                training=training)

    def get_config(self):
        # this method is important for loading the model!
        # for more information, please refer to https://github.com/keras-team/keras/issues/7000
        config = {
            'e': self.e,
            'K': self.K
        }
        base_config = super(LabelSmooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape