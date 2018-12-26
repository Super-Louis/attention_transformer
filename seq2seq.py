# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: seq2seq.py
# Python  : python3.6
# Time    : 18-11-23 11:42
from layers import *
from keras.layers import Dropout, Layer

class Encoder():
    def __init__(self, layer_num, model_size, nb_head, size_per_head, seq_len, mask_zero=False, drop_out=0.1, **kwargs):
        self.layers = layer_num
        self.model_size = model_size
        self.dropout = drop_out
        self.mask = mask_zero
        self.heads = nb_head
        self.headsize = size_per_head
        self.seq_len = seq_len
        self.pos_embedding = PositionEmbedding(self.model_size, self.seq_len)
        self.Drop_out = Dropout(self.dropout)

    def __call__(self, inputs, **kwargs):
        if self.mask:
            # in the encoder, the Q_emb, K_emb, V_emb are the same; the Q_ini, K_ini are the same
            Q_emb, K_emb, V_emb, Q_ini, K_ini = inputs
        else:
            Q_emb, K_emb, V_emb = inputs
            Q_ini, K_ini = None, None
        Q_emb = self.pos_embedding(Q_emb)
        Q_emb = self.Drop_out(Q_emb)
        for _ in range(self.layers):

            attention = Attention(self.heads, self.headsize, self.mask)
            layer_normalization1 = LayerNormalization()
            layer_normalization2 = LayerNormalization()
            pwff = PWFF()

            O_seq = attention([Q_emb, K_emb, V_emb, Q_ini, K_ini])
            O_seq = self.Drop_out(O_seq)
            O_seq = layer_normalization1([O_seq, Q_emb])
            O = pwff(O_seq)
            O = self.Drop_out(O)
            O = layer_normalization2([O, O_seq])
            Q_emb, K_emb, V_emb = O, O, O
        return Q_emb


class Decoder():
    def __init__(self, layer_num, model_size, nb_head, size_per_head, seq_len, mask_zero=False, drop_out=0.1, **kwargs):
        self.layers = layer_num
        self.model_size = model_size
        self.dropout = drop_out
        self.mask = mask_zero
        self.heads = nb_head
        self.headsize = size_per_head
        self.seq_len = seq_len
        self.pos_embedding = PositionEmbedding(self.model_size, self.seq_len)
        self.Drop_out = Dropout(self.dropout)

    def __call__(self, inputs, **kwargs):
        if self.mask:
            # in the decoder, it includes two attention layers
            # in the self_attention layer, Q_emb, K_emb, V_emb are the same, and Q_ini, K_ini are the same
            # in the encoder_decoder layer, Q_emb and Q_ini are from the decoder, and K_emb, V_emb, K_ini are from the encoder
            Q_emb, K_emb, V_emb, Q_ini, K_ini = inputs
        else:
            Q_emb, K_emb, V_emb = inputs
            Q_ini, K_ini = None, None
        Q_emb = self.pos_embedding(Q_emb)
        Q_emb = self.Drop_out(Q_emb)
        for _ in range(self.layers):

            self_attention = Attention(self.heads, self.headsize, self.mask, selfmask_decoder=True)
            attention = Attention(self.heads, self.headsize, self.mask, selfmask_decoder=False)
            layer_normalization1 = LayerNormalization()
            layer_normalization2 = LayerNormalization()
            layer_normalization3 = LayerNormalization()
            pwff = PWFF()

            O_seq = self_attention([Q_emb, Q_emb, Q_emb, Q_ini, Q_ini]) # self attention
            O_seq = self.Drop_out(O_seq)
            O_seq = layer_normalization1([O_seq, Q_emb])
            O_seq2 = attention([O_seq, K_emb, V_emb, Q_ini, K_ini]) # encoder_decoder attention
            O_seq2 = self.Drop_out(O_seq2)
            O_seq2 = layer_normalization2([O_seq2, O_seq])
            O = pwff(O_seq2)
            O = self.Drop_out(O)
            O = layer_normalization3([O, O_seq2])
            Q_emb = O
        return Q_emb