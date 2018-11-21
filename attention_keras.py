#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import ReLU

class PositionEmbedding(Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)
        
    def call(self, inputs, **kwargs):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(inputs.shape[-1])
        batch_size,seq_len = K.shape(inputs)[0],K.shape(inputs)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(inputs[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + inputs
        elif self.mode == 'concat':
            return K.concatenate([position_ij, inputs], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
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
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        # shape inputs: (batch_size, step, step, nb_head)
        if seq_len == None:
            return inputs
        else:
            # (batch_size, step)
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            # [[1,1,1,0,0],
            #  [1,1,0,0,0],
            #  [1,1,1,1,0]]
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            # (batch_size, step, 1, 1)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, inputs, **kwargs):
        # Q_len 输出序列有效长度；V_len 输入序列有效长度
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if not isinstance(inputs, list):
            inputs = [inputs]
        if len(inputs) == 3:
            Q_seq,K_seq,V_seq = inputs
            Q_len,V_len = None,None
        elif len(inputs) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = inputs
        #对Q、K、V做线性变换
        # (batch_size, step, nb_head*head_size)
        Q_seq = K.dot(Q_seq, self.WQ)
        # (batch_size, step, nb_head, head_size)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        # (batch_size, nb_head, step, head_size)
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        # (batch_size, nb_head, step, head_size) * (batch_size, nb_head, step, head_size) (3*3)
        # (batch_size, nb_head, step, step)
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        # (batch_size, step, step, nb_head)
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add') # 令大于有效时间步的权重为0
        # (batch_size, nb_head, step, step)
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        # (batch_size, nb_head, step, step) * (batch_size, nb_head, step, head_size) (3*2)
        # (batch_size, nb_head, step, head_size)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        # (batch_size, step, nb_head, head_size)
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        # (batch_size, step, nb_head*head_size)
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul') # 令大于有效时间步的向量为0向量
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        if 'residual' in kwargs:
            self.residual = True
        else:
            self.residual = False
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.residual: # 残差修正
            input_shape = input_shape[0]
        # scale and shift
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer='glorot_uniform', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer='glorot_uniform', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.residual:
            inputs = inputs[0] + inputs[1]
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        return self.gamma * (inputs - mean)/(std+self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        if self.residual:
            input_shape = input_shape[0]
        return input_shape

class PWFF(Layer):

    def __init__(self, **kwargs):

        super(PWFF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inner_w = self.add_weight(
            name='inner_w',
            shape=(input_shape[-1], 4*input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.inner_b = self.add_weight(
            name='inner_b',
            shape=(4*input_shape[-1],),
            initializer='zeros',
            trainable=True
        ),
        self.outter_w = self.add_weight(
            name='outter_w',
            shape=(4*input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.outter_b = self.add_weight(
            name='outter_b',
            shape=(input_shape[-1], ),
            initializer='zeros',
            trainable=True
        )
        super(PWFF, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner_output = K.bias_add(K.dot(inputs, self.inner_w), self.inner_b)
        inner_output = ReLU(inner_output)
        output = K.bias_add(K.dot(inner_output, self.outter_w), self.outter_b)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

