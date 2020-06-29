# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from observations import ptb
from tensorflow.config import *
from tensorflow.data import Dataset
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
SEQ_LEN = 320
N_LAYERS = 8
D_MODEL = 512
UNITS = 512
N_HEADS = 8
EPS = 1e-6
DROPOUT = 0.0
VOCAB_SIZE = None
DEPTH = D_MODEL // N_HEADS

def pos_encoding():
    pos = tf.range(5000, dtype = tf.float32)[:, tf.newaxis]
    i = tf.range(D_MODEL, dtype = tf.float32)[tf.newaxis, :]
    d_model_f32 = tf.cast(D_MODEL, tf.float32)
    angle_rads = pos / tf.pow(10_000, (2 * (i // 2)) / d_model_f32)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis = -1)
    return tf.expand_dims(pos_encoding, 0)

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = pos_encoding()
    def call(self, x):
        return x + self.pos_encoding[:, :SEQ_LEN, :]

def scaled_dot_prod_attn(q, k, v, m):
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    depth_f32 = tf.cast(DEPTH, tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth_f32) + m * -1e9
    weights = softmax(logits, axis = -1)
    return tf.matmul(weights, v)

def split_heads(inp, batch_size):
    inp = tf.reshape(inp, (batch_size, -1, N_HEADS, DEPTH))
    return tf.transpose(inp, perm = [0, 2, 1, 3])

class MultiHeadAttention(Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.wq = Dense(D_MODEL)
        self.wk = Dense(D_MODEL)
        self.wv = Dense(D_MODEL)
        self.dense = Dense(D_MODEL)
    def call(self, q, k, v, m):
        batch_size = tf.shape(q)[0]
        q = split_heads(self.wq(q), batch_size)
        k = split_heads(self.wk(k), batch_size)
        v = split_heads(self.wv(v), batch_size)
        attn = scaled_dot_prod_attn(q, k, v, m)
        attn = tf.transpose(attn, perm = [0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, D_MODEL))
        return self.dense(attn)

class TiedEmbeddingSoftmax(Layer):
    def __init__(self):
        super(TiedEmbeddingSoftmax, self).__init__()
        self.w = self.add_weight(shape = (VOCAB_SIZE, D_MODEL),
                                 initializer = 'random_normal',
                                 trainable = True)
        self.b = self.add_weight(shape = (VOCAB_SIZE,),
                                 initializer = 'zeros',
                                 trainable = True)
    def call(self, x, embed):
        if embed:
            return embedding_lookup(self.w, tf.cast(x, tf.int32))
        return tf.tensordot(x, tf.transpose(self.w), 1) + self.b

def transformer():
    # Input and look-ahead mask.
    inp = Input(shape = (None,))
    mask = 1 - tf.linalg.band_part(tf.ones((SEQ_LEN, SEQ_LEN)), -1, 0)

    x = Embedding(VOCAB_SIZE, D_MODEL)(inp)
    x *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    x = PositionalEncoding()(x)
    x = Dropout(DROPOUT)(x)
    for _ in range(N_LAYERS):
        normed = LayerNormalization(epsilon = EPS)(x)
        attn = MultiHeadAttention()(normed, normed, normed, mask)
        attn = Dropout(DROPOUT)(attn)
        x = x + attn

        normed = LayerNormalization(epsilon = EPS)(x)
        ffn = Dense(UNITS, activation = 'relu')(normed)
        ffn = Dense(D_MODEL)(ffn)
        ffn = Dropout(DROPOUT)(ffn)
        x = x + ffn
    x = Dense(VOCAB_SIZE, activation = 'softmax')(x)
    return Model(inputs = inp, outputs = x)

def select_strategy():
    gpus = list_physical_devices('GPU')
    print('%d GPU(s)' % len(gpus))
    for gpu in gpus:
        print('  %s' % (gpu,))
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        dev = '/GPU:0' if gpus else '/CPU:0'
        print('No TPU, using %s instead.' % dev)
        return OneDeviceStrategy(device = dev)
    print('TPU address: %s' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    strategy = TPUStrategy(resolver)
    tpus = list_logical_devices('TPU')
    print('%d TPU(s)' % len(tpus))
    for tpu in tpus:
        print('  %s' % (tpu,))
    return strategy

def main():
    global VOCAB_SIZE
    train, _, valid = ptb('./data')
    ch2ix = {c : i for i, c in enumerate(sorted(set(train)))}
    VOCAB_SIZE = len(ch2ix)
    train = [ch2ix[c] for c in train]
    with select_strategy().scope():
        model = transformer()
        opt = RMSprop(learning_rate = 0.00001)
        model.compile(optimizer = opt,
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['sparse_categorical_accuracy'])
    src = tf.constant(train, dtype = tf.int32)
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    ds = Dataset.from_tensor_slices(train) \
            .batch(SEQ_LEN + 1, drop_remainder = True) \
            .map(split_input_target) \
            .batch(BATCH_SIZE, drop_remainder = True)
    model.summary()
    model.fit(x = ds, epochs = 100)

if __name__ == '__main__':
    main()
