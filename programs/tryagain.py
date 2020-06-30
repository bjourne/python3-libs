# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
'''Validation losses:

    nl nh d_mod  ffn bsz dout       lr s  ep25 ep100 ep200 best
     8 16   512  512  32  .15 0.00004 24 1.227 1.019 0.978 0.977
     8 16   512  512  32  .15 0.00008 24 1.097 0.977 0.972 0.972
     8 16   512 1024  32  .15 0.00010 29 1.044 0.958
     8 16   512 1024  32  .20 0.00010 29 1.068 0.960
     4 16   512 1024  32  .20 0.00010 16 1.134 1.004 0.980 0.979
     4 16   512 2048  32  .20 0.00010 18 1.086 0.979 0.968 0.968
     4 16   512 2048  32  .25 0.00020 18 1.050 0.969 0.961 0.961
     4 16  1024 2048  32  .25 0.00020 30 1.039 0.979 0.979 0.979
     8 16   512 2048  32  .20 0.00010 26 1.035 0.951
     8 16   512 2048  32  .30 0.00010 26 1.090 0.957 0.946 0.946
     8 16   512 2048  32  .40 0.00010 26 1.190 1.018 0.977 0.970

Where

 * nl is the number of layers,
 * nh is the number of heads,
 * d_mod is the data model,
 * ffn is the number of units in the point-wise feed forward layers,
 * bsz is the batch size,
 * dout is the dropout rate,
 * lr is the learning
 * s is the number of seconds per epoch and
 * epXXX is the validation loss after epoch XXX.
'''
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from observations import ptb
from tensorflow.config import *
from tensorflow.data import Dataset
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
SEQ_LEN = 320
N_LAYERS = 8
D_MODEL = 512
UNITS = 2048
N_HEADS = 16
EPS = 1e-6
DROPOUT = 0.40
VOCAB_SIZE = None
DEPTH = D_MODEL // N_HEADS
LR = 0.0001

def pos_encoding():
    pos = tf.range(5000, dtype = tf.float32)[:, tf.newaxis]
    i = tf.range(D_MODEL, dtype = tf.float32)[tf.newaxis, :]
    d_model_f32 = tf.cast(D_MODEL, tf.float32)
    angle_rads = pos / tf.pow(10_000, (2 * (i // 2)) / d_model_f32)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis = -1)
    return tf.expand_dims(pos_encoding, 0)

def scaled_dot_prod_attn(q, k, v, m):
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    depth_f32 = tf.cast(DEPTH, tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth_f32) + m * -1e9
    weights = softmax(logits, axis = -1)
    return tf.matmul(weights, v)

def split_heads(inp, batch_size):
    inp = tf.reshape(inp, (batch_size, -1, N_HEADS, DEPTH))
    return tf.transpose(inp, perm = [0, 2, 1, 3])

def transformer():
    # Input and look-ahead mask.
    inp = Input(shape = (None,))
    mask = 1 - tf.linalg.band_part(tf.ones((SEQ_LEN, SEQ_LEN)), -1, 0)

    random_uniform = RandomUniform(-0.1, 0.1)
    x = Embedding(VOCAB_SIZE, D_MODEL,
                  embeddings_initializer = random_uniform)(inp)
    x *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))

    # Hopefully this is only calculated once?
    x = x + pos_encoding()[:, :SEQ_LEN, :]
    # x = PositionalEncoding()(x)
    x = Dropout(DROPOUT)(x)
    batch_size = tf.shape(x)[0]
    for _ in range(N_LAYERS):
        # Multihead attention part
        wq = Dense(D_MODEL)(x)
        wk = Dense(D_MODEL)(x)
        wv = Dense(D_MODEL)(x)

        q = split_heads(wq, batch_size)
        k = split_heads(wk, batch_size)
        v = split_heads(wv, batch_size)

        attn = scaled_dot_prod_attn(q, k, v, mask)
        attn = tf.transpose(attn, perm = [0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, D_MODEL))
        attn = Dense(D_MODEL)(attn)
        attn = Dropout(DROPOUT)(attn)
        x = LayerNormalization(epsilon = EPS)(x + attn)

        # Point-wise feed-forward
        ffn = Dense(UNITS, activation = 'relu')(x)
        ffn = Dropout(DROPOUT)(ffn)
        ffn = Dense(D_MODEL)(ffn)
        ffn = Dropout(DROPOUT)(ffn)
        x = LayerNormalization(epsilon = EPS)(x + ffn)

    x = Dense(VOCAB_SIZE,
              activation = 'softmax',
              kernel_initializer = random_uniform)(x)
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

def sequence_to_samples(seq):
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    src = tf.constant(seq, dtype = tf.int32)
    return Dataset.from_tensor_slices(src) \
        .batch(SEQ_LEN + 1, drop_remainder = True) \
        .map(split_input_target) \
        .batch(BATCH_SIZE, drop_remainder = True)

train, _, valid = ptb('./data')
ch2ix = {c : i for i, c in enumerate(sorted(set(train)))}
VOCAB_SIZE = len(ch2ix)
with select_strategy().scope():
    model = transformer()
    model.compile(optimizer = RMSprop(learning_rate = LR),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['sparse_categorical_accuracy'])
def split_input_target(chunk):
    return chunk[:-1], chunk[1:]
train = sequence_to_samples([ch2ix[c] for c in train])
valid = sequence_to_samples([ch2ix[c] for c in valid])
model.summary()
model.fit(x = train, validation_data = valid, epochs = 500)
