# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
'''Transformer validation losses:

    nl nh dmod  ffn bsz dout       lr s  ep25 ep100 ep200 best
     8 16  512  512  32  .15 0.00004 24 1.227 1.019 0.978 0.977
     8 16  512  512  32  .15 0.00008 24 1.097 0.977 0.972 0.972
     8 16  512 1024  32  .15 0.00010 29 1.044 0.958
     8 16  512 1024  32  .20 0.00010 29 1.068 0.960
     4 16  512 1024  32  .20 0.00010 16 1.134 1.004 0.980 0.979
     4 16  512 2048  32  .20 0.00010 18 1.086 0.979 0.968 0.968
     4 16  512 2048  32  .25 0.00020 18 1.050 0.969 0.961 0.961
     4 16 1024 2048  32  .25 0.00020 30 1.039 0.979 0.979 0.979
     8 16  512 2048  32  .20 0.00010 26 1.035 0.951
     8 16  512 2048  32  .30 0.00010 26 1.090 0.957 0.946 0.946
     8 16  512 2048  32  .40 0.00010 26 1.190 1.018 0.977 0.970
     8 16  512 2048  32  .25 0.00010 26 1.062 0.949 0.946 0.946
     8 16  256 2048  32  .25 0.00010 22 1.142 0.974 0.934 0.918
     8 16  128 2048  32  .25 0.00010 19 1.286 1.051 0.999 0.962
     8 16  128 2048  32  .25 0.00020 19 1.140 0.998 0.966 0.952
     8 16  256 2048  32  .25 0.00020 21 1.056 0.944 0.922 0.912 ft 0.910
     8 16  256 2048  32  .25 0.00030 21 1.151 1.017 0.981 0.972
    12 16  256 2048  32  .25 0.00020 27 2.291
    12 16  256 2048  32  .25 0.00010 27 1.670 0.976 0.932 0.925
     8 16  192 2048  32  .25 0.00020 21 1.083 0.958 0.932 0.917
     8 16  192 2048  32  .20 0.00020 21 1.050 0.936 0.916 0.904 ft 0.902
     8 16  192 2048  32  .15 0.00020 23 1.031 0.939 0.921 0.917

Where

 * nl is the number of layers,
 * nh is the number of heads,
 * dmod is the data model,
 * ffn is the number of units in the point-wise feed forward layers,
 * bsz is the batch size,
 * dout is the dropout rate,
 * lr is the learning
 * s is the number of seconds per epoch and
 * epXXX is the validation loss after epoch XXX.
'''
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from learning.tensorflow import select_strategy
from observations import ptb
from pathlib import Path
from random import randrange
from tensorflow.data import Dataset
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from tqdm import trange

import numpy as np
import tensorflow as tf

# Common params
BATCH_SIZE = 32
SEQ_LEN = 320
DROPOUT = 0.15
VOCAB_SIZE = None
LR = 0.0002
WEIGHTS_PATH = Path('~/tryagain.h5').expanduser()
MODE = 'generation'

# Transformer params
N_LAYERS = 8
D_MODEL = 192
UNITS = 2048
N_HEADS = 16
EPS = 1e-6
DEPTH = D_MODEL // N_HEADS

# LSTM params
LSTM1_UNITS = 512
LSTM2_UNITS = 512
REC_DROPOUT = 0.25
EMB_SIZE = 100

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

def lstm():
    inp = Input(
        shape = (None,),
        batch_size = batch_size,
        dtype = tf.int32)
    embedding = Embedding(VOCAB_SIZE, EMB_SIZE)
    LSTM(LSTM1_UNITS, return_sequences = True,
         dropout = DROPOUT,
         recurrent_dropout = REC_DROPOUT)
    LSTM(LSTM2_UNITS, return_sequences = True,
         dropout = DROPOUT,
         recurrent_dropout = REC_DROPOUT)
    time_dist = TimeDistributed(Dense(VOCAB_SIZE))
    x = time_dist(lstm2(lstm1(embedding(inp))))
    return Model(inputs = inp, outputs = x)

def sequence_to_samples(seq):
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    src = tf.constant(seq, tf.int32)
    return Dataset.from_tensor_slices(src) \
        .batch(SEQ_LEN + 1, drop_remainder = True) \
        .map(split_input_target) \
        .batch(BATCH_SIZE, drop_remainder = True)

def train_model(train, valid):
    with select_strategy().scope():
        model = transformer()
        model.compile(optimizer = RMSprop(learning_rate = LR),
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['sparse_categorical_accuracy'])
    train = sequence_to_samples(train)
    valid = sequence_to_samples(valid)
    if WEIGHTS_PATH.exists():
        model.load_weights(str(WEIGHTS_PATH))
    cb_best = ModelCheckpoint(str(WEIGHTS_PATH),
                              verbose = 1,
                              save_weights_only = True,
                              save_best_only = True,
                              mode = 'min')
    model.fit(x = train, validation_data = valid, epochs = 500,
              verbose = 2, callbacks = [cb_best])

def generate_text():
    model = transformer()
    model.load_weights(str(WEIGHTS_PATH))

    idx = randrange(len(VALID) - SEQ_LEN)
    seed = np.array(VALID[idx : idx + SEQ_LEN])
    seed = np.expand_dims(seed, 0)
    top_p = 0.95

    ixs = []
    for _ in range(5000):
        P = model.predict(seed)[:, -1, :][0]

        prob_ixs = np.argsort(-P)
        PC = np.cumsum(P[prob_ixs])
        top_n = len(PC[PC <= top_p]) + 1
        surv_ixs = prob_ixs[:top_n]
        kill_ixs = prob_ixs[top_n:]

        P[kill_ixs] = 0.0
        P = P / P.sum()

        s1 = ''.join(IX2CH[ix] for ix in seed[0][-20:])
        s2 = ' '.join('%s %.2f' % (IX2CH[ix], P[ix]) for ix in surv_ixs)
        ix = np.random.choice(len(P), p = P)
        s3 = IX2CH[ix]
        print('%s | %s | %s' % (s1, s2, s3))

        seed = np.roll(seed, -1, axis = 1)
        seed[:, -1] = ix
        ixs.append(ix)
    s = ''.join(IX2CH[ix] for ix in ixs)
    print('RESULT: %s' % s)

train, _, valid = ptb('./data')
CH2IX = {c : i for i, c in enumerate(sorted(set(train)))}
IX2CH = {i : c for c, i in CH2IX.items()}
TRAIN = [CH2IX[c] for c in train]
VALID = [CH2IX[c] for c in valid]
VOCAB_SIZE = len(CH2IX)

if MODE == 'training':
    train_model()
else:
    generate_text()
