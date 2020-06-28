# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
'''
Transformer impementation. Most code comes from:

    https://medium.com/tensorflow/ \
        a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2

Variable names:

 * k: key
 * m: mask
 * q: query
 * v: value
'''
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from re import sub
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices,
                               list_physical_devices)
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import *
from tensorflow.keras import losses, metrics
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import get_file
from tensorflow.nn import softmax
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow_datasets import features
import tensorflow as tf

########################################################################
# Configuration
########################################################################
# Since this is a toy example all configuration variables are
# globals.

BATCH_SIZE = 32
MAX_LEN = 40
N_LAYERS = 2
D_MODEL = 256
N_HEADS = 8
EPS = 1e-6
ADAM_EPS = 1e-9

# Hidden units. Corresponds to "dff" I think.
UNITS = 512

DROPOUT = 0.1

# Calculated when loading the dataset
VOCAB_SIZE = None

assert D_MODEL % N_HEADS == 0

########################################################################
# TPU helpers
########################################################################
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

########################################################################
# Transformer definition
########################################################################
def scaled_dot_prod_attn(q, k, v, m):
    """
    Calculate the attention weights. Implements the formula:

        softmax(m*q*k^t/sqrt(d_k))*V,

    where m is masking out padding tokens. Shapes:

        m: (1, 1, 1, s)
    """
    matmul_qk = tf.matmul(q, k, transpose_b = True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    logits += (m * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attn_weights = softmax(logits, axis = -1)

    return tf.matmul(attn_weights, v)

def create_padding_mask(x):
    '''
    One where the tensor is equal to 0. Shapes:

        x: (1, s)
    '''
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    ones_mat = tf.ones((seq_len, seq_len))
    look_ahead_mask = 1 - tf.linalg.band_part(ones_mat, -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

def get_angles(position, i):
    d_model_f32 = tf.cast(D_MODEL, tf.float32)
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / d_model_f32)
    return position * angles

def positional_encoding():
    angle_rads = get_angles(
        tf.range(VOCAB_SIZE, dtype = tf.float32)[:, tf.newaxis],
        tf.range(D_MODEL, dtype = tf.float32)[tf.newaxis, :])

    # Apply sin and cos to every other index.
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = positional_encoding()

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttention(Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.wq = Dense(D_MODEL)
        self.wk = Dense(D_MODEL)
        self.wv = Dense(D_MODEL)
        self.dense = Dense(D_MODEL)

    def split_heads(self, inputs, batch_size):
        depth = D_MODEL // N_HEADS
        inputs = tf.reshape(inputs, (batch_size, -1, N_HEADS, depth))
        return tf.transpose(inputs, perm = [0, 2, 1, 3])

    def call(self, q, k, v, m):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attn = scaled_dot_prod_attn(q, k, v, m)
        scaled_attn = tf.transpose(scaled_attn, perm = [0, 2, 1, 3])

        concat_attn = tf.reshape(scaled_attn, (batch_size, -1, D_MODEL))

        return self.dense(concat_attn)

def encoder_layer():
    inp = Input(shape = (None, D_MODEL))
    inp_mask = Input(shape = (1, 1, None))

    attn = MultiHeadAttention()(inp, inp, inp, inp_mask)
    attn = Dropout(DROPOUT)(attn)
    attn = LayerNormalization(epsilon = EPS)(inp + attn)

    # Point-wise feed-forward
    out = Dense(UNITS, activation = 'relu')(attn)
    out = Dense(D_MODEL)(out)
    out = Dropout(DROPOUT)(out)
    out = LayerNormalization(epsilon = EPS)(attn + out)

    return Model(inputs = [inp, inp_mask], outputs = out)

def decoder_layer():
    inp = Input(shape = (None, D_MODEL))
    enc_out = Input(shape = (None, D_MODEL))
    look_ahead_mask = Input(shape = (1, None, None))
    padding_mask = tf.keras.Input(shape = (1, 1, None))

    attn1 = MultiHeadAttention()(inp, inp, inp, look_ahead_mask)
    attn1 = LayerNormalization(epsilon = EPS)(attn1 + inp)

    attn2 = MultiHeadAttention()(attn1, enc_out, enc_out, padding_mask)
    attn2 = Dropout(DROPOUT)(attn2)
    attn2 = LayerNormalization(epsilon = EPS)(attn2 + attn1)

    # Point-wise feed-forward
    out = Dense(UNITS, activation='relu')(attn2)
    out = Dense(D_MODEL)(out)
    out = Dropout(DROPOUT)(out)
    out = LayerNormalization(epsilon = EPS)(attn2 + out)

    return Model(inputs = [inp, enc_out, look_ahead_mask, padding_mask],
                 outputs = out)

def encoder():
    # Layer definitions.
    inp = Input(shape = (None,))
    inp_mask = Input(shape = (1, 1, None))

    emb = Embedding(VOCAB_SIZE, D_MODEL)(inp)
    emb *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    emb = PositionalEncoding()(emb)
    out = Dropout(DROPOUT)(emb)

    for _ in range(N_LAYERS):
        out = encoder_layer()([out, inp_mask])
    return Model(inputs = [inp, inp_mask], outputs = out)

def decoder():
    inp = Input(shape = (None,))
    enc_outputs = Input(shape = (None, D_MODEL))
    look_ahead_mask = Input(shape = (1, None, None))
    padding_mask = tf.keras.Input(shape = (1, 1, None))

    emb = Embedding(VOCAB_SIZE, D_MODEL)(inp)
    emb *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    emb = PositionalEncoding()(emb)
    out = Dropout(DROPOUT)(emb)

    for i in range(N_LAYERS):
        out = decoder_layer()(inputs = [out,
                                        enc_outputs,
                                        look_ahead_mask,
                                        padding_mask])

    return Model(
        inputs = [inp, enc_outputs, look_ahead_mask, padding_mask],
        outputs = out)

def transformer():
    inp = Input(shape=(None,), name = 'inp')
    dec_inp = Input(shape=(None,), name = 'dec_inp')

    enc_padding_mask = Lambda(create_padding_mask,
                              output_shape = (1, 1, None))(inp)
    enc_out = encoder()(inputs = [inp, enc_padding_mask])

    look_ahead_mask = Lambda(create_look_ahead_mask,
                             output_shape = (1, None, None))(dec_inp)
    dec_padding_mask = Lambda(create_padding_mask,
                              output_shape = (1, 1, None))(inp)
    dec_out = decoder()(inputs = [dec_inp, enc_out,
                                  look_ahead_mask,
                                  dec_padding_mask])
    outputs = Dense(VOCAB_SIZE, name = 'out')(dec_out)

    return Model(inputs=[inp, dec_inp], outputs=outputs)

########################################################################
# Data processing
########################################################################

url = 'http://www.cs.cornell.edu/~cristian/data/' \
    + 'cornell_movie_dialogs_corpus.zip'

def preprocess_line(line):
    line = line.lower().strip()
    line = sub(r"([?.!,])", r" \1 ", line)
    line = sub(r'[" "]+', " ", line)
    line = sub(r"[^a-zA-Z?.!,]+", " ", line)
    return line.strip()

def preprocess_dataset():
    file_name = 'dialogs.zip'
    file_path = get_file(file_name, origin = url, extract = True)
    ds_path = Path(file_path).parent / 'cornell movie-dialogs corpus'
    movie_lines_path = ds_path / 'movie_lines.txt'
    movie_convos_path = ds_path / 'movie_conversations.txt'
    def split_line(line):
        return [part.strip() for part in line.split(' +++$+++ ')]

    with open(movie_lines_path, errors = 'ignore') as f:
        lines = f.readlines()
    lines = [split_line(line) for line in lines]
    id2line = {parts[0] : preprocess_line(parts[4]) for parts in lines}

    with open(movie_convos_path, 'r') as f:
        lines = f.readlines()

    X, Y = [], []
    for line in lines[:500]:
        parts = split_line(line)
        convo = [id2line[line[1:-1]]
                 for line in parts[3][1:-1].split(', ')]
        for x, y in zip(convo, convo[1:]):
            X.append(x)
            Y.append(y)
    return X, Y

def tokenize_line(tokenizer, x):
    sos = tokenizer.vocab_size
    eos = tokenizer.vocab_size + 1
    return [sos] + tokenizer.encode(x) + [eos]

def tokenize_lines(tokenizer, X, Y):
    print('Tokenizing %s pairs...' % len(X))
    X2, Y2 = [], []
    for x, y in zip(X, Y):
        x = tokenize_line(tokenizer, x)
        y = tokenize_line(tokenizer, y)
        if len(x) <= MAX_LEN and len(y) <= MAX_LEN:
            X2.append(x)
            Y2.append(y)
    return X2, Y2

########################################################################
# Training settings
########################################################################
def loss_fn(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, y_true.shape[1]))
    loss = losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

def acc_fn(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, y_true.shape[1]))
    return metrics.sparse_categorical_accuracy(y_true, y_pred)

class CustomSchedule(LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        a1 = tf.math.rsqrt(step)
        a2 = step * (self.warmup_steps**-1.5)
        d_model_f32 = tf.cast(D_MODEL, tf.float32)
        return tf.math.rsqrt(d_model_f32) * tf.math.minimum(a1, a2)

########################################################################
# Inference
########################################################################
def evaluate(model, tokenizer, line1):
    sos = tokenizer.vocab_size
    eos = tokenizer.vocab_size + 1

    x = preprocess_line(line1)
    x = tokenize_line(tokenizer, x)
    x = tf.expand_dims(x, axis = 0)

    y = tf.expand_dims([tokenizer.vocab_size], 0)
    for i in range(MAX_LEN):
        y_hat = model(inputs = [x, y], training = False)
        y_hat = y_hat[:, -1:, :]
        id = tf.cast(tf.argmax(y_hat, axis = -1), tf.int32)
        if tf.equal(id, eos):
            break
        y = tf.concat([y, id], axis = -1)
    y = tf.squeeze(y, axis = 0)
    line2 = tokenizer.decode(y[1:])
    print('%s => %s' % (line1, line2))

def main():
    global VOCAB_SIZE

    print('Preprocessing dataset...')
    X, Y = preprocess_dataset()
    print('%d pairs, building tokenizer...' % len(X))
    cls = features.text.SubwordTextEncoder
    tokenizer = cls.build_from_corpus(X + Y, target_vocab_size = 2**13)
    VOCAB_SIZE = tokenizer.vocab_size + 2

    print('Vocab size %d.' % VOCAB_SIZE)
    X, Y = tokenize_lines(tokenizer, X, Y)
    X = pad_sequences(X, maxlen = MAX_LEN, padding = 'post')
    Y = pad_sequences(Y, maxlen = MAX_LEN, padding = 'post')

    strategy = select_strategy()
    with strategy.scope():
        print('Creating the transformer.')
        model = transformer()
        lr = CustomSchedule()
        opt = Adam(lr, beta_1 = 0.9, beta_2 = 0.98,
                   epsilon = ADAM_EPS)
        model.compile(optimizer = opt, loss = loss_fn, metrics = [acc_fn])
        model.summary()

    pairs = ({'inp' : X, 'dec_inp' : Y[:, :-1]}, {'out' : Y[:,1:]})
    ds = Dataset.from_tensor_slices(pairs) \
                .batch(BATCH_SIZE, drop_remainder = True)

    lines = [
        'Where have you been?',
        'Will this program ever work?',
        'The summer is very hot.',
        'Say hello to my little friend.',
        "It's time to kickass and chew bubble gum and I'm "
        "all out of bubble gum"
        ]
    for line in lines:
        evaluate(model, tokenizer, line)
    model.fit(ds, epochs = 20)
    for line in lines:
        evaluate(model, tokenizer, line)

main()
