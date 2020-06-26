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
# environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
# Since this is just a toy example all configuration variables are
# globals.

BATCH_SIZE = 32
MAX_LEN = 40
N_LAYERS = 2
D_MODEL = 256
N_HEADS = 8
EPS = 1e-6
ADAM_EPS = 1e-9

# Hidden units
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
    print('%d synced replicas.' % strategy.num_replicas_in_sync)

    tpus = list_logical_devices('TPU')
    print('%d TPU(s)' % len(tpus))
    for tpu in tpus:
        print('  %s' % (tpu,))
    return strategy

########################################################################
# Transformer definition
########################################################################
def scaled_dot_prod_attn(q, k, v, m):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(q, k, transpose_b = True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if m is not None:
        logits += (m * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attn_weights = softmax(logits, axis=-1)

    return tf.matmul(attn_weights, v)

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding()

    def get_angles(self, position, i):
        d_model_f32 = tf.cast(D_MODEL, tf.float32)
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / d_model_f32)
        return position * angles

    def positional_encoding(self):
        angle_rads = self.get_angles(
            tf.range(VOCAB_SIZE, dtype = tf.float32)[:, tf.newaxis],
            tf.range(D_MODEL, dtype = tf.float32)[tf.newaxis, :])
        # Apply sin and cos to every other index.
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class MultiHeadAttention(Layer):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.query_dense = Dense(D_MODEL)
        self.key_dense = Dense(D_MODEL)
        self.value_dense = Dense(D_MODEL)
        self.dense = Dense(D_MODEL)

    def split_heads(self, inputs, batch_size):
        depth = D_MODEL // N_HEADS
        inputs = tf.reshape(
            inputs, shape = (batch_size, -1, N_HEADS, depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, m = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(q)[0]

        q = self.query_dense(q)
        k = self.key_dense(k)
        v = self.value_dense(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attn = scaled_dot_prod_attn(q, k, v, m)
        scaled_attn = tf.transpose(scaled_attn, perm = [0, 2, 1, 3])

        concat_attn = tf.reshape(scaled_attn, (batch_size, -1, D_MODEL))

        return self.dense(concat_attn)

def encoder_layer():
    inputs = Input(shape=(None, D_MODEL), name = "inputs")
    padding_mask = Input(shape=(1, 1, None), name = "padding_mask")

    attn = MultiHeadAttention()({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attn = Dropout(DROPOUT)(attn)
    attn = LayerNormalization(epsilon = EPS)(inputs + attn)

    outs = Dense(UNITS, activation='relu')(attn)
    outs = Dense(D_MODEL)(outs)
    outs = Dropout(DROPOUT)(outs)
    outs = LayerNormalization(epsilon = EPS)(attn + outs)

    return Model(inputs = [inputs, padding_mask], outputs = outs)

def encoder():
    inputs = Input(shape = (None,), name = "inputs")
    padding_mask = Input(shape = (1, 1, None), name = "padding_mask")

    embeddings = Embedding(VOCAB_SIZE, D_MODEL)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    embeddings = PositionalEncoding()(embeddings)

    outputs = Dropout(rate = DROPOUT)(embeddings)

    for i in range(N_LAYERS):
        outputs = encoder_layer()([outputs, padding_mask])

    return Model(inputs = [inputs, padding_mask], outputs = outputs)

def decoder_layer(name):
    inputs = Input(shape = (None, D_MODEL),
                   name = "inputs")
    enc_outputs = Input(shape = (None, D_MODEL),
                        name="encoder_outputs")
    look_ahead_mask = Input(shape = (1, None, None),
                            name = "look_ahead_mask")
    padding_mask = tf.keras.Input(shape = (1, 1, None),
                                  name = 'padding_mask')

    attn1 = MultiHeadAttention()(inputs={
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': look_ahead_mask
        })
    attn1 = LayerNormalization(epsilon = EPS)(attn1 + inputs)

    attn2 = MultiHeadAttention()(inputs={
            'query': attn1,
            'key': enc_outputs,
            'value': enc_outputs,
            'mask': padding_mask
        })
    attn2 = Dropout(DROPOUT)(attn2)
    attn2 = LayerNormalization(epsilon = EPS)(attn2 + attn1)

    outputs = Dense(UNITS, activation='relu')(attn2)
    outputs = Dense(D_MODEL)(outputs)
    outputs = Dropout(DROPOUT)(outputs)
    outputs = LayerNormalization(epsilon = EPS)(outputs + attn2)

    return Model(
        inputs = [inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)

def decoder():
    inputs = Input(shape = (None,), name = 'inputs')
    enc_outputs = Input(shape = (None, D_MODEL), name = 'encoder_outputs')
    look_ahead_mask = Input(shape = (1, None, None),
                            name = 'look_ahead_mask')
    padding_mask = tf.keras.Input(shape = (1, 1, None),
                                  name = 'padding_mask')

    embeddings = Embedding(VOCAB_SIZE, D_MODEL)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))
    embeddings = PositionalEncoding()(embeddings)

    outputs = Dropout(DROPOUT)(embeddings)

    for i in range(N_LAYERS):
        outputs = decoder_layer('decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name = 'decoder')

def transformer():
    inputs = Input(shape=(None,), name = 'inputs')
    dec_inputs = Input(shape=(None,), name = 'dec_inputs')

    enc_padding_mask = Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name = 'enc_padding_mask')(inputs)
    look_ahead_mask = Lambda(
        create_look_ahead_mask,
        output_shape = (1, None, None),
        name = 'look_ahead_mask')(dec_inputs)
    dec_padding_mask = Lambda(
        create_padding_mask,
        output_shape = (1, 1, None),
        name = 'dec_padding_mask')(inputs)

    enc_outputs = encoder()(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder()(inputs = [dec_inputs,
                                      enc_outputs,
                                      look_ahead_mask,
                                      dec_padding_mask])
    outputs = Dense(VOCAB_SIZE, name = 'outputs')(dec_outputs)

    return Model(inputs=[inputs, dec_inputs], outputs=outputs)

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
    line = line.strip()
    return line

def preprocess_dataset():
    file_name = 'dialogs.zip'
    file_path = get_file(file_name, origin = url, extract = True)
    ds_path = Path(file_path).parent / 'cornell movie-dialogs corpus'
    movie_lines_path = ds_path / 'movie_lines.txt'
    movie_convos_path = ds_path / 'movie_conversations.txt'

    with open(movie_lines_path, errors = 'ignore') as f:
        lines = f.readlines()
    id2line = {}
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

    with open(movie_convos_path, 'r') as f:
        lines = f.readlines()

    X, Y = [], []
    for line in lines[:5000]:
        parts = line.replace('\n', '').split(' +++$+++ ')
        convo = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(convo) - 1):
            x = preprocess_line(id2line[convo[i]])
            y = preprocess_line(id2line[convo[i+1]])
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
    y_true = tf.reshape(y_true, shape = (-1, y_true.shape[1]))
    loss = losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

def acc_fn(y_true, y_pred):
    y_true = tf.reshape(y_true, shape = (-1, y_true.shape[1]))
    return metrics.sparse_categorical_accuracy(y_true, y_pred)

class CustomSchedule(LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        a1 = tf.math.rsqrt(step)
        a2 = step * (self.warmup_steps**-1.5)
        d_model = tf.cast(D_MODEL, tf.float32)
        return tf.math.rsqrt(d_model) * tf.math.minimum(a1, a2)

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
    print('Got %d pairs.' % len(X))
    print('Building tokenizer...')
    cls = features.text.SubwordTextEncoder
    tokenizer = cls.build_from_corpus(X + Y, target_vocab_size = 2**13)
    VOCAB_SIZE = tokenizer.vocab_size + 2

    print('Vocab size %d.' % VOCAB_SIZE)
    X, Y = tokenize_lines(tokenizer, X, Y)

    strategy = select_strategy()
    with strategy.scope():
        model = transformer()
        lr = CustomSchedule()
        opt = Adam(lr, beta_1 = 0.9, beta_2 = 0.98,
                   epsilon = ADAM_EPS)
        model.compile(optimizer = opt, loss = loss_fn, metrics = [acc_fn])
        model.summary()

    X = pad_sequences(X, maxlen = MAX_LEN, padding = 'post')
    Y = pad_sequences(Y, maxlen = MAX_LEN, padding = 'post')
    pairs = ({'inputs' : X, 'dec_inputs' : Y[:, :-1]},
             {'outputs' : Y[:,1:]})
    ds = Dataset.from_tensor_slices(pairs) \
                .shuffle(10000) \
                .batch(BATCH_SIZE)

    lines = [
        'Where have you been?',
        'Will this program ever work?',
        'The summer is very hot.',
        'Say hello to my little friend.'
        ]
    for line in lines:
        evaluate(model, tokenizer, line)
    model.fit(ds, epochs = 20)
    for line in lines:
        evaluate(model, tokenizer, line)

main()
