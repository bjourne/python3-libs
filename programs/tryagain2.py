# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# A minimal transformer - for demonstration purposes.
from observations import ptb
from os import environ
from tensorflow.config import *
from tensorflow.data import Dataset
from tensorflow.distribute import *
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from tensorflow.tpu.experimental import *
import tensorflow as tf

BATCH_SIZE = 32
LR = 0.0001
SEQ_LEN = 320
N_LAYERS = 8
D_MODEL = 512
UNITS = 512
N_HEADS = 8
EPS = 1e-6
DROPOUT = 0.0
VOCAB_SIZE = None
DEPTH = D_MODEL // N_HEADS

def transformer():
    # Input and look-ahead mask
    inp = Input(shape = (None,))
    mask = 1 - tf.linalg.band_part(tf.ones((SEQ_LEN, SEQ_LEN)), -1, 0)

    # Variables
    depth_f32 = tf.cast(DEPTH, tf.float32)
    d_model_f32 = tf.cast(D_MODEL, tf.float32)

    # Setup pos encoding
    pos = tf.range(5000, dtype = tf.float32)[:, tf.newaxis]
    i = tf.range(D_MODEL, dtype = tf.float32)[tf.newaxis, :]
    rads = pos / tf.pow(10_000, (2 * (i // 2)) / d_model_f32)
    sines = tf.math.sin(rads[:, 0::2])
    cosines = tf.math.cos(rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis = -1)
    pos_encoding = tf.expand_dims(pos_encoding, 0)

    random_uniform = RandomUniform(-0.1, 0.1)
    x = Embedding(VOCAB_SIZE, D_MODEL,
                  embeddings_initializer = random_uniform)(inp)
    x *= tf.math.sqrt(tf.cast(D_MODEL, tf.float32))

    # Shapes
    batch_size = tf.shape(x)[0]

    # Hopefully this is only calculated once?
    x = x + pos_encoding[:, :SEQ_LEN, :]
    x = Dropout(DROPOUT)(x)

    # For head splitting/merging
    split_pat = (batch_size, -1, N_HEADS, DEPTH)
    transp_pat = (0, 2, 1, 3)

    for _ in range(N_LAYERS):
        # Multihead attention part
        wq = Dense(D_MODEL)(x)
        wk = Dense(D_MODEL)(x)
        wv = Dense(D_MODEL)(x)

        # Split heads
        q = tf.transpose(tf.reshape(wq, split_pat), transp_pat)
        k = tf.transpose(tf.reshape(wk, split_pat), transp_pat)
        v = tf.transpose(tf.reshape(wv, split_pat), transp_pat)

        # Scaled dot product attention
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        logits = matmul_qk / tf.math.sqrt(depth_f32) + mask * -1e9
        weights = softmax(logits, axis = -1)
        attn = tf.matmul(weights, v)

        # Merge heads
        attn = tf.transpose(attn, transp_pat)
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

def main():
    global VOCAB_SIZE
    train, _, _ = ptb('./data')
    ch2ix = {c : i for i, c in enumerate(sorted(set(train)))}
    VOCAB_SIZE = len(ch2ix)
    train = [ch2ix[c] for c in train]
    with select_strategy().scope():
        model = transformer()
        model.compile(optimizer = RMSprop(learning_rate = LR),
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['sparse_categorical_accuracy'])
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    ds = Dataset.from_tensor_slices(train) \
                .batch(SEQ_LEN + 1, drop_remainder = True) \
                .map(split_input_target) \
                .batch(BATCH_SIZE, drop_remainder = True)
    model.fit(x = ds, epochs = 100)
main()
