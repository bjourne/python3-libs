# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""
Character-based language model in TensorFlow
============================================
A character-based language model written in TensorFlow and trained on
the Penn Treebank dataset. The model can either be trained using TFs
model.fit() method (auto) or using a custom training loop
(manual). Results should not differ.

Usage:
    char_lm_tf.py [options] ( manual | auto )

Options:
    -h --help               show this screen
    -v --verbose            print more output
    --batch-size=<i>        batch size [default: 32]
    --epochs=<i>            number of epochs [default: 200]
    --seq-len=<i>           sequence length [default: 320]

Validation losses:
             best   ep10   ep20   ep50   ep100
    auto    0.9483 1.0704 0.9892 0.9483 0.9483
    auto    0.9470 1.0924 1.0059 0.9496 0.9470
    auto    0.9483 1.0783 0.9732 0.9483 0.9483
    auto    0.9534 1.0626 0.9885 0.9534 0.9534
    auto    0.9544 1.0447 0.9770 0.9544 0.9544
    manual  0.9544 1.0595 0.9848 0.9544 0.9544
    manual  0.9483 1.1425 1.0478 0.9628 0.9483
    manual  0.9490 1.0937 1.0067 0.9516 0.9490
    manual  0.9491 1.2303 1.0864 0.9728 0.9491
    manual  0.9465 1.1122 1.0206 0.9520 0.9465
"""
from docopt import docopt
from observations import ptb
from os import environ
from pathlib import Path
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.data import Dataset
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import Model, losses, metrics
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.tpu.experimental import initialize_tpu_system
from time import time
import tensorflow as tf

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

def create_model(seq_len, vocab_size):
    inp = Input(shape = (seq_len,), batch_size = None, dtype = tf.int32)
    emb = Embedding(input_dim = vocab_size, output_dim = 100)
    lstm = LSTM(700, return_sequences = True, dropout = 0.3)
    time_dist = TimeDistributed(Dense(vocab_size, activation = 'softmax'))
    out = time_dist(lstm(emb(inp)))
    return MyModel(inputs = [inp], outputs = [out])

def sequence_to_samples(seq, seq_len):
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    src = tf.constant(seq, dtype = tf.int32)
    return Dataset.from_tensor_slices(src) \
        .batch(seq_len + 1, drop_remainder = True) \
        .map(split_input_target)

def select_strategy():
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        return OneDeviceStrategy(device = "/cpu:0")
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    return TPUStrategy(resolver)

def distribute_dataset(strategy, dataset, batch_size):
    def dataset_fn(ctx):
        return dataset.batch(batch_size, drop_remainder = True)
    return strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

class LossAccObserver:
    def __init__(self):
        self.loss = metrics.SparseCategoricalCrossentropy()
        self.acc = metrics.SparseCategoricalAccuracy()
    def reset(self):
        self.loss.reset_states()
        self.acc.reset_states()
    def update(self, y, y_hat):
        self.loss.update_state(y, y_hat)
        self.acc.update_state(y, y_hat)

@tf.function
def train_epoch(model, strategy, batch_size, dataset, obs):
    def step_fn(x, y):
        y_hat = compute_and_apply_gradients(model, x, y)
        obs.update(y, y_hat)
    for x, y in dataset:
        strategy.run(step_fn, args = (x, y))

@tf.function
def evaluate_epoch(model, strategy, dataset, obs):
    def step_fn(x, y):
        y_hat = model(x, training = False)
        obs.update(y, y_hat)
    for x, y in dataset:
        strategy.run(step_fn, args = (x, y))

def manual_training(model, strategy, train, valid, batch_size, epochs):
    with strategy.scope():
        train_obs = LossAccObserver()
        valid_obs = LossAccObserver()

    batch_size_per_replica = batch_size // strategy.num_replicas_in_sync
    train = distribute_dataset(strategy, train, batch_size_per_replica)
    valid = distribute_dataset(strategy, valid, batch_size_per_replica)

    fmt = '\-> %3d / %3d - %4db - %3ds - %.4f / %.4f - %.2f / %.2f %s'
    val_losses = []
    last_time = time()
    last_n_steps = 0
    for i in range(epochs):
        start = time()
        train_epoch(model, strategy, batch_size, train, train_obs)
        evaluate_epoch(model, strategy, valid, valid_obs)
        new_time = time()
        val_loss = valid_obs.loss.result()

        new_n_steps = model.optimizer.iterations.numpy()
        time_delta = new_time - last_time
        n_steps_delta = new_n_steps - last_n_steps
        mark = ' '
        if val_loss < min(val_losses, default = 100):
            mark = '*'
        args = (i + 1, epochs, n_steps_delta, time_delta,
                train_obs.loss.result(), val_loss,
                train_obs.acc.result(), valid_obs.acc.result(), mark)
        print(fmt % args)
        last_time = new_time
        last_n_steps = new_n_steps
        val_losses.append(val_loss)
        train_obs.reset()
        valid_obs.reset()

def automatic_training(model, train, valid, batch_size, epochs):
    train = train.batch(batch_size, drop_remainder = True)
    valid = valid.batch(batch_size, drop_remainder = True)
    model.fit(x = train, validation_data = valid,
              epochs = epochs,
              verbose = 2)

def main():
    # Parameters.
    args = docopt(__doc__, version = 'Char-based LM in TF 1.0')
    batch_size = int(args['--batch-size'])
    seq_len = int(args['--seq-len'])
    epochs = int(args['--epochs'])
    manual_mode = True if args['manual'] else False

    # Select strategy
    strategy = select_strategy()

    # Load and transform data.
    train, _, valid = ptb('./data')
    ix2ch = sorted(set(train))
    ch2ix = {c : i for i, c in enumerate(ix2ch)}
    train = sequence_to_samples([ch2ix[c] for c in train], seq_len)
    valid = sequence_to_samples([ch2ix[c] for c in valid], seq_len)
    vocab_size = len(ix2ch)

    # Create model and optimizer.
    with strategy.scope():
        model = create_model(seq_len, vocab_size)
        model.compile(
            optimizer = SGD(learning_rate = 4),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['sparse_categorical_accuracy'])

    if manual_mode:
        manual_training(model, strategy, train, valid, batch_size, epochs)
    else:
        automatic_training(model, train, valid, batch_size, epochs)

if __name__ == '__main__':
    main()
