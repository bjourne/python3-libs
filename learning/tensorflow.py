# Copyright (C) 2020, 2022 Björn Lindqvist <bjourne@gmail.com>
'''
Machine-learning utilities for TensorFlow.
'''
from os import environ
from tensorflow.config import *
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np

# TensorFlow messes with imports so you have to import classes in this
# roundabout way.
import tensorflow as tf
OneDeviceStrategy = tf.distribute.OneDeviceStrategy
TPUClusterResolver = tf.distribute.cluster_resolver
TPUStrategy = tf.distribute.experimental


def select_strategy():
    '''Selects an appropriate execution strategy based on available
    devices.'''
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

def skew_top_p(P, top_p):
    '''
    Skew a probability distribution using nucleus sampling.
    '''
    prob_ixs = np.argsort(-P)
    PC = np.cumsum(P[prob_ixs])
    top_n = len(PC[PC < top_p]) + 1

    # Clear the prob of those who didn't make it.
    P[prob_ixs[top_n:]] = 0.0
    return P / P.sum()
