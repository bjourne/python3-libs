# Copyright (C) 2018 Björn Lindqvist <bjourne@gmail.com>
#
# RNN for text generation. The code is very much based on the one
# provided in "The Unreasonable Effectiveness of Recurrent Neural
# Networks" article:
#
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#
# @  5000, loss = 59, 62, 60, 58, 62, 57
# @ 10000, loss = 56, 55, 52, 54, 54, 51, 57, 54, 58
# @ 20000, loss = 51, 48, 53
# @ 30000, loss = 50, 47, 51, 51, 51
# @ 40000, loss = 50
from numpy import *
from numpy.random import choice, randn
from sys import argv, exit

def one_hot(n, i):
    x = zeros((n, 1))
    x[i] = 1
    return x

def softmax(x):
    e_x = exp(x)
    return e_x / sum(e_x)

def adagrad_update(state, deriv, mem, eta):
    for p, dp, m in zip(state.params(), deriv.params(), mem.params()):
        m += dp * dp
        p += -eta * dp / sqrt(m + 1e-8)

class TrainingData:
    def __init__(self, filename):
        self.data = open(filename, 'r').read()
        chars = list(set(self.data))
        self.ch2ix = {ch:i for i, ch in enumerate(chars)}
        self.ix2ch = {i:ch for i, ch in enumerate(chars)}

    def generate_samples(self, seq_len):
        for i in range(0, len(self.data) - seq_len - 1, seq_len):
            X = self.data[i : i + seq_len]
            Y = self.data[i + 1 : i + 1 + seq_len]
            yield self.encode(X), self.encode(Y)

    def encode(self, text):
        return [self.ch2ix[ch] for ch in text]

    def decode(self, vec):
        return ''.join(self.ix2ch[i] for i in vec)

class State:
    def __init__(self, m, K, sigma):
        self.m = m
        self.K = K
        # First the three connection matrices.
        if not sigma:
            self.U = zeros((m, K))
            self.W = zeros((m, m))
            self.V = zeros((K, m))
        else:
            self.U = randn(m, K) * sigma
            self.W = randn(m, m) * sigma
            self.V = randn(K, m) * sigma
        # First and second bias vectors.
        self.b = zeros((m, 1))
        self.c = zeros((K, 1))

    def params(self):
        return self.U, self.W, self.V, self.b, self.c

    def sample(self, h, ix0, n):
        U, W, V, b, c = self.params()
        x = one_hot(self.K, ix0)
        for t in range(n):
            h = tanh(dot(U, x) + dot(W, h) + b)
            y = dot(V, h) + c
            p = softmax(y)
            ix = choice(self.K, p = p.ravel())
            x = one_hot(self.K, ix)
            yield ix

    def loss_and_grad(self, X, Y, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = copy(hprev)
        n = len(X)

        # Forward pass
        for t in range(n):
            xs[t] = one_hot(self.K, X[t])
            hs[t] = tanh(dot(self.U, xs[t]) +
                         dot(self.W, hs[t - 1]) + self.b)
            ys[t] = dot(self.V, hs[t]) + self.c
        ps = {t : softmax(ys[t]) for t in range(n)}
        loss = -sum(log(ps[t][Y[t],0]) for t in range(n))

        # Backward pass
        deriv = State(self.m, self.K, 0)
        dhnext = zeros_like(hs[0])
        for t in reversed(range(n)):
            dy = copy(ps[t])
            dy[Y[t]] -= 1
            deriv.V += dot(dy, hs[t].T)
            deriv.c += dy
            dh = dot(self.V.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            deriv.b += dhraw
            deriv.U += dot(dhraw, xs[t].T)
            deriv.W += dot(dhraw, hs[t-1].T)
            dhnext = dot(self.W.T, dhraw)
        for dp in deriv.params():
            clip(dp, -5, 5, out = dp)
        return loss, deriv, hs[n - 1]

def train_epoch(state, mem, training_data, seq_len, smooth_loss,
                gen_interval, gen_len):
    hprev = zeros_like(state.b)
    n = 0
    for X, Y in training_data.generate_samples(seq_len):
        if n % gen_interval == 0:
            print('*** ITER %d LOSS %f ***' % (n, smooth_loss))
            vec = state.sample(hprev, X[0], gen_len)
            print(training_data.decode(vec))
        loss, d, hprev = state.loss_and_grad(X, Y, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        adagrad_update(state, d, mem, 0.1)
        n += 1
    return smooth_loss

if __name__ == '__main__':
    if len(argv) != 2:
        print('usage %s: filename' % argv[0])
        exit(1)

    td = TrainingData(argv[1])
    m, K, seq_len = 100, len(td.ch2ix), 25
    state = State(m, K, 0.01)
    mem = State(m, K, 0)
    smooth_loss = -log(1.0 / K) * seq_len
    while True:
        smooth_loss = train_epoch(state, mem, td, seq_len, smooth_loss,
                                  1000, 200)
