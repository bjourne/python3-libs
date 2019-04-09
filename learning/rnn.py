# Copyright (C) 2018-2019 Björn Lindqvist <bjourne@gmail.com>
#
# RNN for text generation. The code is very much based on the one
# provided in "The Unreasonable Effectiveness of Recurrent Neural
# Networks" article:
#
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
from codecs import open
from numpy import *
from numpy.random import choice, randn
from sys import argv, exit

from random import uniform

def grad_check(state, X, Y, hprev):
    exc_fmt = 'Suspicious d%s: num %f, ana %f, rel %f'
    num_checks, delta = 10, 1e-5
    _, deriv, _ = state.loss_and_grad(X, Y, hprev)
    names = ('U', 'W', 'V', 'b', 'c')
    for p, dp, name in zip(state.params(), deriv.params(), names):
        for i in range(num_checks):
            ri = int(uniform(0, p.size))
            old_val = p.flat[ri]
            p.flat[ri] = old_val + delta
            cg0, _, _ = state.loss_and_grad(X, Y, hprev)
            p.flat[ri] = old_val - delta
            cg1, _, _ = state.loss_and_grad(X, Y, hprev)
            p.flat[ri] = old_val
            d_ana = dp.flat[ri]
            d_num = (cg0 - cg1) / ( 2 * delta )
            delta = abs(d_ana - d_num)
            rel = delta / abs(d_num + d_ana)
            if delta > 1e-5 and rel > 1e-5:
                raise Exception(exc_fmt % (name, d_num, d_ana, rel))

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
    def __init__(self, text):
        '''
        text: text to use as training data.
        '''
        self.text = text
        chars = list(set(self.text))
        self.ch2ix = {ch:i for i, ch in enumerate(chars)}
        self.ix2ch = {i:ch for i, ch in enumerate(chars)}

    def generate_samples(self, seq_len):
        for i in range(0, len(self.text) - seq_len - 1, seq_len):
            X = self.text[i : i + seq_len]
            Y = self.text[i + 1 : i + 1 + seq_len]
            yield self.encode(X), self.encode(Y)

    def encode(self, text):
        return [self.ch2ix[ch] for ch in text]

    def decode(self, vec):
        return ''.join(self.ix2ch[i] for i in vec)

class State:
    '''
    A two layer RNN.
    '''
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

    def sample(self, ix0, n):
        U, W, V, b, c = self.params()
        x = one_hot(self.K, ix0)
        h = copy(self.hprev)
        for t in range(n):
            h = tanh(dot(U, x) + dot(W, h) + b)
            y = dot(V, h) + c
            p = softmax(y)
            ix = choice(self.K, p = p.ravel())
            x = one_hot(self.K, ix)
            yield ix

    def loss_and_grad(self, X, Y):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = copy(self.hprev)
        n = len(X)

        # Forward pass
        for t in range(n):
            xs[t] = one_hot(self.K, X[t])
            hs[t] = tanh(dot(self.U, xs[t]) +
                         dot(self.W, hs[t - 1]) + self.b)
            ys[t] = dot(self.V, hs[t]) + self.c
        ps = {t : softmax(ys[t]) for t in range(n)}
        loss = -sum(log(ps[t][Y[t], 0]) for t in range(n))

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

def training(state, mem, training_data, seq_len, interval, gen_len, eta):
    '''Returns a generator that trains the rnn forever.

    state: the (initial) state of the rnn.
    mem: the (initial) state of the rnn's memory.
    training_data: object with data from which samples are generated.
    seq_len: sequence lengths...
    interval: dump some output every interval samples.
    gen_text: how much text to generate.
    eta: learning rate of something.
    '''
    smooth_loss = -log(1.0 / K) * seq_len
    n = 0
    while True:
        state.hprev = zeros_like(state.b)
        for X, Y in training_data.generate_samples(seq_len):
            if n % interval == 0:
                print('*** STEP %d LOSS %f ***' % (n, smooth_loss))
                vec = state.sample(X[0], gen_len)
                text = training_data.decode(vec)
                print(text)
                yield smooth_loss, text
            loss, deriv, state.hprev = state.loss_and_grad(X, Y)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            adagrad_update(state, deriv, mem, eta)
            n += 1

def draw_diagram(losses, interval):
    import matplotlib.pyplot as plt
    xaxis = arange(len(losses)) * interval
    xaxis = xaxis[20:]
    losses = losses[20:]
    plt.plot(xaxis, losses)
    plt.ylabel('Smooth loss')
    plt.xlabel('Step')
    plt.savefig('losses.png')

if __name__ == '__main__':
    if len(argv) != 2:
        print('usage %s: filename' % argv[0])
        exit(1)

    with open(argv[1], 'r', 'utf-8') as f:
        text = f.read()
    td = TrainingData(text)
    m, K = 100, len(td.ch2ix)
    state = State(m, K, 0.01)
    mem = State(m, K, 0)

    losses = []
    texts = []
    gen_int = 1000
    try:
        for loss, text in training(state, mem, td, 25, 1000, 200, 0.025):
            losses.append(loss)
            texts.append(text)
    except KeyboardInterrupt:
        print(texts)
        print(losses)
        draw_diagram(losses, gen_int)
        vec = state.sample(td.ch2ix['\n'], 1000)
        print(td.decode(vec))
