from numpy import *
from numpy.random import choice, randn
from scipy.special import logsumexp
from sys import argv, exit


def softmax(x):
    # What kind of softmax do we want?
    x = exp(x - logsumexp(x, axis = 0))
    # log(0) errors are annoying
    x[x == 0] = finfo(float).eps
    return x

def one_hot(n, i):
    '''
    Creates a one-hot column vector.
    '''
    x = zeros((n, 1))
    x[i] = 1
    return x

class TrainingData:
    def __init__(self, filename):
        self.data = open(filename, 'rb').read().decode('utf-8')
        self.data_size = len(self.data)
        self.vocab = list(set(self.data))
        self.vocab_size = len(self.vocab)

        self.ch2ix = {ch:ix for (ix, ch) in enumerate(self.vocab)}
        self.ix2ch = {ix:ch for (ix, ch) in enumerate(self.vocab)}

    def encode_text(self, text):
        return [self.ch2ix[ch] for ch in text]

    def decode_text(self, vec):
        return ''.join(self.ix2ch[i] for i in vec)

    def training_generator(self, seq_len):
        for i in range(0, self.data_size - seq_len - 1, seq_len):
            X = self.encode_text(self.data[i : i + seq_len])
            Y = self.encode_text(self.data[i + 1: i + 1 + seq_len])
            assert len(Y) == seq_len
            yield X, Y

class ParamSet:
    def __init__(self, m, K, sigma):
        # First the three connection matrices, then the two bias
        # vectors.
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

        self.m = m
        self.K = K

    def params(self):
        return self.U, self.W, self.V, self.b, self.c

    def sample(self, h, ix0, n):
        '''
        h: Current hidden state.
        ix0: Given character
        n: Length of sequence to generate.
        '''
        (U, W, V, b, c), K = self.params(), self.K
        x = one_hot(K, ix0)
        for t in range(n):
            h = tanh(dot(U, x) + dot(W, h) + b)
            o = dot(V, h) + c
            p = softmax(o)
            ix = choice(K, p = p.ravel())
            x = one_hot(K, ix)
            yield ix

    def loss_fun(self, X, Y, hprev):
        (U, W, V, b, c), K = self.params(), self.K
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = copy(hprev)
        loss = 0
        for t in range(len(X)):
            xs[t] = one_hot(K, X[t])
            hs[t] = tanh(dot(U, xs[t]) + dot(W, hs[t - 1]) + b)
            ys[t] = dot(V, hs[t]) + c
            ps[t] = softmax(ys[t])
            # softmax (cross-entropy loss)
            loss += -log(ps[t][Y[t], 0])

        d = ParamSet(self.m, K, 0)
        dhnext = zeros_like(hs[0])
        for t in reversed(range(len(X))):
            dy = copy(ps[t])
            # Backprop into y.
            dy[Y[t]] = -1
            d.V += dot(dy, hs[t].T)
            d.c += dy
            # Backprop into h
            dh = dot(V.T, dy) + dhnext
            # Backprop through tanh nonlinearity
            dhraw = (1 - hs[t] * hs[t]) * dh
            d.b += dhraw
            d.U += dot(dhraw, xs[t].T)
            d.W += dot(dhraw, hs[t - 1].T)
            dhnext = dot(W.T, dhraw)

        # Clip to mitigate exploding gradients
        dparams = d.U, d.W, d.V, d.b, d.c
        for dparam in dparams:
            clip(dparam, -5, 5, out = dparam)
        return loss, dparams, hs[len(X) - 1]

class RNN:
    def __init__(self, m, K, sigma, eta):
        self.state = ParamSet(m, K, sigma)
        self.mem = ParamSet(m, K, 0)
        self.eta = eta
        self.hprev = zeros((m, 1))

    def train_step(self, X, Y):
        '''
        Runs one training step. X is the input vector and Y the
        desired output vector.
        '''
        loss, dparams, self.hprev = self.state.loss_fun(X, Y, self.hprev)
        params = self.state.params()
        mems = self.mem.params()
        for param, dparam, mem in zip(params, dparams, mems):
            mem += dparam * dparam
            # Adagrad update. What is the best epsilon?
            param += -self.eta * dparam / sqrt(mem + 1e-6)
        return loss

if __name__ == '__main__':
    if len(argv) != 2:
        print('usage %s: filename' % argv[0])
        exit(1)
    td = TrainingData(argv[1])
    rnn = RNN(100, td.vocab_size, 0.01, 0.1)
    n = 0
    seq_length = 25
    # I don't know why the loss is initialized like this.
    smooth_loss = -log(1/td.vocab_size) * seq_length
    while True:
        # Reset RNN memory
        rnn.hprev = zeros_like(rnn.hprev)
        # Then run one epoch of training
        for X, Y in td.training_generator(seq_length):
            if n % 500 == 0:
                print('Iter %d, smooth loss %f' % (n, smooth_loss))
                vec = rnn.state.sample(rnn.hprev, X[0], 200)
                print(td.decode_text(vec))
            loss = rnn.train_step(X, Y)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            n += 1
