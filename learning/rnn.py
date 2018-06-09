from numpy import *
from numpy.random import choice, randn
from scipy.special import logsumexp
from sys import argv, exit

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

def softmax(x):
    # What kind of softmax do we want?
    x = exp(x - logsumexp(x, axis = 0))
    # log(0) errors are annoying
    x[x == 0] = finfo(float).eps
    return x

class ParamSet:
    def __init__(self, hidden_size, vocab_size, sigma):
        in_dim = (hidden_size, vocab_size)
        hidden_dim = (hidden_size, hidden_size)
        out_dim = (vocab_size, hidden_size)
        # First the three connection matrices, then the two bias
        # vectors.
        if not sigma:
            self.Wxh = zeros(in_dim)
            self.Whh = zeros(hidden_dim)
            self.Why = zeros(out_dim)
        else:
            self.Wxh = randn(*in_dim) * sigma
            self.Whh = randn(*hidden_dim) * sigma
            self.Why = randn(*out_dim) * sigma
        self.bh = zeros((hidden_size, 1))
        self.by = zeros((vocab_size, 1))

    def params(self):
        return (self.Wxh, self.Whh, self.Why, self.bh, self.by)

    def sample(self, h, seed_ix, n):
        '''
        h: Current hidden state.
        seed_ix: Given character
        n: Length of sequence to generate.
        '''
        x = zeros_like(self.by)
        x[seed_ix]= 1
        ixes = []
        for t in range(n):
            h = tanh(dot(self.Wxh, x) + dot(self.Whh, h) + self.bh)
            y = dot(self.Why, h) + self.by
            p = softmax(y)
            ix = choice(range(self.by.shape[0]), p = p.ravel())
            x = zeros_like(self.by)
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def loss_fun(self, X, Y, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = copy(hprev)
        loss = 0
        for t in range(len(X)):
            xs[t] = zeros_like(self.by)
            xs[t][X[t]] = 1
            hs[t] = tanh(dot(self.Wxh, xs[t]) +
                         dot(self.Whh, hs[t - 1]) +
                         self.bh)
            ys[t] = dot(self.Why, hs[t]) + self.by
            ps[t] = softmax(ys[t])
            # softmax (cross-entropy loss)
            loss += -log(ps[t][Y[t], 0])

        d = ParamSet(self.bh.shape[0], self.by.shape[0], 0)

        dhnext = zeros_like(hs[0])
        for t in reversed(range(len(X))):
            dy = copy(ps[t])
            # Backprop into y.
            dy[Y[t]] = -1
            d.Why += dot(dy, hs[t].T)
            d.by += dy
            # Backprop into h
            dh = dot(self.Why.T, dy) + dhnext
            # Backprop through tanh nonlinearity
            dhraw = (1 - hs[t] * hs[t]) * dh
            d.bh += dhraw
            d.Wxh += dot(dhraw, xs[t].T)
            d.Whh += dot(dhraw, hs[t - 1].T)
            dhnext = dot(self.Whh.T, dhraw)

        # Clip to mitigate exploding gradients
        dparams = (d.Wxh, d.Whh, d.Why, d.bh, d.by)
        for dparam in dparams:
            clip(dparam, -5, 5, out = dparam)
        return loss, dparams, hs[len(X) - 1]

class RNN:
    def __init__(self, hidden_size, vocab_size, sigma, eta):
        self.state = ParamSet(hidden_size, vocab_size, sigma)
        self.mem = ParamSet(hidden_size, vocab_size, 0)
        self.eta = eta
        self.hprev = zeros((hidden_size, 1))

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
            param += -self.eta * dparam / sqrt(mem + 1e-8)
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
