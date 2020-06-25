# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Character-based language model in PyTorch.
"""
Char-based LM in PyTorch
========================
Character based language models in PyTorch. This module contains
implementations based on recurrent networks (rnn), temporal
convolutional networks (tcn) and transformers. They are trained using
the Peen Treebank dataset.

Usage:
    char_lm_torch.py [options] <path> rnn [--hidden-size=<i>]
    char_lm_torch.py [options] <path> tcn
    char_lm_torch.py [options] <path> trans

Options:
    -h --help               show this screen
    -v --verbose            print more output
    --batch-size=<i>        batch size [default: 32]
    --epochs=<i>            number of epochs [default: 200]
    --em-size=<i>           character embedding size [default: 100]
    --seq-len=<i>           sequence length [default: 320]
    --log-interval=<i>      log every i:th minibatch [default: 200]
    --hidden-size=<i>       features in the hidden state [default: 700]

Validation losses:
             best    ep10    ep20    ep50    ep100  s/e
    trans   0.942   1.040   0.982   0.942    0.942   46
    tcn     0.955   1.055   1.002   0.969    0.955  188
    rnn     0.947   1.041   0.985   0.947    0.947   47

"""
from docopt import docopt
from observations import ptb
from os import cpu_count
from random import shuffle
from time import time
from torch import cuda, no_grad
from torch.nn import *
from torch.nn.utils import clip_grad_norm_, weight_norm
from torch.optim import Adam, SGD
import torch
import math

def positional_encoding(max_len, emb_size):
    '''Creates a (max_len, emb_size) tensor with precomputed values that
    are used for positional encodings.'''
    pe = torch.zeros(max_len, emb_size)
    # (max_len, 1)
    pos = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_size, 2).float() *
                         (-math.log(10000.0) / emb_size))
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

class TransformerModel(Module):
    def __init__(self,
                 vocab_size, emb_size,
                 nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.x_mask = None

        # Put the positional encoding in an untrainable buffer.
        pe = positional_encoding(5000, emb_size)
        self.register_buffer('pe', pe)
        self.dropout = Dropout(p = dropout)

        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid,
                                                 dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, nlayers)
        self.embedding = Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
        self.linear = Linear(emb_size, vocab_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).t()
        mask = mask.float().masked_fill(mask == 0, float('-inf'))\
                           .masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, state):
        x = x.t()
        if (self.x_mask is None or self.x_mask.size(0) != x.size(0)):
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size(0))
            mask = mask.to(device)
            self.x_mask = mask

        x = self.embedding(x) * math.sqrt(self.emb_size)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)

        out = self.transformer_encoder(x, self.x_mask)
        out = self.linear(out)
        out = out.transpose(0, 1)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        return out, state

    def init_state(self, batch_size, device):
        return []

# TCN module comes from https://github.com/locuslab/TCN
class Chomp1d(Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding,
                                        dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(dropout)

        self.conv2 = weight_norm(Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(dropout)

        self.net = Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = Conv1d(n_inputs, n_outputs, 1) \
            if n_inputs != n_outputs else None
        self.relu = ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(Module):
    def __init__(self, num_inputs, num_channels,
                 kernel_size=2,
                 dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels,
                kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(Module):
    def __init__(self,
                 vocab_size, embed_size,
                 num_channels,
                 kernel_size,
                 dropout, emb_dropout):
        super(TCN, self).__init__()
        self.encoder = Embedding(vocab_size, embed_size)
        self.tcn = TemporalConvNet(
            embed_size, num_channels,
            kernel_size=kernel_size,
            dropout=dropout)
        self.linear = Linear(embed_size, vocab_size)
        self.linear.weight = self.encoder.weight
        self.drop = Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, state):
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.linear(y.transpose(1, 2))
        return o.reshape(-1, o.size(2)), state

    def init_state(self, batch_size, device):
        return []

class RNN(Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 n_layers, emb_dropout):
        super(RNN, self).__init__()
        self.encoder = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size,
                         n_layers, batch_first = True)
        self.linear = Linear(hidden_size, vocab_size)
        self.drop = Dropout(emb_dropout)

    def forward(self, x, state):
        x2 = self.drop(self.encoder(x))
        out, state = self.lstm(x2, state)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out, state

    def init_state(self, batch_size, device):
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        hs = torch.zeros(num_layers, batch_size, hidden_size)
        cs = torch.zeros(num_layers, batch_size, hidden_size)
        return hs.to(device), cs.to(device)

def batchify(tensor, batch_size):
    # Cut off remainder
    n_batches = tensor.size(0) // batch_size
    tensor = tensor[:n_batches * batch_size]
    return tensor.view(batch_size, -1)

def successor_samples(batched_tensor, seq_len):
    for i in range(0, batched_tensor.size(1) - seq_len, seq_len):
        x = batched_tensor[:, i:i+seq_len]
        y = batched_tensor[:, (i+1):(i+1) + seq_len]
        yield x, y

def detach(state):
    return [s.detach() for s in state]

def train_epoch(model, opt, clip_norm, crit, state, samples,
                log_interval):
    accum_loss = 0
    last_time = time()
    n_samples = len(samples)
    log_fmt = '%5d / %5d | %3ds | %.3f'
    model.train()
    for i, (x, y) in enumerate(samples):
        y_hat, state = model(x, detach(state))
        loss = crit(y_hat, y.reshape(-1))
        accum_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()

        if (i + 1) % log_interval == 0:
            loss = accum_loss / log_interval
            elapsed = time() - last_time
            print(log_fmt % (i + 1, n_samples, elapsed, loss))
            accum_loss = 0
            last_time = time()

def evaluate_epoch(model, crit, state, samples):
    accum_loss = 0
    model.eval()
    with no_grad():
        for (x, y) in samples:
            y_hat, _ = model(x, detach(state))
            accum_loss += crit(y_hat, y.reshape(-1)).item()
    return accum_loss / len(samples)

def run_training(model_type, path,
                 batch_size, em_size, rnn_hidden_size,
                 seq_len,
                 log_interval, epochs):

    dev = 'cuda' if cuda.is_available() else 'cpu'
    dev = torch.device(dev)
    print('Device: %s, CPU count: %d' % (dev, cpu_count()))

    # Load data and create translation table.
    texts = ptb(path)
    ix2ch = sorted(set(texts[0]))
    ch2ix = {c : i for i, c in enumerate(ix2ch)}
    vocab_size = len(ix2ch)

    tensors = [torch.LongTensor([ch2ix[c] for c in text])
               for text in texts]
    tensors = [batchify(tensor, batch_size) for tensor in tensors]
    train, test, valid = tensors

    if model_type == 'rnn':
        model = RNN(vocab_size, em_size, rnn_hidden_size, 1, 0.1)
    elif model_type == 'tcn':
        n_levels = 3
        n_hidden = 450
        k_size = 3
        n_chans = [n_hidden] * (n_levels - 1) + [em_size]
        model = TCN(vocab_size, em_size, n_chans, k_size, 0.1, 0.1)
    else:
        model = TransformerModel(vocab_size, 100, 4, 500, 4, 0.0)

    model = model.to(dev)

    crit = CrossEntropyLoss()
    opt = SGD(model.parameters(), lr = 4)
    clip_norm = 0.15

    train = successor_samples(train, seq_len)
    valid = successor_samples(valid, seq_len)

    # Copy samples to the device
    train = [(x.to(dev), y.to(dev)) for (x, y) in train]
    valid = [(x.to(dev), y.to(dev)) for (x, y) in valid]

    for x, y in train + valid:
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)

    fmt = '\-> %2d / %2d - %3ds - %.4f - %.3f %s'
    losses = []
    for i in range(epochs):
        last_time = time()
        state = model.init_state(batch_size, dev)
        if model_type != 'rnn':
            shuffle(train)
        train_epoch(model, opt, clip_norm, crit, state, train,
                    log_interval)
        shuffle(valid)
        state = model.init_state(batch_size, dev)
        loss = evaluate_epoch(model, crit, state, valid)
        elapsed = time() - last_time
        lr = opt.param_groups[0]['lr']
        mark = ' '
        if loss < min(losses, default = 100):
            mark = '*'
        print(fmt % (i + 1, epochs, elapsed, lr, loss, mark))
        if i > 5 and loss > max(losses[-3:]):
            opt.param_groups[0]['lr'] /= 10
        losses.append(loss)

def colab_main():
    run_training('trans', '.', 32, 100, 512, 320, 200, 200)

def main():
    args = docopt(__doc__, version = 'Char-based LM 1.0')
    batch_size = int(args['--batch-size'])
    em_size = int(args['--em-size'])
    rnn_hidden_size = int(args['--hidden-size'])
    seq_len = int(args['--seq-len'])
    log_interval = int(args['--log-interval'])
    epochs = int(args['--epochs'])
    path = args['<path>']

    model_type = None
    for t in ['rnn', 'tcn', 'trans']:
        if args[t]:
            model_type = t
    run_training(model_type, path,
                 batch_size, em_size, rnn_hidden_size, seq_len,
                 log_interval, epochs)

if __name__ == '__main__':
    colab_main()
    #main()
