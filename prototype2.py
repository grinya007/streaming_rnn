#!/usr/bin/env python3

import argparse
import numpy as np
import torch as T
from data import read_files_dir, strip_words
from time import time
from vocabularies import StaticVocabulary, DynamicVocabulary

device = T.device("cuda")
LOOKBACK = 4
BATCHSIZE = 100
REPORTPPL = 100
REPORTLOSS = 1000
VOCABSIZE = 10000
FILLVOCAB = 300000
TRAINLIMIT = 3000000

class DS(T.utils.data.IterableDataset):
    def __init__(self, gen):
        super(DS).__init__()
        self.gen = gen
    def __iter__(self):
        for x, y in self.gen:
            yield T.tensor(x).to(device), T.tensor(y).to(device)

class RNN(T.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = T.nn.Embedding(num_embeddings, embedding_dim)
        self.gru = T.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = T.nn.Linear(hidden_dim, num_embeddings)
        self.softmax = T.nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        embeds = self.embed(x)
        out, nh = self.gru(embeds, h)
        logits = self.softmax(self.fc(out[:, -1]))
        return logits, nh

    def init_hidden(self, l):
        hidden = (
            T.zeros(self.n_layers, l, self.hidden_dim).to(device),
            T.zeros(self.n_layers, l, self.hidden_dim).to(device)
        )
        return hidden

def train_stream(input_dir, vocabulary, skip=0, stop=None):
    X = []
    words = 0
    for text in read_files_dir(input_dir):
        for word in strip_words(text):
            words += 1
            if words <= skip:
                continue
            if stop is not None and words > stop:
                break

            idx = vocabulary.word2idx(word)
            if len(X) <= LOOKBACK:
                X.append(idx)
            else:
                x = X.copy()
                yield x[:-1], x[-1]
                X.pop(0)
                X.append(idx)

def train(rnn, loader):
    criterion = T.nn.NLLLoss()
    optimizer = T.optim.Adam(rnn.parameters(), lr=0.001)

    ppl = []
    ti = time()
    counter = 0
    avg_loss = 0.
    rep_loss = 0.
    state_h, state_c = rnn.init_hidden(BATCHSIZE)
    for x, y in loader:
        counter += 1
        optimizer.zero_grad()
        out, (state_h, state_c) = rnn(x, (state_h, state_c))
        # print(out.shape, y.shape)
        loss = criterion(out, y.T)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        rep_loss += loss.item()
        if counter % REPORTPPL == 0:
            ppl.append(np.exp(rep_loss/REPORTPPL))
            rep_loss = 0.
        if counter % REPORTLOSS == 0:
            print("batch: {} (last {} batches in {:.2f} s) Average Loss: {}".format(counter, REPORTLOSS, time() - ti, avg_loss/REPORTLOSS))
            avg_loss = 0.
            ti = time()
    print(ppl)

def make_loader(input_dir, vocab):
    ds = DS(train_stream(input_dir, vocab, FILLVOCAB, TRAINLIMIT))
    loader = T.utils.data.DataLoader(ds, batch_size=BATCHSIZE, drop_last=True)
    return loader

def make_rnn():
    rnn = RNN(VOCABSIZE, 128, 256, 1)
    rnn.to(device)
    rnn.train()
    return rnn


def train_static(input_dir):
    static = StaticVocabulary(VOCABSIZE)

    # warm up
    i = 0
    words = []
    for text in read_files_dir(input_dir):
        for word in strip_words(text):
            if i == FILLVOCAB:
                break
            i += 1
            words.append(word)
    static.fill(words)

    train(make_rnn(), make_loader(input_dir, static))

def train_dynamic(input_dir):
    dynamic = DynamicVocabulary(VOCABSIZE, VOCABSIZE/4)

    # warm up
    i = 0
    for text in read_files_dir(input_dir):
        for word in strip_words(text):
            if i == FILLVOCAB:
                break
            i += 1
            dynamic.word2idx(word)

    train(make_rnn(), make_loader(input_dir, dynamic))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    train_static(args.input_dir)
    # train_dynamic(args.input_dir)




if __name__ == '__main__':
    main()


