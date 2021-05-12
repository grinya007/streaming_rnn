#!/usr/bin/env python3

import argparse
import numpy as np
import torch as T
from collections import namedtuple
from data import read_texts_csv, strip_words
from time import time
from static_vocabulary import StaticVocabulary
from dynamic_vocabulary_2q import DynamicVocabulary2Q

device = T.device("cuda")
LOOKBACK = 10
BATCHSIZE = 100
REPORTLOSS = 1000
VOCABSIZE = 10000
FILLVOCAB = 300000
TRAINLIMIT = 200000


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
        self.gru = T.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.2)
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

def train_stream(text, vocabulary):
    X = []
    for word in strip_words(text):
        idx = vocabulary.word2idx(word)
        if len(X) <= LOOKBACK:
            X.append(idx)
        else:
            x = X.copy()
            yield x[:-1], x[-1]
            X.pop(0)
            X.append(idx)

def print_pred(rnn, vocab, orig):
    with T.no_grad():
        h = rnn.init_hidden(1)
        x = orig[:LOOKBACK]
        text = [vocab.idx2word(idx.item()) for idx in x]
        for i in range(LOOKBACK, 50):
            out, h = rnn(x.view(1, LOOKBACK), h)
            out = out[0][1:]
            p = T.nn.functional.softmax(out, dim=0).detach().cpu().numpy()
            idx = np.random.choice(range(1, VOCABSIZE), p=p)
            text.append(vocab.idx2word(idx))
            x = T.cat([x[1:], T.tensor([idx]).to(device)])

    print(' '.join(text), "\n", flush=True)

class TrainState:
    def __init__(self, time, counter, avg_loss, unknowns, ppl, uwr):
        self.time = time
        self.counter = counter
        self.avg_loss = avg_loss
        self.unknowns = unknowns
        self.ppl = ppl
        self.uwr = uwr

def train(rnn, criterion, optimizer, loader, vocab, state):
    state_h, state_c = rnn.init_hidden(BATCHSIZE)
    for x, y in loader:
        state.counter += 1
        optimizer.zero_grad()
        out, (state_h, state_c) = rnn(x, (state_h, state_c))
        loss = criterion(out, y.T)
        state.unknowns += (out.topk(1)[1] == 0).nonzero().shape[0]

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        optimizer.step()
        state.avg_loss += loss.item()
        if state.counter % REPORTLOSS == 0:
            tt = time() - state.time
            unk = state.unknowns/(REPORTLOSS*BATCHSIZE)
            state.uwr.append(unk)
            ll = state.avg_loss/REPORTLOSS
            ppla = np.exp(ll)
            pplr = ppla/VOCABSIZE
            ppla += VOCABSIZE * unk * pplr
            print("batch: {} (last {} batches in {:.2f} s), unknown word ratio: {:.3f}, average loss: {:.4f}, ppl: {:.4f}".format(
                state.counter, REPORTLOSS, tt, unk, ll, ppla
            ), flush=True)
            print_pred(rnn, vocab, y)
            state.ppl.append(ppla)
            state.avg_loss = 0.
            state.unknowns = 0
            state.time = time()
    # print(ppl, flush=True)
    # print(uwr, flush=True)
    return state

def make_loader(text, vocab):
    ds = DS(train_stream(text, vocab))
    loader = T.utils.data.DataLoader(ds, batch_size=BATCHSIZE, drop_last=True)
    return loader

def make_rnn():
    rnn = RNN(VOCABSIZE, 128, 256, 2)
    rnn.to(device)
    rnn.train()
    criterion = T.nn.NLLLoss()
    optimizer = T.optim.Adam(rnn.parameters(), lr=0.001)

    return rnn, criterion, optimizer

def full_train(text_gen, vocab):
    rnn, criterion, optimizer = make_rnn()
    state = TrainState(time(), 0, 0., 0, [], [])
    for text in text_gen:
        loader = make_loader(text, vocab)
        state = train(rnn, criterion, optimizer, loader, vocab, state)
        if state.counter*BATCHSIZE >= TRAINLIMIT:
            break

def train_static(input_csv):
    static = StaticVocabulary(VOCABSIZE)

    # warm up
    i = 0
    words = []
    text_gen = read_texts_csv(input_csv, 'content')
    for text in text_gen:
        if i >= FILLVOCAB:
            break
        for word in strip_words(text):
            words.append(word)
            i += 1
    static.fill(words)

    full_train(text_gen, static)

def train_dynamic(input_csv):
    dynamic = DynamicVocabulary2Q(VOCABSIZE)

    # warm up
    i = 0
    text_gen = read_texts_csv(input_csv, 'content')
    for text in text_gen:
        if i >= FILLVOCAB:
            break
        for word in strip_words(text):
            dynamic.word2idx(word)
            i += 1

    full_train(text_gen, dynamic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str)
    args = parser.parse_args()

    print("\tDYNAMIC\n", flush=True)
    train_dynamic(args.input_csv)
    print("\tSTATIC\n", flush=True)
    train_static(args.input_csv)




if __name__ == '__main__':
    main()


