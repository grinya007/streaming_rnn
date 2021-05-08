#!/usr/bin/env python3

import argparse
import numpy as np
import os
import re
import torch as T
from cache import Cache
from pathlib import Path
from random import random
from time import time

def read_files(input_dir):
    path = Path(input_dir)
    if not path.is_dir():
        return ''
    for text_file in path.iterdir():
        if not text_file.name.endswith('.txt'):
            continue
        with text_file.open() as f:
            while f.tell() < os.fstat(f.fileno()).st_size:
                line = f.readline()
                if line.startswith('Page |'):
                    continue
                yield line

def strip_words(text):
    for match in re.finditer(r"[-'’a-zA-Z0-9]+|[\.\,\?\!\:\;\(\)]", text):
        words = match.group(0).lower()
        if words == 'br':
            continue
        for word in re.split(r'-{2,}', words):
            yield word

LOOKBACK = 5
CACHESIZE = 10000
cache = Cache(CACHESIZE)
def train_stream(input_dir, skip=0):
    X = []
    c = 0
    words = 0
    hits = 0
    for text in read_files(input_dir):
        for word in strip_words(text):
            c += 1
            if c <= skip:
                continue
            words += 1
            result = cache.get_replace(word)
            hits += 1 if result.is_hit else 0
            if words % 100000 == 0:
                print("Cache hit ratio: {:.4f}".format(hits/words))
                words = 0
                hits = 0
            if len(X) <= LOOKBACK:
                X.append(result.idx)
            else:
                x = X.copy()
                yield x[:-1], x[1:]
                X.pop(0)
                X.append(result.idx)

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

    def forward(self, x, h):
        embeds = self.embed(x)
        out, nh = self.gru(embeds, h)
        logits = self.fc(out)
        return logits, nh

    def init_hidden(self, l):
        hidden = (
            T.zeros(self.n_layers, l, self.hidden_dim).to(device),
            T.zeros(self.n_layers, l, self.hidden_dim).to(device)
        )
        return hidden

BATCHSIZE = 100
ppl = []
def train(rnn, loader, epoch):
    criterion = T.nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(rnn.parameters(), lr=0.001)

    ti = time()
    counter = 0
    avg_loss = 0.
    rep_loss = 0.
    state_h, state_c = rnn.init_hidden(BATCHSIZE)
    for x, y in loader:
        counter += 1
        optimizer.zero_grad()
        out, (state_h, state_c) = rnn(x, (state_h, state_c))
        # print(out.shape, out.transpose(1, 2).shape, y.shape)
        loss = criterion(out.transpose(1, 2), y)

        state_h = state_h.detach()
        state_c = state_c.detach()

        loss.backward()
        # T.nn.utils.clip_grad_norm_(rnn.parameters(), 0.5)
        optimizer.step()
        avg_loss += loss.item()
        rep_loss += loss.item()
        if counter % 100 == 0:
            ppl.append(np.exp(rep_loss/100))
            rep_loss = 0.
        if counter % 1000 == 0:
            print("epoch: {} batch: {} (last 1000 batches in {:.2f} s) Average Loss: {}".format(epoch, counter, time() - ti, avg_loss/counter))
            ti = time()
    print(ppl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    # warm up
    i = 0
    for _ in train_stream(args.input_dir):
        if i == 300000:
            break
        i += 1

    rnn = RNN(CACHESIZE, 128, 256, 1)
    rnn.to(device)
    rnn.train()
    try:
        for e in range(1):
            ds = DS(train_stream(args.input_dir, 300000))
            loader = T.utils.data.DataLoader(ds, batch_size=BATCHSIZE, drop_last=True)
            train(rnn, loader, e)
    except KeyboardInterrupt:
        pass

    rnn.eval()
    while True:
        seed = input('5 words: ')
        seed_x = []
        for word in strip_words(seed):
            result = cache.get_by_key(word)
            if result is None:
                break
            seed_x.append(result.idx)
        if len(seed_x) == 0:
            continue

        text = []
        state_h, state_c = rnn.init_hidden(1)
        for i in range(100):
            out, (state_h, state_c) = rnn(T.tensor([seed_x]).to(device), (state_h, state_c))
            last_word_logits = out[0][-1]
            p = T.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            text.append(cache.get_by_idx(word_index).key)
            seed_x.append(word_index)
            seed_x.pop(0)

        print(' '.join(text))

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    for text in read_files(args.input_dir):
        for word in strip_words(text):
            print(word)

if __name__ == '__main__':
    main()
    # test()


