#!/usr/bin/env python3

import argparse
import numpy as np
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
        yield text_file.read_text()

def strip_words(text):
    for match in re.finditer(r"[-'a-zA-Z0-9]+|[\.\,\?\!\:\;\(\)]", text):
        words = match.group(0).lower()
        if words == 'br':
            continue
        for word in re.split(r'-{2,}', words):
            yield word

LOOKBACK = 10
CACHESIZE = 2000
cache = Cache(CACHESIZE)
def train_stream(input_dir):
    X = []
    for text in read_files(input_dir):
        for word in strip_words(text):
            result = cache.get_replace(word)
            if len(X) <= LOOKBACK:
                X.append(result.idx)
            else:
                x = X.copy()
                yield x[:-1], x[1:]
                X.pop(0)
                X.append(result.idx)

device = T.device("cuda")
class DS(T.utils.data.IterableDataset):
    def __init__(self, gen):
        super(DS).__init__()
        self.gen = gen
    def __iter__(self):
        for x, y in self.gen:
            yield T.tensor(x).to(device), T.tensor(y).to(device)

class GRU(T.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = T.nn.Embedding(num_embeddings, embedding_dim)
        self.gru = T.nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = T.nn.Linear(hidden_dim, num_embeddings)

    def forward(self, x, h):
        embeds = self.embed(x)
        out, h = self.gru(embeds, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, l):
        hidden = T.zeros(self.n_layers, l, self.hidden_dim).to(device)
        return hidden

BATCHSIZE = 100
def train(rnn, loader):
    rnn.to(device)
    rnn.train()
    criterion = T.nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(rnn.parameters(), lr=0.001)

    ti = time()
    counter = 0
    avg_loss = 0.
    h = rnn.init_hidden(BATCHSIZE)
    for x, y in loader:
        counter += 1
        h = h.data
        optimizer.zero_grad()
        out, h = rnn(x, h)
        loss = criterion(out.transpose(1, 2), y)
        h = h.detach()

        loss.backward(retain_graph=True)
        T.nn.utils.clip_grad_norm_(rnn.parameters(), 5)
        optimizer.step()
        avg_loss += loss.item()
        if counter % 1000 == 0:
            print("batch: {} (last 1000 batches in {:.2f} s) Average Loss: {}".format(counter, time() - ti, avg_loss/counter))
            ti = time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    args = parser.parse_args()

    ds = DS(train_stream(args.input_dir))
    loader = T.utils.data.DataLoader(ds, batch_size=BATCHSIZE, drop_last=True)
    rnn = GRU(CACHESIZE, 128, 256, 3)
    try:
        train(rnn, loader)
    except KeyboardInterrupt:
        rnn.eval()
        while True:
            seed = input('5 words: ')
            seed_x = []
            for word in strip_words(seed):
                result = cache.get_replace(word)
                seed_x.append(result.idx)
            if len(seed_x) == 0:
                continue

            text = []
            h = rnn.init_hidden(len(seed_x))
            for i in range(100):
                h = h.data
                out, h = rnn(T.tensor([seed_x]).T.to(device), h)
                last_word_logits = out[0][-1]
                p = T.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
                word_index = np.random.choice(len(last_word_logits), p=p)
                text.append(cache.get_by_idx(word_index).key)
                seed_x.append(word_index)
                seed_x.pop(0)

            print(' '.join(text))



if __name__ == '__main__':
    main()


