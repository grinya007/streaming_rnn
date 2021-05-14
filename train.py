#!/usr/bin/env python3

import argparse
import numpy as np
from data import read_texts_csv, strip_words
from plot import save_plot
from model import torch_device, make_data_loader, make_rnn, train_iteration, RNN, TrainState

from dynamic_vocabulary_2q import DynamicVocabulary2Q
from dynamic_vocabulary_lru import DynamicVocabularyLRU
from static_vocabulary import StaticVocabulary

LOOKBACK = 10
BATCHSIZE = 100
REPORTEVERY = 1000
VOCABSIZE = 10000
FILLVOCAB = 500000
TRAINLIMIT = 3000000
SKIPARTICLES = 50000
DEVICE = 'cuda'

def word_index_generator(text, vocab):
    for word in strip_words(text):
        yield vocab.word2idx(word)

def train(text_gen, vocab, device):
    rnn, criterion, optimizer = make_rnn(VOCABSIZE, 256, 256, 2, 0.2)
    rnn.to(device)

    state = TrainState(BATCHSIZE, REPORTEVERY, print_log=True)
    for text in text_gen:
        if state.batch_counter*BATCHSIZE > TRAINLIMIT:
            break
        loader = make_data_loader(word_index_generator(text, vocab), BATCHSIZE, LOOKBACK)
        state = train_iteration(loader, rnn, criterion, optimizer, state, device)

    return state

def train_static(input_csv, device):
    static = StaticVocabulary(VOCABSIZE)

    # warm up
    i = 0
    words = []
    texts_count = 0
    text_gen = read_texts_csv(input_csv, 'content')
    for text in text_gen:
        texts_count += 1
        if texts_count < SKIPARTICLES:
            continue
        if i > FILLVOCAB:
            break
        for word in strip_words(text):
            words.append(word)
            i += 1
    static.fill(words)

    return train(text_gen, static, device)

def train_dynamic(input_csv, device, dynamic_class):
    dynamic = dynamic_class(VOCABSIZE)

    # warm up
    i = 0
    texts_count = 0
    text_gen = read_texts_csv(input_csv, 'content')
    for text in text_gen:
        texts_count += 1
        if texts_count < SKIPARTICLES:
            continue
        if i > FILLVOCAB:
            break
        for word in strip_words(text):
            dynamic.word2idx(word)
            i += 1

    return train(text_gen, dynamic, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str)
    args = parser.parse_args()

    device = torch_device(DEVICE)

    print("\tSTATIC\n", flush=True)
    static_state = train_static(args.input_csv, device)

    print("\n\tDYNAMIC 2Q\n", flush=True)
    dynamic_2q_state = train_dynamic(args.input_csv, device, DynamicVocabulary2Q)

    print("\n\tDYNAMIC LRU\n", flush=True)
    dynamic_lru_state = train_dynamic(args.input_csv, device, DynamicVocabularyLRU)

    save_plot('Unknown words prediction ratio', 'word count', '%', {
        'StaticVocabulary': static_state.unknowns_ratio,
        'DynamicVocabulary2Q': dynamic_2q_state.unknowns_ratio
    }, REPORTEVERY*BATCHSIZE, 'uwpr.png')

    save_plot('Model perplexity', 'word count', 'ppl', {
        'StaticVocabulary': static_state.perplexity,
        'DynamicVocabulary2Q': dynamic_2q_state.perplexity,
        'DynamicVocabularyLRU': dynamic_lru_state.perplexity,
    }, REPORTEVERY*BATCHSIZE, 'ppl.png')


if __name__ == '__main__':
    main()


