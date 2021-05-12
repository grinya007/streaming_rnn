#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import read_texts_csv, strip_words

from static_vocabulary import StaticVocabulary
from dynamic_vocabulary_lru import DynamicVocabularyLRU
from dynamic_vocabulary_2q import DynamicVocabulary2Q

VOCABSIZE = 10000
FILLVOCAB = 500000
RUNTESTON = 5000000
PLOTEVERY = 100000

def save_plot(title: str, data: dict, plot_file: str):
    lines = []
    values = []
    for key, value in data.items():
        lines.append(key)
        values.append(value)

    data = pd.DataFrame(np.array(values).T, columns=lines)

    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle(title)

    x = np.arange(0, data.shape[0] * PLOTEVERY, PLOTEVERY)
    for col in data.columns.tolist():
        ax.plot(x, col, data=data)

    plt.xlabel(f'word count')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend()

    plt.savefig(plot_file)



def plot_unknown_words_ratio(input_csv):
    stat = {
        'StaticVocabulary': [],
        'DynamicVocabulary2Q': []
    }
    static = StaticVocabulary(VOCABSIZE)
    dynamic = DynamicVocabulary2Q(VOCABSIZE)

    # warm up
    i = 0
    words = []
    text_gen = read_texts_csv(input_csv, 'content')
    for text in text_gen:
        if i >= FILLVOCAB:
            break
        for word in strip_words(text):
            words.append(word)
            dynamic.word2idx(word)
            i += 1
    static.fill(words)

    i = 0
    count = 0
    static_unknowns = 0
    dynamic_unknowns = 0
    for text in text_gen:
        if i >= RUNTESTON:
            break
        for word in strip_words(text):
            static_unknowns += 1 if static.word2idx(word) == 0 else 0
            dynamic_unknowns += 1 if dynamic.word2idx(word) == 0 else 0
            if count == PLOTEVERY:
                stat['StaticVocabulary'].append(static_unknowns/count)
                stat['DynamicVocabulary2Q'].append(dynamic_unknowns/count)
                count = 0
                static_unknowns = 0
                dynamic_unknowns = 0
            count += 1
            i += 1

    save_plot('Unknown words ratio', stat, 'plot.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str)
    args = parser.parse_args()

    stat = plot_unknown_words_ratio(args.input_csv)


if __name__ == '__main__':
    main()


