#!/usr/bin/env python3

import argparse

from data import read_texts_csv, strip_words
from plot import save_plot

from static_vocabulary import StaticVocabulary
from dynamic_vocabulary_2q import DynamicVocabulary2Q

VOCABSIZE = 110000
FILLVOCAB = 10000000
RUNTESTON = 100000000
PLOTEVERY = 2000000

def plot_unknown_words_ratio(input_csv):
    static = StaticVocabulary(VOCABSIZE)
    dynamic = DynamicVocabulary2Q(VOCABSIZE)

    # feeding vocabularies
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

    # 2016-02-19

    # test
    i = 0
    count = 0
    static_unknowns = 0
    dynamic_unknowns = 0
    stat = {
        'StaticVocabulary': [],
        'DynamicVocabulary2Q': []
    }
    for text in text_gen:
        if i >= RUNTESTON:
            break
        for word in strip_words(text):
            # NOTE index 0 means that a word is unknown
            static_unknowns += 1 if static.word2idx(word) == 0 else 0
            dynamic_unknowns += 1 if dynamic.word2idx(word) == 0 else 0
            if count == PLOTEVERY:
                stat['StaticVocabulary'].append(100*static_unknowns/count)
                stat['DynamicVocabulary2Q'].append(100*dynamic_unknowns/count)
                static_unknowns = 0
                dynamic_unknowns = 0
                count = 0
            count += 1
            i += 1

    save_plot('Unknown word ratio', 'word count', '%', stat, PLOTEVERY, 'uwr.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_csv',
        type=str,
        help="Path to the input csv file",
    )
    args = parser.parse_args()

    plot_unknown_words_ratio(args.input_csv)


if __name__ == '__main__':
    main()


