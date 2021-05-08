#!/usr/bin/env python3

# import re

# # Let the text data be
# text = """The scientists of today think deeply instead of clearly.
    # One must be sane to think clearly, but one can think deeply and
    # be quite insane."""

# # The simplistic cleaning function would be
# def clean_text(raw):
    # raw = raw.lower()
    # return [ match.group(0) for match in re.finditer(r"[a-z]+", raw) ]

# text_list = clean_text(text)

# print(text_list, "\n")
# # ['the', 'scientists', 'of', 'today', 'think', 'deeply', 'instead',
# #   'of', 'clearly', 'one', 'must', 'be', 'sane', 'to', 'think',
# #   'clearly', 'but', 'one', 'can', 'think', 'deeply', 'and',
# #   'be', 'quite', 'insane']

# # there are 18 unique words in the text
# # I'll deliberately choose the VOCABULARY_SIZE that is below 18
# VOCABULARY_SIZE = 15


#   STATIC VOCABULARY

# The static vocabulary is the conventional pair
# of word2idx and idx2word dictionaries
# that requires the inspection of the text data
# in order to find the most frequent words
# the least frequent words will share the idx of
# the UNKNOWN_WORD token
class StaticVocabulary:
    def __init__(self, size):
        self.size = size
        self._word2idx = dict()
        self._idx2word = dict()

    def word2idx(self, word):
        if word in self._word2idx:
            return self._word2idx[word]
        return 0

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        return 'UNKNOWN_WORD'

    def fill(self, words):
        word_counts = dict()
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

        uniq_word_list = list(word_counts.keys())
        uniq_word_list.sort(reverse=True, key=lambda w: word_counts[w])
        # NOTE: the size of known words list is one word shorter than
        #   the total vocabulary size since the index 0 is used by
        #   the UNKNOWN_WORD token
        for idx, word in enumerate(uniq_word_list[:self.size - 1]):
            self._word2idx[word] = idx + 1
            self._idx2word[idx + 1] = word


# static_vocabulary = StaticVocabulary(VOCABULARY_SIZE)

# The vocabulary has to be filled before it can be used
# static_vocabulary.fill(text_list)

# Now, as the text flows, I can feed the word indices
# into the embedding layer of my imaginary RNN
# for word in text_list:
    # print(word, static_vocabulary.word2idx(word))
    # if I were getting predictions here
    # I'd have used static_vocabulary.idx2word() to decode them
# the 7
# scientists 8
# of 2
# today 9
# think 1
# deeply 3
# instead 10
# of 2
# clearly 4
# one 5
# must 11
# be 6
# sane 12
# to 13
# think 1
# clearly 4
# but 14
# one 5
# can 0
# think 1
# deeply 3
# and 0
# be 6
# quite 0
# insane 0
# print("\n")

# Here 4 different words share the index 0


#   DYNAMIC VOCABULARY

# The dynamic vocabulary is such that assigns indices to words
# as they flow. Therefore, it doesn't have the fill() method.
# For the sake of illustration, I'll stick to the basic LRU algorithm
from collections import OrderedDict
class DynamicVocabulary:
    def __init__(self, size, fifo_size=None):
        self.size = size
        if fifo_size is None:
            self.fifo_size = size
        else:
            self.fifo_size = fifo_size

        self._fifo = list()
        self._fifo_lookup = dict()

        self._word2idx = OrderedDict()
        self._idx2word = dict()

    def word2idx(self, word):
        idx = 0
        if word in self._word2idx:
            # if word is hit, move it to the head of the queue
            idx = self._word2idx[word]
            del self._word2idx[word]
            self._word2idx[word] = idx
        elif word in self._fifo_lookup:
            if len(self._word2idx) == self.size - 1:
                # if the queue is full, evict the least recently used
                # the idx is then freed for reuse
                _, idx = self._word2idx.popitem(0)
                self._idx2word[idx] = word
            else:
                idx = len(self._word2idx) + 1
                self._idx2word[idx] = word

            self._word2idx[word] = idx

        self._fifo.append(word)
        if word in self._fifo_lookup:
            self._fifo_lookup[word] += 1
        else:
            self._fifo_lookup[word] = 1
        if len(self._fifo) > self.fifo_size:
            old_word = self._fifo.pop(0)
            self._fifo_lookup[old_word] -= 1
            if self._fifo_lookup[old_word] == 0:
                del self._fifo_lookup[old_word]

        return idx

    def idx2word(self, idx):
        if idx in self._idx2word:
            return self._idx2word[idx]
        return 'UNKNOWN_WORD'


# dynamic_vocabulary = DynamicVocabulary(VOCABULARY_SIZE)

# for word in text_list:
    # print(word, dynamic_vocabulary.word2idx(word))
    # if I were getting predictions here
    # I'd have used dynamic_vocabulary.idx2word() to decode them
# the 0
# scientists 1
# of 2
# today 3
# think 4
# deeply 5
# instead 6
# of 2
# clearly 7
# one 8
# must 9
# be 10
# sane 11
# to 12
# think 4
# clearly 7
# but 13
# one 8
# can 14
# think 4
# deeply 5
# and 0
# be 10
# quite 1
# insane 3
# print("\n")

# Here, indices 0, 1, and 3 were reused by words that appear closer to the end of
# the original text, the index 2 has been preserved for 'of' as it appeared
# repeatedly
