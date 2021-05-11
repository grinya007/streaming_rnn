from collections import OrderedDict
from typing import Optional

class DynamicVocabulary2Q:
    """
    The dynamic vocabulary class

    This vocabulary employes two queues: FIFO and LRU

    - FIFO queue serves the purpose of protection against
      infrequent words getting into the vocabulary
    - LRU maintains the set of frequent words

    Only such word that occurs more than once within a text of a
    certain length (fifo_size) gets to the LRU queue and becomes
    a member of the vocabulary
    """
    def __init__(self, size: int, fifo_size: Optional[int] = None):
        self.size = size
        if fifo_size is None:
            self.fifo_size = size
        else:
            self.fifo_size = fifo_size

        self._fifo = list()

        # _fifo_lookup makes it possible to quickly check
        # whether a word is in FIFO
        self._fifo_lookup = dict()

        self._word2idx = OrderedDict()
        self._idx2word = dict()

    def word2idx(self, word: str) -> int:
        """
        Resolves a word to its index
        Returns 0 when the word isn't present in the _word2idx
        or in the _fifo_lookup
        """
        idx = 0
        if word in self._word2idx:
            # if the word already exists in the vocabulary
            # remember its index and delete it from whatever
            # position it's at in the _word2idx queue
            # and insert it into the head of the queue
            # assigning the same idx
            idx = self._word2idx[word]
            del self._word2idx[word]

            self._word2idx[word] = idx

        elif word in self._fifo_lookup:
            # if the word is in the _fifo_lookup
            # it deserves becoming a member of the vocabulary
            # i.e. getting into the _word2idx
            if len(self._word2idx) == self.size - 1:
                # if the queue is full, evict the least recently used
                # item, the idx is then freed for reuse
                _, idx = self._word2idx.popitem(0)
                self._idx2word[idx] = word
            else:
                # if there's still space within self.size - 1
                # assign a new idx to a word
                # NOTE: index 0 is preserved for unknown words
                idx = len(self._word2idx) + 1
                self._idx2word[idx] = word

            self._word2idx[word] = idx

        # maintain the FIFO queue and its lookup dict
        self._fifo.append(word)
        if word in self._fifo_lookup:
            # the value in the lookup dict is the counter
            # of how many entries a word has in the FIFO queue
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
        """
        Resolves an index to a word
        """
        if idx in self._idx2word:
            return self._idx2word[idx]
        return 'UNK'
