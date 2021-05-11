from collections import OrderedDict

class DynamicVocabularyLRU:
    """
    The dynamic vocabulary class

    This vocabulary doesn't have the notion of an unknown word.
    Any word that is passed to word2idx() will obtain its index.
    Words are being replaced in the LRU order.
    """
    def __init__(self, size: int):
        self.size = size
        self._word2idx = OrderedDict()
        self._idx2word = dict()

    def word2idx(self, word: str) -> int:
        """
        Resolves a word to its index
        """
        idx = None
        if word in self._word2idx:
            # if the word already exists in the vocabulary
            # remember its index and delete it from whatever
            # position it's at in the _word2idx queue
            # it then will be inserted into the head of the queue
            # and assigned the same idx
            idx = self._word2idx[word]
            del self._word2idx[word]
        elif len(self._word2idx) == self.size:
            # if the queue is full, evict the least recently used
            # item, the idx is then freed for reuse
            _, idx = self._word2idx.popitem(0)
            self._idx2word[idx] = word
        else:
            # if there's still space within self.size
            # assign a new idx to a word
            # NOTE: in order to make this vocabulary interchangeable with
            #   the other two, index 0 is never used, because it's
            #   meant for unknown words but every word is "known"
            #   in this vocabulary
            idx = len(self._word2idx) + 1
            self._idx2word[idx] = word

        # insert the word into the head of the queue
        self._word2idx[word] = idx

        return idx

    def idx2word(self, idx: int) -> str:
        """
        Resolves an index to a word
        """
        if idx in self._idx2word:
            return self._idx2word[idx]
        # While there is no unknown words in the word2idx()
        # idx2word() still has to somehow handle unknown indices
        # let's return UNK for compatibility sake
        return 'UNK'

