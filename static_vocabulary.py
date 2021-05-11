class StaticVocabulary:
    """
    The static vocabulary class

    This is a conventional pair of word2idx and idx2word dictionaries
    with the handling of unknown words
    """
    def __init__(self, size: int):
        self.size = size
        self._word2idx = dict()
        self._idx2word = dict()

    def word2idx(self, word: str) -> int:
        """
        Resolves a word to its index
        """
        if word in self._word2idx:
            return self._word2idx[word]
        # Any word that doesn't exist in the vocabulary
        # obtains index 0
        return 0

    def idx2word(self, idx: int) -> str:
        """
        Resolves an index to a word
        """
        if idx in self._idx2word:
            return self._idx2word[idx]
        # Any index that doesn't exist in the vocabulary
        # is resolved into UNK token
        return 'UNK'

    def fill(self, words: list):
        """
        Counts word frequencies and fills the vocabulary
        with self.size-1 of the most frequent words in
        an arbitrary piece of text presented as a list of words.
        """
        uniq_word_counts = dict()
        for word in words:
            if word not in uniq_word_counts:
                uniq_word_counts[word] = 0
            uniq_word_counts[word] += 1

        uniq_word_list = list(uniq_word_counts.keys())
        uniq_word_list.sort(reverse=True, key=lambda w: uniq_word_counts[w])

        # NOTE: the size of known words list is one word shorter than
        #   the total vocabulary size because the index 0 is used by
        #   the UNK token
        for idx, word in enumerate(uniq_word_list[:self.size - 1]):

            # this is meant to exclude index 0 from the vocabulary
            idx += 1

            self._word2idx[word] = idx
            self._idx2word[idx] = word

