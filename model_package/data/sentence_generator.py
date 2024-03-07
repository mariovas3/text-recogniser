"""
Use the Brown corpus to generate sentences.

* The Brown Corpus was the first million-word 
electronic corpus of English, created in 1961 at 
Brown University. This corpus contains text from 500 sources, 
and the sources have been categorized by genre, 
such as news, editorial, and so on. For a complete 
list, see http://icame.uib.no/brown/bcm-los.html.
"""

import itertools
import random
import re
import string

import nltk

from model_package.metadata.shared import DATA_DIR


class SentenceGenerator:
    """Generate sentences using Brown corpus."""

    def __init__(self, max_length):
        # get single string from Brown corpus with no punctuation;
        # add a last token so we can consider the true last token;
        self.text = brown_without_punctuation() + " <THE_END>"
        # the re.finditer returns an iterator of re.Match objects
        # with start idx and end idx of pattern to be matched;
        # we take the first idx of the match and increment by 1
        # to go to the idx after the idx of a whitespace.
        self.start_of_words_idxs = [0] + [
            _.start(0) + 1 for _ in re.finditer(" ", self.text)
        ]
        assert self.text[self.start_of_words_idxs[-1] :] == "<THE_END>"
        # max length in characters;
        self.max_length = max_length

    def generate(self, max_length=None):
        """
        Get a string of at most max_length characters.

        Non-white-space elements guaranteed to form words
        as per the Brown corpus.

        The number of words in the string are sampled uniformly
        so it's always better to have bigger max_lenght than lower.
        """
        if max_length is None:
            max_length = self.max_length

        # ignore the <THE_END> token's start location.
        first_word = random.randint(0, len(self.start_of_words_idxs) - 2)
        start_idx = self.start_of_words_idxs[first_word]

        last_idx_pool = self._sample_last_idx_pool(
            first_word, start_idx, max_length
        )
        if len(last_idx_pool) == 0:
            raise RuntimeError(
                f"Couldn't generate a string; max_length: {max_length} is too small."
            )
        chosen_end = random.choice(last_idx_pool)
        return self.text[start_idx:chosen_end].strip()

    def _sample_last_idx_pool(self, first_word, start_idx, max_length):
        """
        Get list of idxs of beggining of words after the first word.

        Could further optimise this func by Binary Search to find idx
        that leads to longest sentence shorter than max_length and
        then take the slice of self.start_of_words_idxs[first_word+1:found_idx].
        Could be implemented with bisect I recon. But that's only good
        for large max_length, in practice max_length should be reached
        quickly after just a few words.
        """
        last_idx_pool = []
        for temp in range(first_word + 1, len(self.start_of_words_idxs)):
            curr_idx = self.start_of_words_idxs[temp]
            # curr_idx is exclusive idx in the slice first_word:curr_idx
            # also it is guaranteed that there is a whitespace in front
            # of curr_idx that will be stripped so we do +1;
            if curr_idx - start_idx > max_length + 1:
                break
            last_idx_pool.append(curr_idx)
        return last_idx_pool


def brown_without_punctuation():
    """Flatten brown corpus to single string ignoring punctuation."""
    # get a list of sentences;
    # each sentence is a list of words;
    # this should be some nltk streamer object
    # that is memory efficient and has at most one file open
    # to read from;
    sents = load_nltk_brown_corpus()
    # collapse to single string;
    text = " ".join(itertools.chain.from_iterable(sents))
    # replace punctuation with white space;
    text = text.translate({ord(c): None for c in string.punctuation})
    # replace consecutive spaces with single space;
    return re.sub(" +", " ", text)


def load_nltk_brown_corpus():
    """Load Brown corpus using nltk."""
    nltk.data.path.append(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.download("brown", download_dir=DATA_DIR)
    # return list of sentences each being a list of words;
    # from all topics in brown;
    return nltk.corpus.brown.sents()
