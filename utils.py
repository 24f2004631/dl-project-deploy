from typing import List
import re
from config import ENGLISH_STOP_WORDS

# Define corpus cleaner
def clean_text(corpus: str) -> List[str]:
    # Turn lowercase & remove punctuation
    clean_corpus = re.findall(r"[a-z0-9]+", corpus.lower())
    # Remove stop words
    clean_corpus = [word for word in clean_corpus if word not in ENGLISH_STOP_WORDS]
    # Return
    return clean_corpus

# Encoder
def encode(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens[:max_len]]
    length = len(ids)
    ids = ids + [vocab["<pad>"]] * (max_len - length)
    return ids, max(length, 1)
