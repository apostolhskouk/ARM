from typing import List
import string
import re

def simple_tokenize(text: str) -> List[str]:
    #remove all occurances of [SEP] and [H] tags
    text = re.sub(r'\[SEP\]', '', text)
    text = re.sub(r'\[H\]', '', text)
    #replace all punct with space
    text = ''.join(' ' if c in string.punctuation else c for c in text)
    #remoce extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    #split by spaces
    tokens = text.split(' ')
    #remove empty tokens
    tokens = [token for token in tokens if token]
    #lowercase all tokens
    tokens = [token.lower() for token in tokens]
    return tokens

def generate_ngrams(tokens: List[str], min_n: int, max_n: int):
    """Generates n-grams from a list of tokens."""
    num_tokens = len(tokens)
    for n in range(min_n, max_n + 1):
        for i in range(num_tokens - n + 1):
            yield tuple(tokens[i:i+n])