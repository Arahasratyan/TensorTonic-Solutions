import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        word_lst = []
        for text in texts:
            word_lst.extend(text.lower().split())
            
        word_lst = sorted(list(set(word_lst)))
        special_word_lst = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.vocab_size += len(word_lst) + 4
        self.id_to_word.update(dict(enumerate(special_word_lst)))
        self.id_to_word.update(dict(enumerate(word_lst, start=4)))
        self.word_to_id = {v:k for k, v in self.id_to_word.items()}
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        text_lst = text.lower().split()
        ids = []
        for word in text_lst:
            if word not in self.word_to_id.keys():
                ids.append(self.word_to_id[self.unk_token])
            else:
                ids.append(self.word_to_id[word])

        return ids

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for id in ids:
            if id in self.id_to_word:
                words.append(self.id_to_word[id])
            else:
                words.append(self.unk_token)
        return " ".join(words)
