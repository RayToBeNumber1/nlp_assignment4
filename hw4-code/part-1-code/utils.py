import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################


    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    
    keyboard_neighbors = {
        'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
        'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
        'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
        'p': ['o', 'l'], 'a': ['q', 's', 'z'], 's': ['w', 'a', 'd', 'z', 'x'],
        'd': ['e', 's', 'f', 'x', 'c'], 'f': ['r', 'd', 'g', 'c', 'v'],
        'g': ['t', 'f', 'h', 'v', 'b'], 'h': ['y', 'g', 'j', 'b', 'n'],
        'j': ['u', 'h', 'k', 'n', 'm'], 'k': ['i', 'j', 'l', 'm'],
        'l': ['o', 'k', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
        'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'],
        'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k']
    }
    
    text = example["text"]
    

    words = word_tokenize(text)
    transformed_words = []
    
    for word in words:

        if len(word) <= 2 or not word.isalpha():
            transformed_words.append(word)
            continue
        

        if random.random() < 0.22:
            synsets = wordnet.synsets(word)
            if synsets:
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != word.lower() and '_' not in synonym:
                            synonyms.append(synonym)
                
                if synonyms:
                    word = random.choice(synonyms)
        

        if random.random() < 0.18 and len(word) > 3:
            word_list = list(word)

            char_idx = random.randint(0, len(word) - 1)
            original_char = word[char_idx].lower()
            
            if original_char in keyboard_neighbors:
            
                new_char = random.choice(keyboard_neighbors[original_char])
           
                if word[char_idx].isupper():
                    new_char = new_char.upper()
                word_list[char_idx] = new_char
                word = ''.join(word_list)
        
        transformed_words.append(word)
    

    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)


    return example
