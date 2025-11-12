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

# this is used to split transformations (half get typo half get synonym)
transform_counter = 0

""" Setup for letter replacement  """
## I wrote the first 5 letters and adjacent chars on my own, then asked gemini to do the rest 
## (I verified if it matched up with my computer keyboard and added what it got wrong)
close_letters = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['e', 'r', 'f', 'c', 'x', 's'],
    'e': ['4', '3', 'w', 's', 'd', 'r'],
    'f': ['r', 't', 'g', 'v', 'c', 'd'],
    'g': ['t', 'y', 'h', 'b', 'v', 'f'],
    'h': ['y', 'u', 'j', 'n', 'b', 'g'],
    'i': ['8', '9', 'o', 'l', 'k', 'u'],
    'j': ['u', 'i', 'k', 'm', 'n', 'h'],
    'k': ['i', 'o', 'l', 'm', 'j'],
    'l': ['o', 'p',  'k'],
    'm': ['n', 'j', 'k', 'l'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['9', '0', 'p', 'k', 'l', 'i'],
    'p': ['0', 'l', 'o'],
    'q': ['1', '2', 'w', 'a'],
    'r': ['4', '5', 't', 'f', 'd', 'e'],
    's': ['w', 'e', 'd', 'x', 'z', 'a'],
    't': ['5', '6', 'y', 'f', 'g', 'r'],
    'u': ['7', '8', 'i', 'h', 'j', 'y'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['2', '3', 'e', 'q', 's', 'a'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['6', '7', 't', 'g', 'h', 'u'],
    'z': ['a', 's', 'x'],
}

def choose_typo(char: str) -> str:
    char_lower = char.lower()
    
    # if character is not 
    if char_lower not in close_letters:
        return char
    typo = random.choice(close_letters[char_lower])
    # preserve original capitalization
    if char.isupper():
        return typo.upper()
    return typo
    
def add_typos(example: str, typo_rate: float = 0.03) -> str:
    if len(example) == 0:
        return example
    
    result = []
    for char in example:
        # don't change non-letters
        if not char.isalpha():
            result.append(char)
            continue
        
        if random.random() < typo_rate:
            result.append(choose_typo(char))
        else:
            result.append(char)
    
    return ''.join(result)

def get_synonym(word: str) -> str:

    if len(word) <= 2:
        return word

    synsets = wordnet.synsets(word.lower())
    
    if not synsets:
        return word
    
    synset = synsets[0]

    synonyms = []
    for lemma in synset.lemmas():
        synonym = lemma.name().replace('_', ' ')
        # don't replace word with itself
        if synonym.lower() != word.lower():
            synonyms.append(synonym)
    
    # pick random synonym if found in the sysnet
    if synonyms:
        return random.choice(synonyms)
    return word

def replace_with_synonyms(text: str, replacement_rate: float = 0.15) -> str:
    words = text.split()
    num_replacements = max(1, int(len(words) * replacement_rate))
    
    if len(words) == 0:
        return text
        
    replacement_positions = random.sample(range(len(words)), min(num_replacements, len(words)))
    
    for pos in replacement_positions:
        word = words[pos]
        synonym = get_synonym(word)
        
        if word and word[0].isupper():
            synonym = synonym.capitalize()
            
        words[pos] = synonym
    
    return ' '.join(words)


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
    ##### YOUR CODE BEGINGS HERE ###
    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    global transform_counter
    
    if transform_counter % 2 == 0:
        # Apply keyboard-based typo transformation
        example["text"] = add_typos(example["text"], typo_rate=0.03)
    else:
        # Apply synonym replacement
        example["text"] = replace_with_synonyms(example["text"], replacement_rate=0.15)
    
    transform_counter += 1
    ##### YOUR CODE ENDS HERE ######

    return example
