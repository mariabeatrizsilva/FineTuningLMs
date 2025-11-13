import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        tokenizer_checkpoint = 'google-t5/t5-small'
        self.split = split
        self.data_folder = data_folder
        tokenizer = T5TokenizerFast.from_pretrained(tokenizer_checkpoint)
        self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_queries = load_lines(os.path.join(data_folder, f'{split}.nl'))
        if split != 'test':
            sql_queries = load_lines(os.path.join(data_folder, f'{split}.sql'))
        else:
            sql_queries = None
        
        # Store tokenized data
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []

        for i, nl_query in enumerate(nl_queries):
            # tokenize nl query and save to encoder 
            enc_tokens = tokenizer(nl_query, return_tensors='pt') # pt -> pytorch output
            self.encoder_inputs.append(enc_tokens['input_ids'].squeeze(0))
            
            # tokenize sql query --> add to decoder 
            if sql_queries is not None:
                sql_query = sql_queries[i]
                dec_input = '<extra_id_0>' + sql_query # extra_id_0 is built in special token in T5's vocab! allows us to shift decoder stuff for targets
                dec_tokens = tokenizer(dec_input, return_tensors='pt')
                dec_ids = dec_tokens['input_ids'].squeeze(0)

                # targets are shifted by 1
                self.decoder_inputs.append(dec_ids[:-1])  # decoder inputs are 0-n-1
                self.decoder_targets.append(dec_ids[1:])  # targets are 1 to n

    
    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == 'test':
            return self.encoder_inputs[idx]
        else:
            return (
                self.encoder_inputs[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx]
            )

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    return [], [], [], [], []

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    return [], [], []

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x