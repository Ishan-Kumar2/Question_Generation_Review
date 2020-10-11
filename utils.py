import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import string
import config
import torch
import torch.nn as nn
import torch.utils.data as data
import nltk
import json
import argparse
from collections import Counter


def init_weights(m):
    for name, param in m.named_parameters():
        #Normal Init       # TODO: Xavier
        nn.init.normal_(param.data, mean = 0, std = 0.01)

spacy_eng=spacy.load('en')
nlp=spacy.load("en_core_web_sm")
def tokenize_eng(text):
    sent=[]
    lema=nlp(text)
    for tok in spacy_eng.tokenizer(text):
        word=tok.lemma_
        #word=tok.text
        if word not in string.punctuation:
            sent.append(word)

    return sent

def load_dataset_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def build_word2id(sequences, min_word_count):
    """Creates word2id dictionary.
    """
    num_seqs = len(sequences)
    counter = Counter()

    for i, sequence in enumerate(sequences):
        tokens = nltk.tokenize.word_tokenize(sequence.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<pad>'] = 0
    word2id['<start>'] = 1
    word2id['<end>'] = 2
    word2id['<unk>'] = 3

    # if word frequency is less than 'min_word_count', then the word is discarded
    words = [word for word, count in counter.items() if count >= min_word_count]

    # add the words to the word2id dictionary
    for i, word in enumerate(words):
        word2id[word] = i + 4

    id2word = {value: key for key, value in word2id.items()}
    return word2id, id2word

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, path):
        """Reads source and target sequences from txt files."""
        self.data=pd.read_csv(path)
        inp=[]
        outp=[]
        print(self.data.iloc[0])
        for i in range(len(self.data)):
            inp.append(self.data.iloc[i]['text'])
            inp.append(self.data.iloc[i]['context'])
            outp.append(self.data.iloc[i]['question'])

        #I have decideed to make 1 combined vocab for input, can have seperate ones too
        self.word2ind_input,self.id2word_input=build_word2id(inp, config.min_word_count_inp)
        self.word2ind_output,self.id2word_output=build_word2id(outp, config.min_word_count_out)

        with open(config.source_dict_dump_path,'w') as f:
            json.dump(self.word2ind_input,f)
        with open(config.target_dict_dump_path,'w') as f:
            json.dump(self.word2ind_output,f)

        print("Input Vocab Size",len(self.word2ind_input))
        print("Output Vocab Size",len(self.word2ind_output))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src = self.data.iloc[index]
        ans=src['text']
        context=src['context']
        question=src['question']

        src_1 = self.preprocess(ans, self.word2ind_input)
        src_2 = self.preprocess(context, self.word2ind_input)
        trg=self.preprocess(question, self.word2ind_output)

        return src_1,src_2,trg

    def __len__(self):
        return len(self.data)

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        tokens = nltk.tokenize.word_tokenize(sequence.lower())
        sequence = []
        sequence.append(word2id['<start>'])
        sequence.extend([word2id[token] for token in tokens if token in word2id])
        sequence.append(word2id['<end>'])
        sequence = torch.Tensor(sequence)
        return sequence

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Parameters:
    ---------------------------
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
    ---------------------------
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sorted by context length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # seperate source and target sequences
    src_seqs,src_seqs_2, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    src_seqs_2, src_lengths_2 = merge(src_seqs_2)
    trg_seqs, trg_lengths = merge(trg_seqs)
    #return src_seqs, src_lengths, trg_seqs, trg_lengths
    #print()
    #print("LENGTHS",src_seqs.shape,src_2.shape,trg_seqs.shape)
    return src_seqs,src_seqs_2,trg_seqs

def get_loader(path, batch_size=32):
    """Returns data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(path)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)             ## TODO: BucketIterato

    return data_loader
