import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#One t
import spacy
spacy_eng=spacy.load('en')
nlp=spacy.load("en_core_web_sm")
import string
def tokenize_eng(text):
    sent=[]
    lema=nlp(text)
    for tok in spacy_eng.tokenizer(text):

        word=tok.lemma_
        #word=tok.text

        if word not in string.punctuation:
            sent.append(word)

    return sent

from torchtext.data import Field, ReversibleField

TEXT=Field(sequential=True,tokenize=tokenize_eng,
           lower=True,
           init_token='<START>',
          eos_token='<END>',
          batch_first=True)

LABEL=Field(sequential=True,
            tokenize=tokenize_eng,
            lower=True,
            init_token='<START>',
            eos_token='<END>',
            use_vocab=True,
           batch_first=True)

from torchtext.data import TabularDataset

fields=[("Unnamed: 0",None),
        ("Unnamed: 0.1",None),
        ('index',None),
        ('question',LABEL),

        ('context',TEXT),
        ('answer_start',None),
        ('text',TEXT),
        ('c_id',None)]

train=TabularDataset(path='./data/train_small.csv',
                    format='csv',
                   skip_header=True,
                    fields=fields)


TEXT.build_vocab(train,min_freq=1)

LABEL.build_vocab(train,min_freq=1)#vectors='glove.6B.50d')

print(f"TEXT Vocab len {len(TEXT.vocab)}")
print(f"LABEL Vocab len {len(LABEL.vocab)}")
print(TEXT.vocab.stoi)
print(TEXT.vocab.stoi['hell'])
print(TEXT.vocab.stoi['the'])
from torchtext.data import Iterator,BucketIterator

train_iterator=BucketIterator(train,
                             batch_size=32,
                             sort_key=lambda x: len(x.context),
                             sort_within_batch=True,
                             repeat=False)

class BatchWrapper:
    def __init__(self,dl,x_vars1,x_vars2,y_vars):
        self.dl=dl
        self.x_vars=x_vars1
        self.y_vars=y_vars
        self.x_vars2=x_vars2
    def __iter__(self):

        for batch in self.dl:
            x1=getattr(batch,self.x_vars)
            x2=getattr(batch,self.x_vars2)
            y=getattr(batch,self.y_vars)

            #x=torch.cat((x1,x2),dim=1)

            yield (x1,x2,y)

    def __len__(self):
        return len(self.dl)
