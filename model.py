import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

class Encoder_Model(nn.Module):
    """
    Encoder class for the seq2seq Model
    Parameters
    ---------
    n_vocab : ``int``, Required
        Size of input vocabulary.
    pretrained_vecs : ``tensor [vocab size, embedding_dim]``, Optional
        Pretrained Embedding Weights
        Default- random initialisation
    arch: ``int`` Optional
        To choose arch between
        RNN - 1, LSTM - 2, GRU - 3
        Default 1
    embedding_dim : ``int``, Optional
        Embedding dim of Input( Ensure its equal to the embedding dim given as pretrained vecs).
        Default- 100
    hidden_dim : ``int``, Optional
        Hidden dim size of Encoder
        Default- 64
    num_layers: ``int``,Optional.
        Number of layers required in encoder
        Default- 1
    bidirectional: ``bool``,Optional.
        Flag denoting whether the encoder is bidirectional
        default- False
    dropout: ``int``, Optional.
        Amount of dropout used between the RNN and final FC layers
        default- 0.3
    """
    def __init__(self,n_vocab,
                pretrained_vecs,
                arch=1,
                embedding_dim=100,
                hidden_dim=64,
                num_layers=4,
                bidirectional=True,
                dropout=0.3):

        super(Encoder_Model,self).__init__()
        self.n_vocab=n_vocab
        #self.batch_size=batch_size
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.arch=arch
        self.dropout=dropout
        self.pretrained_vecs=pretrained_vecs

        self.embedding=nn.Embedding(self.n_vocab,self.embedding_dim)
        self.embedding.weight.data.copy_(self.pretrained_vecs)
        self.embedding.weight.requires_grad=False

        #Encoder
        if self.arch==1:
            self.rnn=nn.RNN(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True)
        if self.arch==2:
            self.rnn=nn.LSTM(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True)
        if self.arch==3:
            self.rnn=nn.GRU(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True)

        #self.dropout=nn.Dropout(dropout)

    def forward(self,src):
        #src [batch_size,seq_len]
        embedded=self.embedding(src)
        #embedded [batch_size, seq_len, embedding_dim]
        if self.arch==1:
            outputs,hidden=self.rnn(embedded)
        else:
            outputs,(hidden,cell)=self.rnn(embedded)
        #output [batch_size, seq_len, hidden_dim*dir]  its the top hidden layer drop in this case
        #hidden [num_layers*dir, batch_size, hidden_dim]
        #Cell state if present will be same dim as hidden
        if self.bidirectional:
            hidden=torch.cat((hidden[-1,:,:],hidden[-2,:,:]),dim=1)

        #hidden [batch_size, hidden_dim*2] if bidirectional
        return hidden

class Decoder(nn.Module):
    """
    Decoder class for the seq2seq Model
    Parameters
    ---------
    self,output_dim,emb_dim,hid_dim,dropout,pretrained_vecs
    output_dim : ``int``, Required
        Size of output vocabulary.
    embedding_dim : ``int``, Optional
        Embedding dim of Output( Ensure its equal to the embedding dim given as pretrained vecs).
        Default- 100
    num_layers : ``int``, Optional
        Depth of Decoder
        Default- 1
    arch: ``int`` Optional
        To choose arch between
        RNN - 1, LSTM - 2, GRU - 3
        Default 1
    hidden_dim : ``int``, Optional
        Hidden dim size of Encoder
        Default- 128 (2* encoder_hidden_size)
    dropout: ``int``, Optional.
        Amount of dropout used between the RNN and final FC layers
        default- 0.3
    pretrained_vecs : ``tensor [vocab size, embedding_dim]``, Optional
        Pretrained Embedding Weights
        Default- random initialisation
    """
    def __init__(self,
                output_dim,
                emb_dim=100,
                num_layers=1,
                arch=1,
                hid_dim=128,
                dropout=0.3,
                pretrained_vecs=None):

        super(Decoder,self).__init__()
        self.hidden_dim=hid_dim
        self.embedding_dim=emb_dim
        self.dropout=dropout
        self.output_dim=output_dim
        self.pretrained_vecs=pretrained_vecs
        self.num_layers=num_layers
        self.arch=arch
        self.embedding=nn.Embedding(output_dim,emb_dim) #Output dim is number of words in vocab
        if self.pretrained_vecs is not None:
            self.embedding.weight.data.copy_(self.pretrained_vecs)
            self.embedding.weight.requires_grad=False

        if self.arch==1:
            self.rnn=nn.RNN(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True)
        if self.arch==2:
            self.rnn=nn.LSTM(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True)
        if self.arch==3:
            self.rnn=nn.GRU(self.embedding_dim,self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True)



        self.fc_out=nn.Linear(hid_dim+emb_dim,output_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,input,hidden_prev,hidden_2=None):
        input=input.unsqueeze(1)
        #input [batch_size,1]
        embedded=self.embedding(input)
        #embedded [batch_size, 1, embedding_dim]

        # hidden_prev [num_layers, batch_size, hidden_dim]
        if self.arch==1:
            output,hidden=self.rnn(embedded,hidden_prev)
        else:
            output,(hidden,cell)=self.rnn(embedded,hidden_prev)

        #Output [batch_size, seq_len , hid_dim] The one we threw away in encoder
        #hidden [num_layers, batch_size, hid_dim]

        output=torch.cat((output,embedded),dim=2)
        #output [batch_size, seq_len,hidden+embedded]
        prediction=self.fc_out(output)
        return prediction,hidden


class Seq2Seq(nn.Module):
    """
    Wrapper seq2seq class which encodes first and then performs decoding step 1 by 1
    Parameters
    ---------
    encoder ``Encoder`` required
        The encoder instance
    decoder ``Decoder`` required
        The decoder instance
    device required
    teacher_forcing_ratio ``int`` Optional
        Fraction of the time actual output is given to decoder as input rather than prev output
        Default - 0.5
    """
    def __init__(self,encoder,decoder,device,teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device
        self.teacher_forcing_ratio=0.01

    def forward(self,src_batch,trg_batch,context_batch=None):
        batch_size=src_batch.shape[0]
        trg_len=trg_batch.shape[1]
        trg_vocab_size=self.decoder.output_dim

        outputs=torch.zeros(batch_size,trg_len,trg_vocab_size).to(self.device)
        #tensor to store probabilities of each word at each time step

        context_vector=self.encoder(src_batch)
        #context_vector [batch_size, encoder_hidden_size*dir]

        if context_batch is not None:
            context_vector2=self.encoder(context_batch)
            context_vector=torch.cat((context_vector,context_vector2),dim=1)

        hidden=context_vector.unsqueeze(0).repeat(self.decoder.num_layers,1,1)
        #hidden_2 is a copy of context, can feed it to decoder fc at all time steps also
        hidden_2=context_vector

        #First input should be <START> token
        input=torch.tensor(trg_batch[:,0])
        for t in range(1,trg_len):
            output,hidden=self.decoder(input,hidden,hidden_2)
            #hidden [1(seq_len), batch_size, dec_hidden]
            #output [1(seq_len), batch_size, vocab_size(dec)]

            output=output.squeeze(0)
            outputs[:,t:t+1,:]=output

            top1=output.argmax(2)

            if(random.random()<self.teacher_forcing_ratio):
                input=trg_batch[:,t]
            else:
                input=top1.squeeze(1)

        return outputs
