from model import *
from utils import *

ENC_EMB_DIM=100
DEC_EMB_DIM=100
HID_DIM=10
ENC_DROPOUT=0.5
DEC_DROPOUT=0.5
Input_vocab=len(TEXT.vocab)
pretrained_wvs=torch.ones((Input_vocab,100))


enc=Encoder_Model(n_vocab=Input_vocab,pretrained_vecs=pretrained_wvs
                  ,embedding_dim=ENC_EMB_DIM,hidden_dim=HID_DIM,dropout=ENC_DROPOUT)
from torchtext.data import metrics
Output_dim=len(LABEL.vocab)
pretrianed_wvs=torch.ones((Output_dim,100))
HID_DIM=40
dec=Decoder(output_dim=Output_dim,emb_dim=DEC_EMB_DIM,hid_dim=HID_DIM,
            dropout=DEC_DROPOUT,pretrained_vecs=pretrianed_wvs)

use_cuda=torch.cuda.is_available()
device=torch.device("cuda:0" if use_cuda else "cpu")
model=Seq2Seq(enc,dec,device).to(device)

optimizer=optim.Adam(model.parameters())
import random
TRG_PAD_IDX=LABEL.vocab.stoi[LABEL.pad_token]
criterion=nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

sent=[]
sent_correct=[]

train_iterator=BucketIterator(train,
                             batch_size=32,
                             device=-1,
                             sort_key=lambda x: len(x.context),
                             sort_within_batch=True,
                             repeat=False)


#train_=BatchWrapper(train_iterator,'context','text','question')

def train(model,iterator,optimizer,criterion,clip):
    model.train()
    epoch_loss=0
    print("Here")
    sent_=[]
    sent_correct_=[]
    for i,batch in enumerate(iterator):


        input=batch.text
        output=batch.question
        context=batch.context

        print(input.shape)
        print(output.shape)
        print(context.shape)

        print(f"Batch Number {i}")
        optimizer.zero_grad()
        output_pred=model(input,output,context)

        if i==0:

            print("Here")
            #for batch_no in range(output_pred.shape[0]):
            sent=[]
            sent_correct=[]

            for k in range(1,output_pred.shape[1]):
                sent.append(LABEL.vocab.itos[output_pred[0,k,:].argmax()])
                sent_correct.append(LABEL.vocab.itos[output[0,k]])
            sent_.append(sent)
            sent_correct_.append(sent_correct)
            """for p in [1,10,15,20,30]:
                try:
                    print(f"Sent Predicted {sent_[p]}")
                    print(f"Sent Correct {sent_correct_[p]}")
                except IndexError:
                    print("Smaller Batch")
                    continue"""
        break

        output_pred=output_pred[:,1:,:]

        output=output[:,1:]

        output_dim=output_pred.shape[-1]
        output_pred=output_pred.reshape(-1,output_dim)

        output=output.reshape(output.shape[0]*output.shape[1])

        loss=criterion(output_pred,output)
        epoch_loss+=loss.item()
        loss.backward()
        print(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()

    return epoch_loss/len(iterator),sent_,sent_correct_





N_EPOCHS=1
CLIP=1
optimizer=optim.Adam(model.parameters())
for epoch in range(N_EPOCHS):
    train_loss,sent_,sent_correct_=train(model,train_iterator,optimizer,criterion,CLIP)
    print(f"Epoch Number{epoch} Train LOSS {train_loss: .3f} ")
print(sent_)
print(sent_correct_)
print(len(sent_),len(sent_correct_))
print(metrics.bleu_score(sent[0],sent_correct_[0]))
torch.save(model.state_dict(), 'model.pt')
