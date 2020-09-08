from model import *
from utils import *
import config as config
from torchtext.data import metrics
from torchtext.data import Iterator,BucketIterator

import random

ENC_EMB_DIM=config.ENC_EMB_DIM
HID_DIM=config.ENC_HID_DIM
ENC_DROPOUT=config.ENC_DROPOUT
input_vocab=len(TEXT.vocab)
pretrained_wvs=TEXT.vocab.vectors

enc=Encoder_Model(n_vocab=input_vocab,pretrained_vecs=pretrained_wvs
                  ,embedding_dim=ENC_EMB_DIM,hidden_dim=HID_DIM,dropout=ENC_DROPOUT)

output_dim=len(LABEL.vocab)
pretrianed_wvs=LABEL.vocab.vectors
HID_DIM=config.DEC_HID_DIM

dec=Decoder(output_dim=output_dim,emb_dim=config.DEC_EMBEDDING_DIM,
            hid_dim=HID_DIM,
            dropout=config.DEC_DROPOUT,
            pretrained_vecs=pretrianed_wvs)


device=config.device
model=Seq2Seq(enc,dec,device).to(device)

optimizer=optim.Adam(model.parameters())

TRG_PAD_IDX=LABEL.vocab.stoi[LABEL.pad_token]
criterion=nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)



train_iterator=BucketIterator(train,
                             batch_size=config.batch_size,
                             device=config.device,
                             sort_key=lambda x: len(x.context),
                             sort_within_batch=True,
                             repeat=False)


#train_=BatchWrapper(train_iterator,'context','text','question')

def train(model,iterator,optimizer,criterion,clip):
    model.train()
    epoch_loss=0

    sent_=[]
    sent_correct_=[]
    for i,batch in enumerate(iterator):

        input=batch.text
        output=batch.question
        context=batch.context


        print(f"Batch Number {i}")
        optimizer.zero_grad()
        output_pred=model(input,output,context)

        if i%10==0:

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


N_EPOCHS=config.N_EPOCHS
CLIP=config.CLIP
loss_epochs=[]
for epoch in range(N_EPOCHS):
    train_loss,sent_,sent_correct_=train(model,train_iterator,optimizer,criterion,CLIP)
    print(f"Epoch Number{epoch} Train LOSS {train_loss: .3f} ")
    loss_epochs.append(train_loss)
for i in range(len(sent_)):
    print("Predicted Question")
    print(sent_[i])
    print("Actual Question")
    print(sent_correct_[i])
print(loss_epochs)
#print(len(sent_),len(sent_correct_))
#print(metrics.bleu_score(sent[0],sent_correct_[0]))
torch.save(model.state_dict(), './seq2seq_model.pt')
