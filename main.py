from model import *
from utils import *
import config as config
from torchtext.data import metrics
from torchtext.data import Iterator,BucketIterator
import random

train_iterator = get_loader(config.squad_path)

ENC_EMB_DIM = config.ENC_EMB_DIM
HID_DIM = config.ENC_HID_DIM
ENC_DROPOUT = config.ENC_DROPOUT
input_vocab = len(train_iterator.dataset.word2ind_input)

pretrained_wvs = torch.ones((input_vocab,ENC_EMB_DIM))

enc=Encoder_Model(n_vocab=input_vocab,
                pretrained_vecs=pretrained_wvs,
                embedding_dim=ENC_EMB_DIM,
                hidden_dim=HID_DIM,
                dropout=ENC_DROPOUT)

output_dim = len(train_iterator.dataset.word2ind_output)
pretrained_wvs = torch.ones((output_dim,config.DEC_EMBEDDING_DIM))
HID_DIM = config.DEC_HID_DIM
id2word_output = train_iterator.dataset.id2word_output
print(id2word_output)
dec=Decoder(output_dim=output_dim,
            emb_dim=config.DEC_EMBEDDING_DIM,
            hid_dim=HID_DIM,
            dropout=config.DEC_DROPOUT,
            pretrained_vecs=pretrained_wvs)

device = config.device
model = Seq2Seq(enc,dec,device).to(device)
if config.USING_SAVED:
    pass
else:
    model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

print("Hello")
for i in enumerate(train_iterator):
    print(i[1][0].shape)
    break

#train_=BatchWrapper(train_iterator,'context','text','question')

def train(model,iterator,optimizer,criterion,clip):
    model.train()
    epoch_loss = 0

    sent_ = []
    sent_correct_ = []
    for i,batch in enumerate(iterator):

        input=batch[0]
        output=batch[2]
        context=batch[1]

        print(f"Batch Number {i}")
        optimizer.zero_grad()
        output_pred=model(input,output,context)

        if i%10 == 0:
            #for batch_no in range(output_pred.shape[0]):
            sent = []
            sent_correct = []

            for k in range(1,output_pred.shape[1]):
                sent.append(id2word_output[output_pred[0,k,:].argmax().item()])
                sent_correct.append(id2word_output[output[0,k].item()])
            sent_.append(sent)
            sent_correct_.append(sent_correct)
            return 0,sent_,sent_correct_
        output_pred = output_pred[:,1:,:]
        output = output[:,1:]
        output_dim = output_pred.shape[-1]
        output_pred = output_pred.reshape(-1,output_dim)

        output = output.reshape(output.shape[0]*output.shape[1])

        loss = criterion(output_pred,output)
        epoch_loss += loss.item()
        loss.backward()
        print(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()

    return epoch_loss/len(iterator),sent_,sent_correct_


N_EPOCHS=1#config.N_EPOCHS
CLIP=config.CLIP
loss_epochs=[]
for epoch in range(N_EPOCHS):
    train_loss,sent_,sent_correct_ = train(model,train_iterator,optimizer,criterion,CLIP)
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
torch.save(model.state_dict(), config.weight_save_path)
