import torch
weight_save_path = './model_weight.pt'
USING_SAVED = False
squad_path = '../train_smaller.csv'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

min_word_count_inp = 10
min_word_count_out = 1

source_dict_dump_path = './source_dict'
target_dict_dump_path = './target_dict'

batch_size = 32
N_EPOCHS = 10
CLIP = 1

#Encoder Configs
ENC_EMB_DIM = 100
ENC_HID_DIM = 128
ENC_DROPOUT = 0.3
ENC_EMBEDDING_DIM = 300

#Decoder Configs
DEC_HID_DIM=4*ENC_HID_DIM
DEC_EMBEDDING_DIM = 300
DEC_DROPOUT = 0.3
