import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchtext.data import Field
from torchtext.data import TabularDataset


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):

        context = self.data.iloc[index]["context"]
        answer = self.data.iloc[index]["text"]
        question = self.data.iloc[index]["question"]

        return {"question": question, "answer": answer, "context": context}


train_set = SquadDataset("./data/train_small.csv")
print(len(train_set))


params = {"batch_size": 2, "shuffle": True, "num_workers": 2, "pin_memory": False}
train_iterator = torch.utils.data.DataLoader(train_set, **params, drop_last=False)

for i in enumerate(train_iterator):
    print(i)
    print()
