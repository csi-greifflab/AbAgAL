from __future__ import print_function
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class AbAgConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence_length = 15
        self.convolution = nn.Conv2d(in_channels=1,
                                     out_channels=400,
                                     kernel_size=(8, 5),  # Input: 18x20, Output: 11x16
                                     padding=(1, 1))  # Output: 13x18
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=20)  # Output: 18x20
        self.dropout = nn.Dropout(p=0.2)
        self.max_pooling = nn.MaxPool2d((2, 2), stride=1)  # Output: 12x17
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400 * (self.sequence_length - 8 + 2) * (20 - 5 + 2), 300)
        self.fc2 = nn.Linear(300, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        #print("Tensor device:", x.device)
        x = self.embedding(x).view(-1, 1, self.sequence_length, 20)
        #print("Tensor device:", x.device)
        x = self.convolution(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class AbAgConvNet_grad(nn.Module):
    def __init__(self):
        super().__init__()

        #self.sequence_length = 18
        self.sequence_length = 15
        self.convolution = nn.Conv2d(in_channels=1,
                                     out_channels=400,
                                     kernel_size=(8, 5),  # Input: 18x20, Output: 11x16
                                     padding=(1, 1))  # Output: 13x18
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=20)  # Output: 18x20
        self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(p=0.2)
        self.max_pooling = nn.MaxPool2d((2, 2), stride=1)  # Output: 12x17
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400 * (self.sequence_length - 8 + 2) * (20 - 5 + 2), 300)
        self.fc2 = nn.Linear(300, 1)
        self.activation = nn.LeakyReLU()
        self.embetter = 0  

    def forward(self, x):
        #print("Tensor device:", x.device)
        self.embetter = self.embedding(x)
        self.embetter.requires_grad = True
        x = self.embetter.view(-1, 1, self.sequence_length, 20) 
        #print("Tensor device:", x.device)
        x = self.convolution(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class AbAgConvNet_confounding(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence_length = 15
        self.convolution = nn.Conv2d(in_channels=1,
                                     out_channels=400,
                                     kernel_size=(8, 5),  # Input: 18x20, Output: 11x16
                                     padding=(1, 1))  # Output: 13x18
        self.embedding = nn.Embedding(num_embeddings=20, embedding_dim=20)  # Output: 18x20
        self.dropout = nn.Dropout(p=0.2)
        self.max_pooling = nn.MaxPool2d((2, 2), stride=1)  # Output: 12x17
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400 * (self.sequence_length - 8 + 2) * (20 - 5 + 2), 300)
        self.fc2 = nn.Linear(300, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        #print("Tensor device:", x.device)
        x = self.embedding(x).view(-1, 1, self.sequence_length, 20)
        #print("Tensor device:", x.device)
        x = self.convolution(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
def one_hot_encoder(sequence):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_indices = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
    encoded_sequence = np.zeros((len(amino_acids), len(sequence)))
    encoded_sequence[[amino_acid_indices[aa] for aa in sequence], range(len(sequence))] = 1
    return encoded_sequence


def sequence_embedding(sequence):
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    amino_acid_indices = {amino_acid: index for index, amino_acid in enumerate(amino_acids)}
    sequence_encoded = [amino_acid_indices[aa] for aa in sequence]
    return sequence_encoded


class AbAgDataset(Dataset):
    def __init__(self, device, df=None, data_file=None):
        if df is not None:
            self.df = df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(data_file, sep='\t').reset_index(drop=True)
        if not ('AASeq' in self.df.columns):
            self.df['AASeq'] = self.df.apply(lambda row: row.AgSeq[:20] + row.AbSeq, axis=1)
        self.x = torch.LongTensor([sequence_embedding(sequence) for sequence in self.df.AASeq])
        self.y = torch.FloatTensor(self.df[['BindClass']].to_numpy().reshape(-1))
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
