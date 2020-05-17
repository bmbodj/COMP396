
#LSTM_Model.py
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#Add default information from the beginning change later 
'''
batch_size = 64
lr = 1e-3
weight_decay = 1e-4
max_len = 110
embedding_size = 1
hidden_size = 64
output_size = 1  
use_cuda = True
'''
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, padding_idx=padding_idx , embedding_size=1, hidden_size=64, output_size=1,max_len=25):
        super(LSTMModel, self).__init__()
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        
        # LSTM (RNN)
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Fully connected layer
        #self.output = nn.Linear(hidden_size, output_size)
 
        
    def forward(self, data):  # batch_size x seq_length
      
        input=np.asarray(data.smiles)
        out=torch.from_numpy(input).to(device)

        out=out.reshape(out.shape[0],max_len)
        embedded = self.embed(out)
        # Run RNN
        o, _ = self.rnn(embedded) 

        # Max pooling across sequence
        y, _ = torch.max(o, dim=1)    
        
        # Output layer
       # out = self.output(o)  
        
        return y,o # Means return y and the feature map
