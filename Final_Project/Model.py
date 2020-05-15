"""## Model.py"""

#Model.py
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

#from infomax import *

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 64), ReLU(), Linear(64, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)


    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)

        out = self.set2set(out, data.batch)
        return out, feat_map[-1]

    # return feature map and y 
    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y        


class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
   

class FF_LSTM(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()     
        self.output = nn.Linear(input_dim, dim)
    def forward(self, x):
        return self.output(x)

class Net(torch.nn.Module):
    def __init__(self, num_features, dim, use_unsup_loss=False, separate_encoder=False):
        super(Net, self).__init__()

        self.embedding_dim = dim

        self.local = True

        self.lstm_encoder = LSTMModel()
        self.unsup_encoder = Encoder(num_features, dim)
        
        self.global_d = FF(2*dim, dim)
        self.global_s = FF_LSTM(dim, dim)

        # For local global
        self.local_g = FF(dim, dim)
        self.global_g = FF(2*dim, dim)

        self.init_emb()

    def init_emb(self):
      initrange = -1.5 / self.embedding_dim
      for m in self.modules():
          if isinstance(m, nn.Linear):
              torch.nn.init.xavier_uniform_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.fill_(0.0)

# Applied changes here
    def forward(self, data):

    # batch_size = data.num_graphs
      if data.x is None:
         data.x  = torch.ones(data.batch.shape[0]).to(device)

      y, M = self.unsup_encoder(data)
      y_, M_ = self.lstm_encoder(data)
    
      g_enc = self.global_d(y)
      s_enc = self.global_s(y_) #smiles encoder # this is wrong 


      mode='fd'
      measure='JSD'
      global_global_loss = global_global_loss_(g_enc, s_enc, data.edge_index, data.batch, measure)
    
      return global_global_loss

    def forward_local_global(self, data):
      y, M = self.unsup_encoder(data)
    
      g_enc = self.global_g(y)
      l_enc = self.local_g(M) #smiles encoder # this is wrong 

      mode='fd'
      measure='JSD'
      local_global_loss = local_global_loss_(l_enc, g_enc, data.edge_index, data.batch, measure)
    
      return local_global_loss