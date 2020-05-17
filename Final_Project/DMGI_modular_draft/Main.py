
import os
import sys
import os.path as osp
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
#from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
#from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""## Training Settings and Procedure"""

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
  
  
  if __name__ == '__main__':
    seed_everything()

    target = 0
    dim = 64
    batch_size = 64
    lamda =0.5
    use_unsup_loss = False
    separate_encoder = False
    alpha=0.5
    epochs = 10
    log_interval = 1


    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()  

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y[:, target].mean().item()
    std = dataset.data.y[:, target].std().item()
    dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std
    print(dataset.data)

    unsup_train_dataset = dataset[2000:13000]
    global_global_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=False)
    local_global_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=False)
    
    lr=0.01
    print(dataset.num_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    
    for epoch in range(1, epochs+1):
        global_global_loss_all = 0
        local_global_loss_all = 0
        loss_all=0
        model.train()
        print("Training..")
        for data, data2 in zip(global_global_train_loader, local_global_train_loader):
            data = data.to(device)
            data2 = data2.to(device)
            optimizer.zero_grad()
            global_global_loss = model(data)
            local_global_loss = model.forward_local_global(data2)
            loss = alpha *global_global_loss + local_global_loss* lamda
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            global_global_loss_all += global_global_loss.item()
            local_global_loss_all += local_global_loss.item()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(global_global_train_loader)))

        if epoch % log_interval == 0:
            model.eval() 
            emb, y = model.unsup_encoder.get_embeddings(global_global_train_loader)
            res = evaluate_embedding(emb, y,std )