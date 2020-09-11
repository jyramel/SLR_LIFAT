# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:52:00 2020

@author: natha
"""
import numpy as np
import dgl

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from scipy.linalg import block_diag



def load_data(trainset, validset, testset, batch_size):
    """Charge les données dans un DataLoader PyTorch"""
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate)
    valid_loader = DataLoader(validset, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=collate)
    
    return train_loader, valid_loader, test_loader



# form a mini batch from a given list of samples = [(graph, label) pairs]
def collate(samples):
    """ The input samples is a list of pairs (graph, label)."""
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels))

    for idx, graph in enumerate(graphs):
        graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
        graphs[idx].edata['feat'] = graph.edata['feat'].float()
    batched_graph = dgl.batch(graphs)

    return batched_graph, labels