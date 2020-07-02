# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:52:00 2020

@author: natha
"""
import numpy as np

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



def collate(samples):
    """ The input `samples` is a list of pairs (graph, label)."""
    batched_nodes, batched_edges, labels = map(list, zip(*samples))
    
    graph_shape = list(map(lambda g: g.shape[0], batched_nodes))
    
    # Return Node features, adjacency matrix, graph size and labels
    return  torch.tensor(np.concatenate(batched_nodes, axis=0)).float(), \
            torch.tensor(block_diag(*batched_edges)).float(), \
            torch.tensor(graph_shape), \
            torch.tensor(labels)