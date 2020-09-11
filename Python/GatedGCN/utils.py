"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

try:
    from GatedGCN.train import *
    from GatedGCN.gated_gcn_net import *
except:    
    sys.path.append('/home/nmiguens/Python')
    from GatedGCN.train import *
    from GatedGCN.gated_gcn_net import *


"""
    TRAINING CODE
"""

def train_val_pipeline(train_loader, test_loader, val_loader, params, net_params):
    t0 = time.time()
    per_epoch_time = []
    
    #device = torch.device( "cuda:{}".format(net_params["gpu"]) if torch.cuda.is_available() else "cpu" )
    device = torch.device("cpu")
    net_params["device"] = device
    model = GatedGCNNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], [] 
    
    # import train functions for all other GCNs
    from GraphSAGE.train import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                start = time.time()
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)                
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    
    return model, [epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs]

def init_params(trainset, nb_classes, epoch = 1000):
    """Initialise les paramètres"""
    jsonPath = "/home/nmiguens/Python/GatedGCN/graph_classification_GatedGCN.json"
    with open(r"{}".format(jsonPath), "r") as read_file:
        data = json.load(read_file)
    net_params = data["net_params"]
    params = data["params"]
    
    net_params['n_classes'] = nb_classes
    net_params['in_dim'] = trainset[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = trainset[0][0].edata['feat'][0].size(0)
    
    return net_params, params












