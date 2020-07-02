# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:13:36 2020

@author: natha
"""
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm

def train(model, train_loader, nb_epoch, display = True):
    device = torch.device( "cuda:1" if torch.cuda.is_available() else "cpu" )
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    epoch_losses = []
    if display : 
        for epoch in tqdm(range(nb_epoch)):
            epoch_loss = 0
            for iter, (bn, be, gs, label) in enumerate(train_loader):
                bn, be, gs, label = bn.to(device), be.to(device), gs.to(device), label.to(device)
                prediction = model(bn, be, gs)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            #print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
    else :
        for epoch in range(nb_epoch):
            epoch_loss = 0
            for iter, (bn, be, gs, label) in enumerate(train_loader):
                bn, be, gs, label = bn.to(device), be.to(device), gs.to(device), label.to(device)
                prediction = model(bn, be, gs)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            #print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
    return epoch_losses

def display_losses(epoch_losses, nb = 1, save = False):
    plt.plot(epoch_losses)
    plt.yscale('log')
    if save :
        plt.savefig('test_{}.png'.format(nb))
    else :
        plt.show()
    return epoch_losses[-1]

def accuracy(output, target):
    """Accuacy given a logit vector output and a target class """
    _, pred = output.topk(1)
    pred = pred.squeeze()
    correct = pred == target
    correct = correct.float()
    return correct.sum() * 100.0 / correct.shape[0]

def test(model, test_loader, testset):
    """Test le model"""
    device = torch.device( "cuda:1" if torch.cuda.is_available() else "cpu" )
    predictions = []
    model.eval()
    acc = 0
    with torch.no_grad():
        for iter, (bn, be, gs, label) in enumerate(test_loader):
            bn, be, gs, label = bn.to(device), be.to(device), gs.to(device), label.to(device)
            prediction = model(bn, be, gs)
            predictions.append(prediction)
            predictions.append(label)
            acc += accuracy(prediction, label) * label.shape[0]
    acc = acc/len(testset)
    return acc, predictions

def confusion_M(predictions, save = False):
    for iter, pred in enumerate(predictions):
        if iter % 2 == 0 :
            if iter == 0 :
                y_pred = pred.topk(1)[1].squeeze().cpu()
            else :
                y_pred = torch.cat([y_pred, pred.topk(1)[1].squeeze().cpu()], axis = 0)
        else :
            if iter == 1 :
                label = pred.cpu()
            else :
                label = torch.cat([label, pred.cpu()], axis = 0)
    confusion_matrix = pd.crosstab(label, y_pred)
    sn.heatmap(confusion_matrix, annot=False)
    if not save :
        plt.show()
    else :
        plt.savefig('CM.png')