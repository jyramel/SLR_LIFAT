# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:52:00 2020

@author: natha
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, nb_couches, n_classes):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([
            GraphConvolution(in_dim, hidden_dim)] + [
            GraphConvolution(hidden_dim, hidden_dim) for i in range(nb_couches)],
        )
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.dropout = 0.5
        
    def forward(self, h, adj, gs):
        # Add self connections to the adjacency matrix
        device = torch.device( "cuda:1" if torch.cuda.is_available() else "cpu" )
        id = torch.eye(h.shape[0]).to(device)
        adj = adj.to(device)
        adj2 = torch.pow(adj,2).to(device)
        for conv in self.layers:
            h = F.relu(conv(h, [id,adj,adj2]))
            h = F.dropout(h, self.dropout, training=self.training)
        # Average the nodes
        #here we make the mean of the all the node embedding by graph
        #we do that to obtain a single vector by graph
        #we do that for classification purpose
        count=0
        hg=torch.zeros((gs.shape[0],h.shape[1]))
        for i in range(0,gs.shape[0]):
            hg[i]=h[count:count+gs[i]].mean(axis=0)
            count=count+gs[i]

        return self.classify(hg.to(device))
    
class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, batchnorm=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.fc = nn.Linear(3*self.in_features, self.out_features, bias=self.bias)

        self.batchnorm = batchnorm
    
      
      #x are node features for all graphs batch
      #W are adjacency matrix for all graphs batch
      # GraphConv = AHW
    
    def forward(self, H, A):
        res = torch.zeros((H.shape[0],self.in_features*3))
        device = torch.device( "cuda:1" if torch.cuda.is_available() else "cpu" )
        res = res.to(device)
        H = H.to(device)
        A = [a.to(device) for a in A]
        
        output1 = torch.matmul(A[0], H)
        output1 = output1.to(device)
        res[:,0:self.in_features] = output1

        output2 = torch.matmul(A[1], H)
        output2 = output2.to(device)
        degree= A[1].sum(axis=0)
        
        deg = torch.zeros((H.shape[0],self.in_features))
        deg = deg.to(device)
        deg[:,0]=degree
        deg[:,1]=degree
        deg=deg+1
        output2=torch.div(output2,deg)
        res[:,self.in_features:2*self.in_features]=output2

        output3 = torch.matmul(A[2], H)
        output3 = output3.to(device)
        output3=torch.div(output3,deg)
        res[:,2*self.in_features:3*self.in_features]=output3

        #FC is just a linear function input multiplied by the paramaters W
        output = self.fc(res)

        return output