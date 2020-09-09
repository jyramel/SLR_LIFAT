# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:09:05 2020

@author: natha
"""

import numpy as np
import os
import json
import torch.utils.data as data
import torch
import dgl
import glob

from scipy.spatial.distance import cdist


    

    
def dict_to_STgraph(data, seuil):
    """Construit les noeuds du graph avec les frames"""
    STgraph = None
    nb_frame = len(data)
    ratio = 1
    modulo = 0
    for frame in data:
        if type(frame["keypoints"]["pose_keypoints_2d"]) != float :
            oneGraph = oneFrame_to_oneGraph(frame, seuil, nb_frame)
            STgraph = add_oneGraph_to_STgraph(oneGraph, STgraph)
    return STgraph

def add_oneGraph_to_STgraph(oneGraph, STgraph):
    """Ajoute un graph au STgraph"""
    if type(STgraph) != np.ndarray :
        return oneGraph
    nodes = np.concatenate((STgraph, oneGraph), axis=0)
    return nodes

def json_to_dict(jsonPath):
    """Transform un fichier json de keypoints en dictionnaire de numpy array"""
    with open(r"{}".format(jsonPath), "r") as read_file:
        data = json.load(read_file)
    return data

def oneFrame_to_oneGraph(frame, seuil, nb_frame):
    """Prend les keypoints d'une frame pour retourner un graph"""
    valeur_frame = 2*(frame["frame"] -1)/(nb_frame) -1
    frame = frame["keypoints"]
    
    body = np.array(frame["pose_keypoints_2d"])[0][:8]
    handG = np.array(frame["hand_left_keypoints_2d"])[0]
    handD = np.array(frame["hand_right_keypoints_2d"])[0]
    
    body = add_body(body, valeur_frame)
    handG = add_hand(-1, handG, valeur_frame)
    handD = add_hand(1, handD, valeur_frame)
    
    nodes = np.concatenate((handG, handD), axis=0)
    nodes = np.concatenate((nodes, body), axis=0)
    nodes = nodes[np.where(nodes[:,2] > seuil[0])]
    nodes = nodes[np.where(nodes[:,2] < seuil[1])]
    
    return nodes

def add_hand(left, hand, valeur_frame):
    nodes = np.zeros((hand.shape[0], hand.shape[1]+23+8)); nodes[:,:-23-8] = hand
    nodes[:,3:5] = left, valeur_frame
    nodes[:,5:26] = np.identity(21)
    nodes = nodes[np.where(nodes[:,2] != 0)]
    return nodes

def add_body(body, valeur_frame):
    nodes = np.zeros((body.shape[0], body.shape[1]+23+8)); nodes[:,:-23-8] = body
    nodes[:,3:5] = 0, valeur_frame
    nodes[:,26:] = np.identity(8)
    nodes = nodes[np.where(nodes[:,2] != 0)]
    return nodes

def get_STgraphs(jsonList, seuil):
    """Retourne la list des noeuds, arcs et y de la list de JSON"""
    Nodes = list()
    Y = list()
    path = ""
    for json in jsonList:
        node = dict_to_STgraph(json_to_dict(path + json), seuil)           
        mean, var = all_var_mean()
        node[:,0:2] = (node[:,0:2] - mean[0:2])/np.sqrt(var[0:2] + 1e-10)
        Nodes.append(node)
        Y.append(1)
    return Nodes, Y

def compute_adjacency_matrix(nodes, sigmas):
    # Compute distances
    s_dist = cdist(nodes[:,0:2], nodes[:,0:2])
    t = np.zeros((nodes[:,3].reshape(-1,1).shape[0], 2)); t[:,:-1] = nodes[:,3].reshape(-1,1)
    t_dist = cdist(t, t)
    m = np.zeros((nodes[:,4].reshape(-1,1).shape[0], 2)); m[:,:-1] = nodes[:,4].reshape(-1,1)
    m_dist = cdist(m, m)
    
    S = nodes[:,2].reshape(-1,1).dot(nodes[:,2].reshape(1,-1))
    
    A = np.exp(-((s_dist/sigmas[0])**2 + (t_dist/sigmas[1])**2 + (m_dist/sigmas[2])**2)/S)
    
    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    
    return A

def compute_edges_list(A, kth=8+1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth
    
    if num_nodes > 9:
        knns = np.argpartition(A, new_kth-1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth-1, axis=-1)[:, new_kth:-1] # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A # NEW
        
        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1) # NEW
            knns = knns[knns != np.arange(num_nodes)[:,None]].reshape(num_nodes,-1)
    return knns, knn_values # NEW

class STgraph(data.Dataset):
    def __init__(self, jsonList, seuil = [0.5, 0.99], sigmas = [1,1,1], kth = 9):
        
        self.nodes, self.labels = get_STgraphs(jsonList, seuil)
        self.graph_labels = torch.LongTensor(self.labels)
        self.sigmas = sigmas
        self.kth = kth
        self.graph_lists = []
        self._prepare()
        
    def __getitem__(self, idx):
        # Read the graph and label
        return self.graph_lists[idx], self.graph_labels[idx]
    
    def __len__(self):
        # Subset length
        return len(self.labels)
    
    def _prepare(self):
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []
        for index, sample in enumerate(self.nodes):
            A = compute_adjacency_matrix(sample, self.sigmas)
            edges_list, edge_values_list = compute_edges_list(A, kth = self.kth) # NEW

            N_nodes = A.shape[0]

            edge_values_list = edge_values_list.reshape(-1) # NEW # TO DOUBLE-CHECK !
            
            self.node_features.append(np.array(sample))
            self.edge_features.append(edge_values_list) # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
        
        for index in range(len(self.nodes)):
            g = dgl.DGLGraph()
            #print(g.device)
            #g = g.to('cuda:0')
            #print(g.device)
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata['feat'] = torch.Tensor(self.node_features[index]).half() 

            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts!=src])
            
            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = g.ndata['feat'].shape[1] # dim same as node feature dim
            #g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half() 
            g.edata['feat'] = torch.Tensor(self.edge_features[index]).unsqueeze(1).half()  # NEW 

            g = g.to('cuda:0')
            self.graph_lists.append(g)

def all_var_mean():
    """Calcul la variance et la moyenne des keypoints des vidéos
       Ecrit le résultat dans un fichier JSON
    """
    path = ""
    jsonFiles = glob.glob("*.json")
    for file in jsonFiles:
        with open(r"{}".format(os.path.join(path, file)), "r") as read_file:
            data = json.load(read_file)
        return get_var_mean(data)
        
        
def get_var_mean(data):        
    """Calcul la variance et la moyenne des keypoints d'une vidéo"""
    nodes = np.array([[0,0,0]])
    for Frame in data :
        frame = Frame["keypoints"]
        if type(frame["pose_keypoints_2d"]) != float : 
            body = np.array(frame["pose_keypoints_2d"])[0][:9]
            handG = np.array(frame["hand_left_keypoints_2d"])[0]
            handD = np.array(frame["hand_right_keypoints_2d"])[0]
            nodes = np.concatenate((nodes, body), axis=0)
            nodes = np.concatenate((nodes, handG), axis=0)
            nodes = np.concatenate((nodes, handD), axis=0)
    nodes = nodes[~np.all(nodes == 0, axis = 1)]
    return np.mean(nodes, axis = 0), np.var(nodes, axis = 0)
