# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:09:05 2020

@author: natha
"""

import numpy as np
import os
import json
import sys
import networkx as nx
import torch.utils.data as data
import torch

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
try:
    from loader import *
    from model import *
except:
    sys.path.append('/home/nmiguens/Python/GCN')
    from loader import *
    from model import *
    

def json_to_dict(jsonPath):
    """Transform un fichier json de keypoints en dictionnaire de numpy array"""
    with open(r"{}".format(jsonPath), "r") as read_file:
        data = json.load(read_file)
    return data
    
def dict_to_STgraph(data, seuil, ratio, modulo):
    """Construit le ST graph avec les frames"""
    STgraph = None
    for frame in data :
        if frame["frame"] % ratio == modulo :
            if type(frame["keypoints"]["pose_keypoints_2d"]) != float : 
                oneGraph = oneFrame_to_oneGraph(frame["keypoints"], seuil)
                STgraph = add_oneGraph_to_STgraph(oneGraph, STgraph)
    return STgraph

def add_oneGraph_to_STgraph(oneGraph, STgraph):
    """Ajoute un body graph au ST graph"""
    if not STgraph :
        return oneGraph
    STnodes, STadj = STgraph
    oneNodes, oneAdj = oneGraph
    
    adj = adj_fusion(STadj, oneAdj)
    nodes = np.concatenate((STnodes, oneNodes), axis=0)
    return nodes, adj

handGraph = [[1, 5, 9, 13, 17], 
             [0,2], [1,3], [2,4], [3], 
             [0,6], [5,7], [6,8], [7], 
             [0,10], [9,11], [10,12], [11],
             [0,14], [13,15], [14,16], [15],
             [0,18], [17,19], [18,20], [19]]

bodyGraph = [[1], [0,2,5,8], 
             [1,3], [2,4], [3],
             [1,6], [5,7], [6],
             [1]]

def list_to_adjMatrix(graph):
    """Retourne la matrice adjacente d'un graph défini par une liste"""
    A = np.zeros((len(graph), len(graph)))
    for index, node in enumerate(graph):
        for voisin in node :
            A[index, voisin] = 1
            A[voisin, index] = 1
    return A

def adj_fusion(STadj, oneAdj):
    """Chaque nouveau noeud est lié au même noeud de la frame précédente"""
    adj = block_diag(STadj, oneAdj)
    last_node = adj.shape[0] - 1
    nb_new_node = oneAdj.shape[0] 
    for i in range(nb_new_node):
        adj[last_node - i, last_node - i - nb_new_node] = 1
        adj[last_node - i - nb_new_node, last_node - i] = 1
    return adj
    
def oneFrame_to_oneGraph(frame, seuil):
    """Prend les keypoints d'une frame pour retourner un graph(nodes, adj)"""
    body = np.array(frame["pose_keypoints_2d"])[0][:9]
    handG = np.array(frame["hand_left_keypoints_2d"])[0]
    handD = np.array(frame["hand_right_keypoints_2d"])[0]
    
    nodes = np.concatenate((body, handG), axis=0)
    nodes = np.concatenate((nodes, handD), axis=0)
    
    adj = get_BodyAdj(nodes, seuil)
    return nodes, adj

def get_BodyAdj(nodes, seuil):
    """Retourne la matrice adjacente des nodes d'un body"""
    bodyAdj, handAdj = list_to_adjMatrix(bodyGraph), list_to_adjMatrix(handGraph)
    adj = block_diag(bodyAdj, handAdj, handAdj)
    adj[7, 9], adj[9, 7], adj[4, 9 + 21], adj[9+21, 4] = 1, 1, 1, 1
    length = adj.shape[0]
    for i, node in enumerate(nodes):
        if node[2] <= seuil :
            adj[i,:] = np.zeros((length))
            adj[:,i] = np.zeros((length))
    return adj

def json_to_MV(file):
    """Retourne les moyennes et variances pour la video file"""
    path = "/home/nmiguens/JSON/WLASL_MV/MV.json"
    with open(r"{}".format(path), "r") as read_file:
        data = json.load(read_file)
    return np.array(data[file])    

def all_var_mean():
    """Calcul la variance et la moyenne des keypoints des vidéos
       Ecrit le résultat dans un fichier JSON
    """
    path = "/home/nmiguens/JSON/WLASL"
    outpath = "/home/nmiguens/JSON/WLASL_MV"
    MV = dict()
    jsonFiles = os.listdir(path)
    for file in tqdm(jsonFiles):
        if file.split(".")[1] == 'json' :
            with open(r"{}".format(os.path.join(path, file)), "r") as read_file:
                data = json.load(read_file)
            MV[file] = get_var_mean(data) 
    with open(os.path.join(outpath, "MV.json"), 'w') as fout:
        json.dump(MV , fout)
    return 0
        
        
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
    return np.mean(nodes, axis = 0).tolist(), np.var(nodes, axis = 0).tolist() 

def get_Classes():
    """Renvoie le dictionnaire des classes présente dans WLASL avec le nb d'itération et la liste de ces itérations"""
    classes = dict()
    jsonFiles = os.listdir("/home/nmiguens/JSON/WLASL")
    for file in jsonFiles :
        if file.split(".")[1] == 'json':
            class_it = file.split(".")[0]
            classFile, iteration = class_it.split("_")
            if not classFile in classes :
                classes[classFile] = {"iteration" : 0, "list" : list()}
            classes[classFile]["iteration"] += 1 
            classes[classFile]["list"].append(file)
    return classes

def get_Classes_mIterations(m):
    """Retourne n classes avec m itérations"""
    classes = get_Classes()
    classesList = list()
    for classBdd in classes :
        if classes[classBdd]["iteration"] >= m : 
            classesList.append(classes[classBdd]["list"][0:m])
    return classesList

def get_list_json(m):
    """Retourne la list des fichiers JSON des classes avec m itérations"""
    classes = get_Classes_mIterations(m)
    result = list()
    for idClass in classes :
        result += idClass
    return result

def get_STgraphs(jsonList, seuil, ratio, only_xy = False, test = False):
    """Retourne la list des noeuds, arcs et y de la list de JSON"""
    Nodes, Edges, Y = list(), list(), list()
    path = "/home/nmiguens/JSON/WLASL/"
    for json in jsonList :
        if test :
            node, adj = dict_to_STgraph(json_to_dict(path + json), seuil, ratio, 1)
            mean, var = json_to_MV(json)
            node = (node - mean)/np.sqrt(var + 1e-10)
            y = json.split("_")[0]
            
            # adj = normalize(adj)
            if only_xy :
                Nodes.append(node[:,:2])
            else :
                Nodes.append(node)    
            Edges.append(adj)
            Y.append(y)
        else :    
            for modulo in range(ratio):
                node, adj = dict_to_STgraph(json_to_dict(path + json), seuil, ratio, modulo)
                mean, var = json_to_MV(json)
                node = (node - mean)/np.sqrt(var + 1e-10)
                y = json.split("_")[0]

                   
                if only_xy :
                    Nodes.append(node[:,:2])
                else :
                    Nodes.append(node)    
                Edges.append(adj)
                Y.append(y)    
    Y = np.array(Y)
    unique_labels = np.unique(Y)   
    Y = [np.where(target == unique_labels)[0][0] 
                   for target in Y]
    return Nodes, Edges, Y 

def which_device(n):
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(n))
        print("Running on the GPU")
    else :
        device = torch.device("cpu")
        print("Running on the CPU")
    return device

class STgraph(data.Dataset):
    def __init__(self, jsonList, seuil, ratio, model = "FiG", only_xy = False, test = False):
        
        if model == "FiG":
            self.nodes, self.edges, self.labels = get_STgraphs(jsonList, seuil, ratio, only_xy, test)
        elif model == "FiN": 
            self.nodes, self.edges, self.labels = get_STgraphs(jsonList, seuil, ratio, only_xy, test)

    def __getitem__(self, index):
        # Read the graph and label
        target = self.labels[index]
        nodes = self.nodes[index]
        edges = self.edges[index]

        return nodes, edges, target
    
    def __len__(self):
        # Subset length
        return len(self.labels)
    
    