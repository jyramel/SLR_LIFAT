# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:52:00 2020

@author: natha
"""
import numpy as np
from ipycanvas import Canvas
from matplotlib import pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from sklearn.model_selection import train_test_split
from scipy.linalg import block_diag


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
    A = np.zeros((len(graph), len(graph)))
    for index, node in enumerate(graph):
        for voisin in node :
            A[index, voisin] = 1
            A[voisin, index] = 1
    return A

def AdjacencyMatrixHand():
    """ Retourne la matrice adjacente du graph de la main """
    A = np.zeros((21,21))
    A = OnePrintLigne(A, 0, [1,5,9,13,17])
    for index in range(1, 21):
        if index % 4 == 0 :
            A[index, index-1] = 1
        elif index % 4 == 1 : 
            A[index, index+1] = 1
            A[index, 0] = 1
        else :
            A = OneVoisins(A, index)
    return A
    
def OnePrintLigne(A, index, pointsList):
    """ Utile pour le calcul de la matrice adjacente du graph de la main """
    for i in pointsList :
        A[index, i] = 1
    return A

def OneVoisins(A, index):
    """ Utile pour le calcul de la matrice adjacente du graph de la main """
    A[index, index-1] = 1
    A[index, index +1] = 1
    return A

def getNodes(X):
    """ Retourne la liste des graphs obtenus avec les keypoints de X"""
    nodes = np.zeros((X.shape[0], 21, 3))
    for index, x in enumerate(X):
        hand = leftBestHand(x)
        if hand : 
            node = getNode(x[:21*3])
        else :
            node = getNode(x[21*3:])
        nodes[index] = node
    return nodes

def getNode(x):
    """ Retourne le graph de la main correspondant aux keypoints de x """
    node = np.zeros((21, 3))
    node[0] = np.array((0, 0, x[2]))
    for index in range(1,21):
        score = getScore(x, index)
        angle = getAngle(x, index)
        length = getLength(x, index)
        node[index] = np.array((angle, length, score))
    
    return node

def getPoints(x, index):
    """ Retourne les points """
    return x[3*index:3*index+2]

def getScore(x, index):
    """ Retourne le score """
    return x[3*index+2]

def getAngle(x, index):
    """ Retourne la valeur d'angle correspondant au point de la main indexé par index"""
    if index in [1,5,9,13,17]:
        pt1 = getPoints(x, 9)
        pt2 = getPoints(x, index)
        center = getPoints(x, 0)
    elif index in [2,6,10,14,18]:
        pt1 = getPoints(x, 0)
        pt2 = getPoints(x, index)
        center = getPoints(x, index-1)
    else :
        pt1 = getPoints(x, index-2)
        pt2 = getPoints(x, index)
        center = getPoints(x, index-1)
    return calcAngle(pt1, center, pt2)

def getLength(x, index):
    """ Retourne la valeur de longueur correspondant au point de la main indexé par index"""
    if index in [1,5,9,13,17]:
        pt1 = getPoints(x, 0)
        pt2 = getPoints(x, index)
    else :
        pt1 = getPoints(x, index-1)
        pt2 = getPoints(x, index)
    return calcLength(pt1, pt2)    

def calcAngle(pt1, center, pt2):
    """ Retourne la valeur de l'angle (pt1 center pt2) en radian modulo pi """
    
    a = np.linalg.norm(pt1 - center)
    if a != 0 :
        vec1 = (pt1 - center) / a
    else :
        vec1 = np.array([0,0])
    
    a = np.linalg.norm(pt2 - center)
    if a != 0 :
        vec2 = (pt2 - center) / a
    else :
        vec2 = np.array([0,0])
    return np.angle(vec2[0]+vec2[1] * 1j) - np.angle(vec1[0]+vec1[1] * 1j)

def calcLength(pt1, pt2):
    """ Retourne la longueur entre deux points """
    return np.linalg.norm(pt2 - pt1)

def leftBestHand(x):
    """Retourne la main la mieux déterminée
    - True pour la main gauche
    - False pour la main droite """
    scoreL = 0 
    scoreR = 0
    for score in range(2,2*3*21,3):
        if score <= 3*21 :
            scoreL += x[score]
        else : 
            scoreR += x[score]
    if scoreL >= scoreR :
        return True
    return False

def getBestHand(x):
    """Retourne la main la mieux déterminée"""
    scoreL = 0 
    scoreR = 0
    for score in range(2,2*3*21,3):
        if score <= 3*21 :
            scoreL += x[score]
        else : 
            scoreR += x[score]
    if scoreL >= scoreR :
        return x[3*21:]
    return x[:3*21]

def displayGraph(graph, coef = 1):
    """Affiche le graph en plt et en canvas"""
    x, y = calcXY(graph)
    canvas = Canvas(width=200, height=200)
    canvas = printCircles(x, y, canvas)
    canvas = printText(x,y, canvas)
    canvas = printBones(x,y, canvas)
    
    handGraph = [[1, 5, 9, 13, 17], 
             [0,2], [1,3], [2,4], [3], 
             [0,6], [5,7], [6,8], [7], 
             [0,10], [9,11], [10,12], [11],
             [0,14], [13,15], [14,16], [15],
             [0,18], [17,19], [18,20], [19]]
    
    plt.scatter(x,y)
    for index in range(len(x)):
        for link in handGraph[index]:
            if index <= link :
                plt.plot([x[index], x[link]], [y[index], y[link]])
                
    plt.axis('equal')
    
    plt.show()
    
    return canvas
    
def calcXY(graph):
    """Transforme en coordonnées polaires relatives en coordonnées cartésiens générales"""
    x = [0 for i in range(21)]
    y = [0 for i in range(21)]
    y[9] = graph[9,2]
    for index in [1,5, 9, 13,17]:
        node = graph[index]
        x[index] = node[1] * np.sin(node[0])
        y[index] = node[1] * np.cos(node[0])
    for finger in range(2, 19, 4):
        x,y = addFinger(finger, x, y, graph)
    return x, y

def addFinger(indexFinger, x, y, graph):
    """Ajoute à x,y le doigt indexFinger"""
    vec = np.array((x[indexFinger-1], y[indexFinger-1]))
    v10 = - vec / np.linalg.norm(vec)
    v10p = orthoVec(v10)
    
    for index in range(indexFinger, indexFinger + 3):
        
        ri = graph[index,1]
        ti = graph[index,0]
        vec = ri * np.cos(ti) * v10 + ri * np.sin(ti) * v10p
        
        x[index], y[index] = x[index-1] + vec[0], y[index-1] + vec[1]
        
        vec = np.array((x[indexFinger-1], y[indexFinger-1]))
        v10 = - vec / np.linalg.norm(vec)
        v10p = orthoVec(v10)
        
    return x, y

def orthoVec(vec):
    """Calcul le vecteur orthogonal de vec dans le plan"""
    x = np.concatenate((vec, 0), axis = None)
    x = x / np.linalg.norm(x)
    ortho = np.cross(x, np.array((0,0,1)))
    return ortho[:2]

def printCircles(x,y, canvas):
    """Affiche les keypoints"""
    canvas.fill_style = 'red'
    for index in range(len(x)):
        canvas.fill_arc(x[index], y[index], 3, 0, np.pi)
    return canvas

def printText(x,y, canvas):
    """Affiche le numéro des keypoints"""
    canvas.font = '8px serif'
    for index in range(len(x)):
        canvas.fill_text('{}'.format(index), x[index] + 5, y[index] + 5)
    return canvas

def printBones(x,y, canvas):
    """Affiche le lien entre les keypoints"""
    handGraph = [[1, 5, 9, 13, 17], 
             [0,2], [1,3], [2,4], [3], 
             [0,6], [5,7], [6,8], [7], 
             [0,10], [9,11], [10,12], [11],
             [0,14], [13,15], [14,16], [15],
             [0,18], [17,19], [18,20], [19]]
    for index in range(len(x)):
        for link in handGraph[index]:
            if index <= link :
                canvas.line_width = 2
                canvas.begin_path()
                canvas.move_to(x[index], y[index])
                canvas.line_to(x[link], y[link])
                canvas.stroke()
    return canvas


def splitXy(X, y):
    """"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

    trainset = HandGrapClass(X_train, y_train)
    validset = HandGrapClass(X_val, y_val)
    testset = HandGrapClass(X_test, y_test)
    
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)
    valid_loader = DataLoader(validset, batch_size=32, collate_fn=collate)
    test_loader = DataLoader(testset, batch_size=32, collate_fn=collate)
    
    return [trainset, validset, testset], [train_loader, valid_loader, test_loader]



def collate(samples):
    """ The input `samples` is a list of pairs (graph, label)."""
    batched_nodes, batched_edges, labels = map(list, zip(*samples))
    
    graph_shape = list(map(lambda g: g.shape[0], batched_nodes))
    
    # Return Node features, adjacency matrix, graph size and labels
    return  torch.tensor(np.concatenate(batched_nodes, axis=0)).float(), \
            torch.tensor(block_diag(*batched_edges)).float(), \
            torch.tensor(graph_shape), \
            torch.tensor(labels)


# --------------------------------------------------------------------------- #
class NxHand(data.Dataset):
    
    def __init__(self, x, seuil):
        self.x = x
        self.auriculaire = NxFinger(x, 17, seuil)
        self.annulaire = NxFinger(x, 13, seuil)
        self.majeur = NxFinger(x, 9, seuil)
        self.index = NxFinger(x, 5, seuil)
        self.pouce = NxFinger(x, 1, seuil)
        self.fingers = [self.pouce, self.index, self.majeur, self.annulaire, self.auriculaire]
        self.graph = nx.Graph()
        self.update()
        
    def update(self):
        twoFingers1 = nx.union(self.auriculaire.graph, self.annulaire.graph)
        twoFingers2 = nx.union(self.majeur.graph, self.index.graph)
        threeFingers = nx.union(twoFingers2, self.pouce.graph)
        self.graph = nx.union(threeFingers, twoFingers1)
        self.join_fingers()
        
    def join_fingers(self):
        self.graph.add_node(0)
        self.graph.nodes[0]['angle'] = 0
        self.graph.nodes[0]['keypoint'] = 0
        self.graph.nodes[0]['finger'] = -1
        
        for finger in self.fingers :
            if finger.first_node != None :
                self.graph.add_edge(0, finger.first_node)
                pt1, pt2 = getPoints(self.x, 0), getPoints(self.x, finger.first_node)
                self.graph.edges[0,finger.first_node]['length'] = calcLength(pt1, pt2)
                
    
        

class NxFinger():
    
    def __init__(self, x, num_node, seuil):
        
        self.graph = nx.Graph()
        self.seuil = seuil
        self.first_node = self.get_first_node(x, num_node)
        self.last_node = self.get_last_node(x, num_node)
        
        if self.first_node != None :
            self.add_next_nodes(x)
        
    def get_first_node(self, x, num_node):
        """"""
        for index in range(4):
            if getScore(x, num_node + index) >= self.seuil :
                return num_node + index
        return None
    
    def get_last_node(self, x, num_node):
        """"""
        for index in range(3,-1,-1):
            if getScore(x, num_node + index) >= self.seuil :
                return num_node + index
        return None
    
    def add_next_nodes(self,x):
        """"""
        self.graph.add_node(self.first_node)
        
        prec_node = 0
        current_node = self.first_node
        next_node = current_node + 1
        
        while current_node != self.last_node :
            if getScore(x, next_node) >= self.seuil :
                
                self.graph.add_node(next_node)
                pt1, center, pt2 = getPoints(x, prec_node), getPoints(x, current_node), getPoints(x, next_node)
                self.graph.nodes[current_node]['angle'] = calcAngle(pt1, center, pt2)
                self.graph.nodes[current_node]['keypoint'] = current_node
                self.graph.nodes[current_node]['finger'] = (current_node - 1)//4
                self.graph.add_edge(current_node, next_node)
                self.graph[current_node][next_node]['length'] = calcLength(center, pt2)
                
                prec_node = current_node
                current_node = next_node
                next_node += 1
            else :
                next_node += 1
        self.graph.nodes[current_node]['angle'] = 0
        self.graph.nodes[current_node]['keypoint'] = current_node
        self.graph.nodes[current_node]['finger'] = (current_node - 1)//4
        return 
    
class HandGrapClass(data.Dataset):
  def __init__(self, X, y):
    
    self.X = X
    self.nodes = getNodes(X)
    self.edge = np.ones((21,21)) - np.identity(21) # AdjacencyMatrixHand()
    
    # Labels to numeric value
    self.labels = y
    self.unique_labels = np.unique(self.labels)
    self.num_classes = len(self.unique_labels)
    
    self.labels = [np.where(target == self.unique_labels)[0][0] 
                   for target in self.labels]
    
    
  def __getitem__(self, index):
    # Read the graph and label
    
    target = self.labels[index]
    nodes = self.nodes[index]
    edges = self.edge #s[index]
    
    return nodes, edges, target
  
  def label2class(self, label):
    # Converts the numeric label to the corresponding string
    return self.unique_labels[label]
  
  def __len__(self):
    # Subset length
    return len(self.labels)

class GraphConvolution(nn.Module):
  """
    Simple graph convolution
  """
  
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
    
    output1 = torch.matmul(A[0], H)
    res[:,0:self.in_features]=output1
    
    output2 = torch.matmul(A[1], H)
    degree=A[1].sum(axis=0)
    deg=torch.zeros((H.shape[0],self.in_features))
    deg[:,0]=degree
    deg[:,1]=degree
    deg=deg+1
    output2=torch.div(output2,deg)
    res[:,self.in_features:2*self.in_features]=output2
    
    output3 = torch.matmul(A[2], H)
    output3=torch.div(output3,deg)
    res[:,2*self.in_features:3*self.in_features]=output3
        
    #FC is just a linear function input multiplied by the paramaters W
    output = self.fc(res)
    
    return output

class GraphConvolution2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
class Net(nn.Module):
  def __init__(self, in_dim, hidden_dim, n_classes):
    super(Net, self).__init__()
    self.layers = nn.ModuleList([
        GraphConvolution(in_dim, hidden_dim)] + [
        GraphConvolution(hidden_dim, hidden_dim) for i in range(1)],
    )
    self.classify = nn.Linear(hidden_dim, n_classes)
    
  def forward(self, h, adj, gs):
    # Add self connections to the adjacency matrix
    id = torch.eye(h.shape[0])
    adj2=torch.pow(adj,2)
    for conv in self.layers:
      h = F.relu(conv(h, [id,adj,adj2]))
    
    
    # Average the nodes
    #here we make the mean of the all the node embedding by graph
    #we do that to obtain a single vector by graph
    #we do that for classification purpose
    count=0
    hg=torch.zeros((gs.shape[0],h.shape[1]))
    for i in range(0,gs.shape[0]):
        hg[i]=h[count:count+gs[i]].mean(axis=0)
        count=count+gs[i]
    
    return self.classify(hg)