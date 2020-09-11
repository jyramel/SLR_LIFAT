# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:54:43 2020

@author: natha
"""
import numpy as np
import os
import sys
import time

from builder import *
from utils import *
from loader import *


if __name__=="__main__":
    
    net_params, params = init_params(trainset, nb_classes)
    net_params["gpu"] = 1
    
    m = params["m"] # nombre d'échantillon par classe
    jsonList = get_list_json(m)
    nb_classes = len(get_Classes_mIterations(m))
    s = [0.5, 0.9]
    
    X_train, X_test, y_train, y_test = train_test_split(jsonList, jsonList, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    trainset = STgraph(X_train, s)
    validset = STgraph(X_val, s)
    testset = STgraph(X_test, s)
    train_loader, valid_loader, test_loader = load_data(trainset, validset, testset, 32)
    
    model, A = train_val_pipeline(train_loader, test_loader, valid_loader, params, net_params)
    
    PATH = "/home/nmiguens/Python/GatedGCN/modelSaved.model"
    torch.save(model.state_dict(), PATH)
    


