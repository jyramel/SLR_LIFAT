# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:36:43 2020

@author: natha
"""

import numpy as np
import os
import sys
import time
try:
    from GCN.loader import *
    from GCN.model import *
    from GCN.STgraph import *
    from GCN.utils import *
except:
    sys.path.append('/home/nmiguens/Python')
    from GCN.loader import *
    from GCN.model import *
    from GCN.STgraph import *
    from GCN.utils import *
    
# Hyperparamètres 
    
m = 24
batch_size = 16
nb_epoch = 500


# Paramètres 

ratio = [6,8]
seuil = [0.5,0.75]
nb_couche = [1,2,3]
nb_dimHidden = [64, 256, 1024, 2048]

# Split données

jsonList = get_list_json(m)
nb_classes = len(get_Classes_mIterations(m))
X_train, X_test, y_train, y_test = train_test_split(jsonList, jsonList, test_size=0.2, random_state=42)
_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

def write_txt(message):
    f = open('/home/nmiguens/resultat.txt','a')
    f.write('\n' + message)
    f.close()
    
def test_params(r, s, c, d, only_xy):
    write_txt("\n Test avec r = {}, s = {}, c = {}, d= {}, xy = {}".format(r, s, c, d, only_xy))
    start = time.time()
    trainset = STgraph(X_train, s, r, only_xy = only_xy)
    validset = STgraph(X_val, s, r, only_xy = only_xy)
    testset = STgraph(X_test, s, r, only_xy = only_xy, test = True)
    train_loader, valid_loader, test_loader = load_data(trainset, validset, testset, batch_size)
    
    if only_xy :
        model = GCN(2, d, c, nb_classes)
    else :
        model = GCN(3, d, c, nb_classes)     
    
    epoch_losses = train(model, train_loader, nb_epoch, display = False)
    display_losses(epoch_losses, nb = (r, s, c, d, only_xy), save = True)
    
    acc, predictions = test(model, test_loader, testset)
    write_txt('Test accuracy {:.4f}'.format(acc))
    
    # y_pred = predictions[0].topk(1)[1].squeeze().cpu()
    # label = predictions[1].cpu()
    plt.figure()
    confusion_M(predictions, save = True)
    end = time.time()
    write_txt("Time : {}".format(end - start))
    return end - start

if __name__ == "__main__":
    test_params(3, 0.55, 1, 256, True)