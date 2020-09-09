# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:19:19 2020

@author: natha
"""

from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from GatedGCN.traitementVideo import keypoints_to_json, dict_to_json
from GatedGCN.builder import STgraph
from GatedGCN.loader import load
from GatedGCN.gated_gcn_net import GatedGCNNet
from GatedGCN.utils import init_params

import torch
import glob
from scipy.special import softmax

class GCNThread(QThread):
    change_value = pyqtSignal(int)
    
    def __init__(self, keypoints):
        QThread.__init__(self)
        self.pbar = QProgressBar()
        self.pbar.setGeometry(0, 0, 300, 50)
        self.pbar.setAlignment(Qt.AlignCenter) 
        self.length = len(keypoints)
        self.pbar.setMaximum(self.length)
        self.keypoints = keypoints
        self.json = []
        self.word = "Une erreur est survenue, veuillez recommencer svp."
        self.labels = ["AVANT", "LIVRE", "BONBON", "CHAISE", "VÊTEMENTS", "ORDINATEUR", "BOIRE", "ALLER (go)", "QUI ?"]
    
    def run(self):
        cnt = 0
        for datum in self.keypoints:
            self.datum2json(cnt, datum)
            self.change_value.emit(cnt)
            cnt+=1

        inputClassifier, object2class = self.loadData()
        model, device = self.initGCN(object2class)
        self.classification(model, device, inputClassifier)
        return

    def classification(self, model, device, inputClassifier):
        for iter, (batch_graphs, batch_labels) in enumerate(inputClassifier):
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            scores = model.forward(batch_graphs, batch_x, batch_e)
        indice = scores.detach().argmax(dim=1)
        self.word = self.labels[indice]
        self.top3 = scores.topk(3)[1].squeeze().cpu()
        self.prob = softmax(scores.detach().cpu().numpy())


    def initGCN(self, object2class):
        params, _ = init_params(object2class, 9)
        model = GatedGCNNet(params)
        model.load_state_dict(torch.load("GatedGCN/modelSaved.model"))
        device = params["device"]
        model.to(device)
        model.eval()
        return model, device

    def loadData(self):
        dict_to_json(self.json, 1, 1)
        object2class = STgraph(glob.glob("*.json"))
        inputClassifier = load(object2class)
        return inputClassifier, object2class

    def setProgressVal(self, val):
        self.pbar.setValue(val)
    
    def getWord(self):
        return self.word

    def getTop3(self):
        text = "Scores des 3 meilleurs mots :\n"
        for indice in self.top3 :
            text += "   - " + self.labels[indice] + " : {:.2f} % \n".format(100 * self.prob[0,indice])
        return text
    
    def stop(self):
        self.exit(0)

    def datum2json(self, cnt, datum):
        self.json.append({"frame" : cnt, "keypoints" : keypoints_to_json(datum)})