# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:16:02 2020

@author: nathan
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def json_2_numpy(path_jsonFile, body = True, hand = True, face = False):
    """Transforme un fichier Json output d'Openpose en vecteur numpy"""
    data = pd.read_json(path_jsonFile)
    data = data["people"][0]
    
    if body :
        npBody = np.array(data["pose_keypoints_2d"])
    else : 
        npBody = np.array([])
        
    if hand :
        npHands = np.concatenate((np.array(data["hand_left_keypoints_2d"]), np.array(data["hand_right_keypoints_2d"])), axis=None)
    else : 
        npHands = np.array([])
    
    if face :
        npFace = np.array(data["face_keypoints_2d"])
    else :
        npFace = np.array([])
           
    full_body = np.concatenate((npBody, npHands), axis=None)
    return np.concatenate((full_body, npFace), axis=None)


def jsonDir_2_matrixNumpy(path_jsonFiles, body = True, hand = True, face = False):
    """Transforme le répertoire des keypoints en Json produits par Openpose en matrice"""
    matrix = []
    for id_json in tqdm(range(len(os.listdir(path_jsonFiles)))):
        
        path_json = os.path.join(path_jsonFiles, "{}_keypoints.json".format(id_json))
        matrix.append(json_2_numpy(path_json, body, hand, face))
    return np.array(matrix)