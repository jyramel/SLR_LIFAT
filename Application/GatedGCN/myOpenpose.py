# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:34:14 2020

@author: nathan
"""

import cv2
import time
import numpy as np
import os
#import pyopenpose as op
from matplotlib import pyplot as plt
from tqdm import tqdm

def set_params(face = False, body = True, hand = True, hand_detector = 0, 
               hand_opti = False, write_json = "", nb_people_max = 1, num_gpu = 0,
               model_folder = "openpose/models"):
    """
    face : bool, calcul des keypoints du visage
    body : bool, calcul des keypoints du corps
    hand : bool, calcul des keypoints des mains
    hand_detector : int, Select 0 to select OpenPose body detector (most accurate
        one and fastest one if body is enabled), 2 to indicate that it will be 
        provided by the user, 3 to also apply hand tracking (only for hand)
    hand_opti : bool, paramètres optimaux pour l'efficacité, perte de rapidité
    write_json : str, endroit où enregistrer fichiers JSON des keypoints, "" 
        pour ne pas enregistrer
    nb_people_max : int, nombre de personne max à chercher sur la frame/photo
    num_gpu : int, si l'on souhaite travailler manuellement sur les gpu, 0
        pour ne pas s'en préoccuper
    model_folder : str, chemin ou se situe les models d'openpose
    """
    params = dict()
    params["model_folder"] = model_folder
    
    # Noyau plante si face = True
    params["face"] = face 
    params["hand"] = hand
    params["hand_detector"] = hand_detector
    params["body"] = body
    
    # On fixe le nombre de personne
    params["number_people_max"] = nb_people_max
    
    # Best results found with 6 scales
    # --hand_scale_number 6 --hand_scale_range 0.4 
    
    if hand_opti :
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
    
    # Save results in JSON format 
    if write_json :
        params["write_json"] = write_json

    # Pour calcul multi-GPU : 
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_python/05_keypoints_from_images_multi_gpu.py
    
    return params


def init_openpose(params):
    """Initialise openpose et retourne le opWrapper"""
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

def opImages_2_json(imagePaths, params):
    """Applique openpose selon les paramètres params sur les images de la liste
    imagePaths. Le résultat est en format JSON au path params["write_json"] """
    
    if params["write_json"] == "":
        raise ValueError("Le chemin d'écriture ne peut pas être vide. Définissez vos paramètres avec un chemin d'écriture pour write_json")
        
    opWrapper = init_openpose(params)
    start = time.time()
        
    for imageId in tqdm(range(len(imagePaths))):
    
        imagePath = imagePaths[imageId]
        datum = op.Datum()
        image = cv2.imread(imagePath)
        datum.cvInputData = image
        if params["hand_detector"] == 2 :
            imageSize = max(image.shape[0:2])
            handRectangles = [[op.Rectangle(0., 0., imageSize, imageSize), op.Rectangle(0., 0., imageSize, imageSize),]]
            datum.handRectangles = handRectangles
        opWrapper.emplaceAndPop([datum])
        
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    
def opImage_display(imagePath, params):
    """Applique openpose sur une image et retourne l'object datum contenant :
        - l'image avec le squelette
        - les keypoints en numpy
    """
    opWrapper = init_openpose(params)
    datum = op.Datum()
    image = cv2.imread(imagePath)
    if params["hand_detector"] == 2 :
        imageSize = max(image.shape[0:2])
        handRectangles = [[op.Rectangle(0., 0., imageSize, imageSize), op.Rectangle(0., 0., imageSize, imageSize),]]
        datum.handRectangles = handRectangles
    opWrapper.emplaceAndPop([datum])
    return datum