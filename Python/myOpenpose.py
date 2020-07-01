# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:34:14 2020

@author: nathan
"""

import pyopenpose as op
import cv2
import time
import numpy as np
import os
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm

def set_params(face = False, body = 1, hand = True, hand_detector = 0, 
               hand_opti = False, write_json = "", nb_people_max = 1, num_gpu = 0,
               model_folder = '/home/conte/openpose/models', tracking = -1):
    """
    face : bool, calcul des keypoints du visage
    body : int, calcul des keypoints du corps
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
        #params["hand_scale_range"] = float(0.4)
    
    # Save results in JSON format 
    if write_json :
        params["write_json"] = write_json
    
    if num_gpu :
        params["num_gpu"] = int(op.get_gpu_number())
    
    params["tracking"] = tracking
    # Pour calcul multi-GPU : 
    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/tutorial_api_python/05_keypoints_from_images_multi_gpu.py
    
    return params

def opVideo(videopath, outputpath, params):
    """ Ecrit une vidéo au chemin outputpath (format .avi nécessaire) à partir de la vidéo videopath"""
    opWrapper = init_openpose(params)

    vs = cv2.VideoCapture(videopath)
    (ret, frame) = vs.read()
    print((int(frame.shape[1]), int(frame.shape[0])))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps, length = int(vs.get(cv2.CAP_PROP_FPS)), int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None 

    for i in tqdm(range(length)):
        (ret, frame) = vs.read()
        if not ret:
            break
        datum = opFrame_run(frame, params, opWrapper)
        frame = datum.cvOutputData
        try :
            scoreL = np.mean(datum.handKeypoints[0][:,:,2])
            scoreR = np.mean(datum.handKeypoints[1][:,:,2])
        except:
            scoreL = 0
            scoreR = 0
        text = "LH : {0:.2f}".format(scoreL) + " ## RH : {0:.2f}".format(scoreR)
        frame = cv2.putText(frame, text, (5, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        if writer == None :
            writer = cv2.VideoWriter(outputpath, fourcc, fps, (int(frame.shape[1]), int(frame.shape[0])))
        writer.write(frame)

    writer.release()
    vs.release()

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
    
def opImage_display(imagePath, params, display = True, frame = None):
    """Applique openpose sur une image et retourne l'object datum contenant :
        - l'image avec le squelette
        - les keypoints en numpy
    """
    opWrapper = init_openpose(params)    
    datum = op.Datum()
    if frame:
        image = frame
    else : 
        image = cv2.imread(imagePath)
    datum.cvInputData = image
    if params["hand_detector"] == 2 :
        imageSize = float(max(image.shape[0:2]))
        handRectangles = [[op.Rectangle(0., 0., imageSize, imageSize), op.Rectangle(0., 0., imageSize, imageSize),]]
        datum.handRectangles = handRectangles
    opWrapper.emplaceAndPop([datum])
    if display :
        plt.imshow(datum.cvOutputData)
        plt.show()
    return datum

def opFrame_run(frame, params, opWrapper):
    """Applique openpose sur une frame et retourne l'object datum contenant :
        - l'image avec le squelette
        - les keypoints en numpy
    """    
    datum = op.Datum()
  
    datum.cvInputData = frame
    if params["hand_detector"] == 2 :
        imageSize = float(max(frame.shape[0:2]))
        handRectangles = [[op.Rectangle(0., 0., imageSize, imageSize), op.Rectangle(0., 0., imageSize, imageSize),]]
        datum.handRectangles = handRectangles
    opWrapper.emplaceAndPop([datum])
    return datum

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

def transform_Path2Class(imagePaths):
    """Retourne le tableau de class des images présentes dans le tableau de chemin _imagePaths_"""
    y = []
    for path in imagePaths :
        image_name = path.split("/")[-1]
        image_class = image_name.split("_")[0]
        y.append(int(image_class))
    return np.array(y)

def get_accuracy(y_test, y):
    return np.sum(y == y_test)/y.shape[0]