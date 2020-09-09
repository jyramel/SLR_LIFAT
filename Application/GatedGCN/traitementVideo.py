# -*- coding: utf-8 -*-
"""
LIFAT : SLR
"""
import os
import json
import tqdm
import cv2
from GatedGCN.myOpenpose import set_params, init_openpose
#import pyopenpose as op

def frame_to_keypoints(frame, opWrapper):
    """Transforme une frame d'une vidéo en keypoints avec Openpose"""
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    return datum

def keypoints_to_json(datum):
    """Enregistre les keypoints obtenues avec Openpose sous une architecture JSON"""
    jsonDict = dict()
    jsonDict["pose_keypoints_2d"] = datum.poseKeypoints.tolist()
    if datum.faceKeypoints.size > 0 :
        jsonDict["face_keypoints_2d"] = []
    else : 
        jsonDict["face_keypoints_2d"] = datum.faceKeypoints.tolist()
    jsonDict["hand_left_keypoints_2d"] = datum.handKeypoints[0].tolist()
    jsonDict["hand_right_keypoints_2d"] = datum.handKeypoints[1].tolist()
    return jsonDict

def video_to_dict(inputVideo, start, end):
    """Transforme une video en dictionnaire "JSON" de ses keypoints"""
    params = set_params(tracking = 0, hand_opti = True)
    opWrapper = init_openpose(params)
    videoJson = []
    vs = cv2.VideoCapture(inputVideo)
    vs.set(cv2.CAP_PROP_FPS, 25)
    id_frame = 1
    while(1):
        (ret, frame) = vs.read()
        if (not ret) or (end != -1 and id_frame > end):
            break
        if id_frame >= start :
            datum = frame_to_keypoints(frame, opWrapper)
            videoJson.append({"frame" : id_frame - start, "keypoints" : keypoints_to_json(datum)})
        id_frame += 1
    return videoJson

def dict_to_json(videoJson, id_gloss, id_instance):
    """Ecrit en format JSON le contenu du dictionnaire videoJson sous le nom {id_gloss}_{id_instance}.json"""
    path = ""
    name = "{}_".format(id_gloss) + "{}.json".format(id_instance)
    with open(os.path.join(path, name), 'w') as fout:
        json.dump(videoJson , fout)
    return 0

def json_to_dict(jsonPath):
    """Transform un fichier json de keypoints en dictionnaire de numpy array"""
    with open(r"{}".format(jsonPath), "r") as read_file:
        data = json.load(read_file)
    return data
