# -*- coding: utf-8 -*-
"""
LIFAT : SLR
"""
import os
import json
import tqdm
import cv2

try:
    from myOpenpose import *    
except:
    sys.path.append('/home/nmiguens/Python')
    from myOpenpose import *

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
    path = "/home/nmiguens/JSON/WLASL"
    name = "{}_".format(id_gloss) + "{}.json".format(id_instance)
    with open(os.path.join(path, name), 'w') as fout:
        json.dump(videoJson , fout)
    return 0

def WLASL_parcours():
    """Parcours le json fourni par WLASL"""
    path = "/home/nmiguens/Datasets/WLASL"
    write_txt("\n" + "---------------------------- \n" + "Nouveau processus \n")
    nb_video = 0
    with open(r"{}".format(path + "/start_kit/WLASL_v0.3.json"), "r") as read_file:
        WLASL = json.load(read_file)
    for glosses in WLASL:
        for instance in glosses["instances"]:
            inputVideo = os.path.join(path, "videos/" + instance["video_id"] +".mp4")
            if os.path.exists(inputVideo):
                if not os.path.exists("/home/nmiguens/JSON/WLASL/{}_{}.json".format(glosses["gloss"], instance["instance_id"])):
                    videoDict = video_to_dict(inputVideo, 0, -1) #instance["frame_start"], instance["frame_end"])
                    dict_to_json(videoDict, glosses["gloss"], instance["instance_id"])
        nb_video = len(os.listdir("/home/nmiguens/JSON/WLASL")) - nb_video         
        message = "{} vidéos traitées pour la classe {}".format(nb_video, glosses["gloss"])
        write_txt(message)
    return 0

def write_txt(message):
    f = open('/home/nmiguens/Datasets/WLASL/count.txt','a')
    f.write('\n' + message)
    f.close()
    
def json_to_dict(jsonPath):
    """Transform un fichier json de keypoints en dictionnaire de numpy array"""
    with open(r"{}".format(jsonPath), "r") as read_file:
        data = json.load(read_file)
    return data

if __name__ == "__main__":
    WLASL_parcours()