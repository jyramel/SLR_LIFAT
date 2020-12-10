# -*- coding: utf-8 -*-
"""

@author: conte
"""
import cv2
import pyopenpose as op


def openpose_function(window, progress_bar, progress_text, video_image, states, list_frame_input, list_frame_output):
    opWrapper = initOpenpose(set_params(tracking=0))
    keypoints = []
    list_frame_output.clear()
    total_images = len(list_frame_input)
    skipParameter = 3
    sizeReduction = 2
    cnt = 0
    datum = None
    for frame in list_frame_input:
        event, values = window.read(timeout=20)
        if event in ('Exit', None):
            exit(0)
        if event == 'reset':
            states.reset = True
            return
        imgbytes=cv2.imencode('.png', frame)[1].tobytes()
        video_image.update(data=imgbytes)
        if sizeReduction!=1:
            cv_img = cv2.resize(frame,(int(frame.shape[1]/sizeReduction),int(frame.shape[0]/sizeReduction)))
        if cnt%skipParameter==0:
            datum = opFrameRun(cv_img, opWrapper)
            keypoints.append(datum)
            if sizeReduction!=1:
                fr_save = cv2.resize(datum.cvOutputData, (frame.shape[1],frame.shape[0]))
                list_frame_output.append(fr_save)
            else:
                list_frame_output.append(datum.cvOutputData)
        cnt += 1
        percent = (cnt/total_images)*100
        progress_bar.update_bar(percent)
        progress_text.update('skeleton processing {:3.1f}%'.format(percent))
    return keypoints

def opFrameRun(frame, opWrapper):
    """Applique openpose sur une image et retourne l'object datum contenant :
    	- l'image avec le squelette
    	- les keypoints en numpy
    """
    datum = op.Datum()
    datum.cvInputData = frame
    #old version of pyopenpose library
    #opWrapper.emplaceAndPop([datum])
	
    #new version of pyopenpose library
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum


def set_params(face=False, body=1, hand=True, hand_detector=0, hand_opti=False, 
			   nb_people_max=1, model_folder="openpose/models", tracking=-1):
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
    tracking : int, 0 pour mode tracking
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

    if hand_opti:
        params["hand_scale_number"] = 6
        params["hand_scale_range"] = 0.4
    params["tracking"] = tracking
    return params


def initOpenpose(params):
    """Initialise openpose et retourne le opWrapper"""
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper


