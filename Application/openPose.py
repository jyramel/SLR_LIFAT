# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:46:08 2020

@author: natha
"""



import pyopenpose as op
from PyQt5.QtWidgets import QProgressBar
import cv2
from PyQt5.QtCore import pyqtSignal, Qt, QThread


class OpenPoseThread(QThread):
    change_value = pyqtSignal(int)

    def __init__(self, filename):
        QThread.__init__(self)
        self.pbar = QProgressBar()
        self.pbar.setGeometry(0, 0, 300, 50)
        self.pbar.setAlignment(Qt.AlignCenter)

        self.filename = filename
        self.cap = cv2.VideoCapture(self.filename)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if length:
            self.pbar.setMaximum(int(length))
        else:
            self.pbar.setMaximum(int(fps * 4))

        self.opWrapper = initOpenpose(set_params(tracking=0))
        self.keypoints = []
        self.frames = []
        self.writer = None

    def run(self):
        skipParameter = 2
        sizeReduction = 2
        cnt = 0
        ret, cv_img = self.cap.read()
        if sizeReduction!=1:
            cv_img = cv2.resize(cv_img,(int(cv_img.shape[1]/sizeReduction),int(cv_img.shape[0]/sizeReduction)))
		
        while ret :
            if cnt%skipParameter==0:
               self.openPoseTraitement(cv_img)
               self.change_value.emit(cnt)
            cnt += 1
            ret, cv_img = self.cap.read()
            if ret == True:
               if sizeReduction!=1:
                   cv_img = cv2.resize(cv_img,(int(cv_img.shape[1]/sizeReduction),int(cv_img.shape[0]/sizeReduction)))
			
        self.saveFrames()
        return

    def openPoseTraitement(self, frame):
        # Create writer if not exist
        if not self.writer:
            H, W = frame.shape[0:2]
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter("test2.avi", fourcc, fps,
                                      (int(W), int(H)), True)
        # save keypoints and modified frame
        datum = self.opFrameRun(frame)
        self.keypoints.append(datum)
        self.frames.append(datum.cvOutputData)

    def getKeypoints(self):
        return self.keypoints

    def saveFrames(self):
        for frame in self.frames :
            self.writer.write(frame)
        self.writer.release()

    def setProgressVal(self, val):
        self.pbar.setValue(val)

    def stop(self):
        self.exit(0)

    def opFrameRun(self, frame):
        """Applique openpose sur une image et retourne l'object datum contenant :
            - l'image avec le squelette
            - les keypoints en numpy
        """
        datum = op.Datum()
        datum.cvInputData = frame
        self.opWrapper.emplaceAndPop([datum])
        return datum


def set_params(face=False, body=1, hand=True, hand_detector=0,
               hand_opti=False, nb_people_max=1, model_folder="openpose/models", tracking=-1):
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


