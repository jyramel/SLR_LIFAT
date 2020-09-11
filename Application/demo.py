# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 11:49:05 2020

@author: natha
"""
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QWidget, QApplication, QLabel, QVBoxLayout, QProgressBar,
                             QHBoxLayout, QPushButton, QGroupBox, QFileDialog, QDesktopWidget)
from PyQt5.QtGui import QPixmap, QFont
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import time
import os

sys.path.append('C:/openpose-master/build/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose-master/build/x64/Release;' + 'C:/openpose-master/build/bin;'

from openPose import OpenPoseThread
from gcn import GCNThread
        
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self._save_flag = False
        self.writer = None
        self.video = []
        
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                
            if self._save_flag :
                if not self.writer : 
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    H, W = cv_img.shape[0:2]
                    self.writer = cv2.VideoWriter("test.avi",fourcc, fps, 
                                                  (int(W), int(H)), True)
                self.video.append(cv_img)
                #self.writer.write(cv_img)         
        # shut down capture system
        cap.release()
        
    def capture_running(self):
        cap = cv2.VideoCapture(0)
        return cap.read()[0]
    
    def save(self):
        for frame in self.video :
            self.writer.write(frame) 
        #self.writer.release()
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.exit(0)

class VideoReaderThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._run_flag = True
        
        self.cap = cv2.VideoCapture(self.filename)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
    def run(self):
        # capture from web cam
        while self._run_flag:
            self.msleep(int(1000.0//self.fps))
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            else : 
                self.cap = cv2.VideoCapture(self.filename)
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.exit(0)

class BarProgressTimer(QThread):
    change_value = pyqtSignal(int)
    
    def __init__(self, limitTime, color):
        QThread.__init__(self)
        self.pbar = QProgressBar()
        self.pbar.setStyleSheet("QProgressBar::chunk {background:" + color + "}")
        self.pbar.setGeometry(0, 0, 300, 30)
        self.pbar.setMaximum(limitTime * 100)
        self.pbar.setAlignment(Qt.AlignCenter) 
        self.limitTime = limitTime
        self.car = [i for i in range(limitTime, -1, -1)]
        self.lim = 0
    
    def run(self):
        cnt = 0
        start = time.time()
        while cnt < self.limitTime * 100 :
            self.pbar.setFormat(str(self.car[cnt//100]))
            cnt = int(100 * (time.time() - start))
	    self.change_value.emit(cnt)
        self.pbar.setFormat("0") 
        if self.lim == 1 :
            return
 
    def setProgressVal(self, val):
        self.pbar.setValue(val)

    def stop(self):
        self.exit(0)
        
        
    def reset(self, limitTime, color):
        if self.lim < 1 :
            self.pbar.setStyleSheet("QProgressBar::chunk {background:" + color + "}")
            self.pbar.setGeometry(0, 0, 300, 50)
            self.pbar.setMaximum(limitTime * 100)
            self.pbar.setValue(0)
            self.limitTime = limitTime
            self.car = [i for i in range(limitTime, 0, -1)]
            self.lim += 1
           
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reconnaissance du Langage des Signes")
        self.setWindowIcon(QtGui.QIcon('Image/UT.png'))

        size = QDesktopWidget().screenGeometry(-1)
        self.h, self.w = size.height(), size.width()

        self.createTopLabelGroup()
        self.createVideoGroup()
        self.createBottomGroup()
        
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLabelGroup)
        self.mainLayout.addLayout(self.videoGroup)
        self.mainLayout.addWidget(self.bottomGroup)
        
        self.setLayout(self.mainLayout)

        self.fileName = None
        self.bar, self.reader = None, None
        self.progress = None
        
    def createTopLabelGroup(self):
        """Zone d'information pour l'utilisateur"""
        self.topLabelGroup = QVBoxLayout()
        self.textLabel = QLabel("")
        self.undertext = QLabel("Bienvenue sur le traducteur" + "\n \n" + "du langage des signes.")
        self.undertext.setAlignment(Qt.AlignCenter)
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setFont(QFont('Colibri', 15, QFont.Bold))
        self.undertext.setFont(QFont('Colibri', 20, QFont.Bold))

        textGroup = QVBoxLayout()
        textGroup.addWidget(self.textLabel)
        textGroup.addWidget(self.undertext)

        imageUT = QLabel()
        imageLifat = QLabel()
        imageUT.setPixmap(QPixmap("Image/UT.png"))
        imageLifat.setPixmap(QPixmap("Image/lifat.png"))
        imageUT.setAlignment(Qt.AlignRight)
        imageLifat.setAlignment(Qt.AlignLeft)

        LabelGroup = QHBoxLayout()
        LabelGroup.addWidget(imageLifat)
        LabelGroup.addLayout(textGroup)
        LabelGroup.addWidget(imageUT)

        self.topLabelGroup.addLayout(LabelGroup)
    
    def createVideoGroup(self):
        """Zone où la vidéo est jouée"""
        self.disply_width = 2 * self.w // 3
        self.display_height = 2 * self.h // 3
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.textScore = QLabel("")
        self.textScore.setAlignment(Qt.AlignRight)
        self.textScore.setFont(QFont('Colibri', 12, QFont.Bold))


        self.videoGroup = QHBoxLayout()
        self.videoGroup.addWidget(QLabel(""))
        self.videoGroup.addWidget(self.image_label)
        self.videoGroup.addWidget(self.textScore)



        
    
    def createBottomGroup(self):
        """Zone des boutons de navigation"""
        self.bottomGroup = QGroupBox("Commencez par choisir une vidéo à traduire :")
        self.bottomGroup.setFont(QFont('Colibri', 13, QFont.Bold))
        
        boutonParours = QPushButton("Parcourir dans mes fichiers")
        boutonParours.clicked.connect(self.appui_boutonParours)
        
        boutonLecture = QPushButton("Utiliser la webcam")
        boutonLecture.clicked.connect(self.appui_boutonLecture)
        
        group = QHBoxLayout()
        group.addWidget(boutonParours)
        group.addWidget(boutonLecture)
        
        self.bottomGroup.setLayout(group)

    def appui_boutonParours(self):
        """Déclanche le choix d'une vidéo parmis les fichiers enregistrés"""
        self.undertext.setText("Choix de la vidéo à traduire")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "./","All Files (*);; Video Files (*.mov, *.mp4, *.avi) ", options=options)
        self.bottomGroup.deleteLater()
        if not self.fileName:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        self.lecture()
        self.validation()
        
    def lecture(self):
        """Déclanche la lecture de la vidéo du nom de fileName"""
        self.reader = VideoReaderThread(self.fileName)
        self.reader.change_pixmap_signal.connect(self.update_image)
        self.reader.start()
    
    def stop_lecture(self):
        """Stop la lecture de la vidéo en cours"""
        self.reader.stop()
        
    def appui_boutonLecture(self):
        """Lance un enregistrement de vidéo via la webcam en 2 temps :
        - Préparation
        - puis enregistrement"""
        self.undertext.setText("Préparation de la webcam :")

        # Initialisation
        self.bottomGroup.deleteLater()
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # Laisse le temps à la caméra de se lancer
        self.undertext.setText("Enregistrement dans :")
        time.sleep(2)

        # Préparation utilisateur
        self.progress = BarProgressTimer(3, "red")
        self.bar = self.progress.pbar
        self.topLabelGroup.addWidget(self.bar)
        self.progress.change_value.connect(self.progress.setProgressVal)
        self.progress.finished.connect(self.enregistrement)
        self.progress.start()
                
    def enregistrement(self):
        """Suite de appui_boutonLecture, lance l'enregistrement"""
        self.undertext.setText("Enregistrement en cours")
        self.thread._save_flag = True
        self.progress.reset(3, "green")
        self.progress.finished.disconnect(self.enregistrement)
        self.progress.finished.connect(self.verification)
        self.progress.start()

    def verification(self):
        """Fin du choix du fichier ou de l'enregistrement, lance l'étape de validation"""
        self.bar.deleteLater()
        self.progress.stop()

        self.thread._save_flag = False
        self.thread.save()
        self.thread.stop()
        
        self.fileName = "test.avi"
        self.lecture()
        
        #self.validation()
        self.newCreateBottomGroup()
        
    def validation(self):
        """Demande à l'utilisateur si la vidéo lui convient"""
        self.undertext.setText("Cette vidéo va être traduite :")

        self.newCreateBottomGroup()
        self.mainLayout.addWidget(self.newBottomGroup)
        self.setLayout(self.mainLayout)

        
    def newCreateBottomGroup(self):
        """Zone de bouton de confirmation"""
        self.newBottomGroup = QGroupBox()
        self.newBottomGroup.setFont(QFont('Colibri', 13, QFont.Bold))

        boutonParours = QPushButton("Oui")
        boutonParours.clicked.connect(self.appui_boutonOui)
        
        boutonLecture = QPushButton("Non")
        boutonLecture.clicked.connect(self.appui_boutonNon)
        
        group = QHBoxLayout()
        group.addWidget(boutonParours)
        group.addWidget(boutonLecture)
        
        self.newBottomGroup.setLayout(group)
        self.appui_boutonOui()
    
    def appui_boutonNon(self):
        """Redémarre l'application si l'utilisateur ne veut pas de cette vidéo"""
        self.stop_lecture()
        os.execv(sys.executable, [sys.executable] + sys.argv)
        
    def appui_boutonOui(self):
        """Lance le début de la classification"""
        self.newBottomGroup.deleteLater()
        self.classification1()        
        
    def classification1(self):
        """Etape 1 : Calcul du squelette via openpose"""
        self.textLabel.setText("Veuillez patienter s'il vous plaît,")
        self.undertext.setText("Détection du squelette en cours")
        self.progress = OpenPoseThread(self.fileName)
        self.bar = self.progress.pbar
        self.topLabelGroup.addWidget(self.bar)
        self.progress.change_value.connect(self.progress.setProgressVal)
        self.progress.finished.connect(self.classification2)
        self.progress.start()
    
    def classification2(self):
        """Etape 2 : Affichage du résultat d'openpose + classification avec GCN"""
        self.stop_lecture()
        self.fileName = "test2.avi"
        self.lecture()

        self.bar.deleteLater()
        keypoints = self.progress.getKeypoints()
        self.textLabel.setText("Veuillez patienter s'il vous plaît,")
        self.undertext.setText("Traduction en cours")
        self.progress.stop()
        self.progress = GCNThread(keypoints)
        self.bar = self.progress.pbar
        self.topLabelGroup.addWidget(self.bar)
        self.progress.change_value.connect(self.progress.setProgressVal)
        self.progress.finished.connect(self.getResults)
        self.progress.start()
    
    def getResults(self):
        """Récupération du résultat"""
        self.bar.deleteLater()
        word = self.progress.getWord()
        self.textLabel.setText("Le signe devrait correspondre au mot :")
        self.undertext.setText(word)

        top3 = self.progress.getTop3()
        self.textScore.setText(top3)
        self.progress.stop()
        self.finalFen()
        
    def finalFen(self):
        bouton = QPushButton("Recommencer")
        bouton.setFont(QFont('Colibri', 20))
        bouton.clicked.connect(self.appui_boutonNon)
        
        boutonClose = QPushButton("Sortie")
        boutonClose.setFont(QFont('Colibri', 20))
        boutonClose.clicked.connect(self.closeEvent)
		
        self.mainLayout.addWidget(bouton)
        self.mainLayout.addWidget(boutonClose)
        self.setLayout(self.mainLayout)
        
        
    def closeEvent(self, event):
        if self.progress :
            self.progress.exit(0)
        if self.reader :
            self.stop_lecture()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    #a.show()
    a.showMaximized()
    #a.showFullScreen()
    sys.exit(app.exec_())
