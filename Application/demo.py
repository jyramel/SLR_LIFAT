import PySimpleGUI as sg
import cv2
import time
import sys
import os

sys.path.append('C:/openpose-master/build/python/openpose/Release')
os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose-master/build/x64/Release;' + 'C:/openpose-master/build/bin;'


from openPose import openpose_function
from gcn import classification_function

def getTop3(top3, prob):
	text = "Scores des 3 meilleurs mots :\n"
	for indice in top3 :
		text += "   - " + labels[indice] + " : {:.2f} % \n".format(100 * prob[0,indice])
	return text

def time_as_int():
    return int(round(time.time() * 100))

class StateFlags():
    def __init__(self):
        self.start_processing = False
        self.progress_processing = False
        self.start_cap = False
        self.start_classification = False
        self.results = False
        self.reset = False
		
#constants
sec_rec = 3
labels = ["AVANT", "LIVRE", "BONBON", "CHAISE", "VETEMENTS", "ORDINATEUR", "BOIRE", "ALLER (go)", "QUI ?"]

sg.theme('SystemDefault')



logo1 = cv2.resize(cv2.imread('Images/lifat.png'),(200,100))
logo2 = cv2.resize(cv2.imread('Images/UT.png'),(200,120))
imgbytes1= cv2.imencode('.png', logo1)[1].tobytes()
imgbytes2= cv2.imencode('.png', logo2)[1].tobytes()


# define the window layout
progress_bar = sg.ProgressBar(max_value=100, orientation='h', size=(100, 20), key='progress')
progress_text = sg.Text('', size=(30, 1), justification='center', font=("Colibri", 10))
video_image = sg.Image(filename='', key='-IMAGE-')
result_text = sg.Text('', size=(30, 5), justification='center', font=("Colibri", 15))
layout = [[sg.Image(data=imgbytes1),
		   sg.Text('Bienvenue sur le traducteur\n\ndu langage des signes.', size=(30, 3), justification='center', font=("Colibri", 30)),
		   sg.Image(data=imgbytes2)], 
		  [sg.Text('', size=(30, 1), justification='center', font=("Colibri", 30))],
		  [progress_text, progress_bar], 
		  [result_text, video_image],
		  [sg.Button('Start', key='start', font=("Colibri", 15)),sg.Button('Reset', key='reset', font=("Colibri", 15))],]

# create the window and show it without the plot
window = sg.Window('Reconnaissance du Langage des Signes', 
					layout, 
					grab_anywhere=True, 
					element_justification='c', 
					location=(300,1),
					finalize=True)  
window.Maximize()

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap = cv2.VideoCapture(0)                               # Setup the OpenCV capture device (webcam)
list_frame_input = []
list_frame_result = []
states = StateFlags()
time_rec = 0
start_time = time_as_int()
keypoints = None
word = None
top3 = None
prob = None

while True:
    event, values = window.read(timeout=20)
    ret, frame = cap.read()                             # Read image from capture device (camera)
    if event in ('Exit', None):
        break
    if event == 'start':
        states.start_cap = True
        time_rec = 0
        start_time = time_as_int()
        window['start'].update(disabled=True)
    if event == 'reset':
        states.reset = True
    if states.reset == True:
        word = None
        top3 = None
        prob = None
        states.reset = False
        list_frame_input = []
        list_frame_result = []
        window['start'].update(disabled=False)
        states.start_processing = False
        time_rec = 0
        states.results=False
        states.start_cap = False
        states.start_processing = False
        states.start_classification = False
        progress_bar.update_bar(0)
        progress_text.update('')
        result_text.update('')
        keypoints = None
    if states.start_cap:
        list_frame_input.append(frame)
        time_rec = time_as_int() - start_time
        time_sec = ((time_rec // 100) % 60) + 1
        percent_time = (time_sec*100)/sec_rec
        progress_bar.update_bar(percent_time)
        if (time_rec // 100) % 60 < sec_rec:
            progress_text.update(str(time_sec) + ' sec')
    if (time_rec // 100) % 60 >= sec_rec:
        states.start_processing = True
        time_rec = 0
        states.start_cap = False
        progress_bar.update_bar(0)
        progress_text.update('')
    if states.start_processing == True:
        keypoints = openpose_function(window,progress_bar, progress_text, video_image, states, list_frame_input, list_frame_result)
        if states.reset == True:
            continue
        states.start_classification = True
        states.start_processing = False
    if states.start_classification == True:
        word, top3, prob = classification_function(window, progress_bar, progress_text, video_image, list_frame_result, keypoints)
        states.start_classification = False
        states.results=True
    if states.results:
        progress_bar.update_bar(0)
        progress_text.update('')
        #result_function(window,progress_bar, progress_text, video_image, states, list_frame_input, list_frame_result)	
        result_text.update(getTop3(top3,prob))
        event, values = window.read(timeout=20)
        total_images = len(list_frame_result)
        cnt = 0
        while event != 'reset':
            event, values = window.read(timeout=20)
            if event in ('Exit', None):
                exit(0)
            if cnt == total_images:
                cnt = 0
            else:
                frame = list_frame_result[cnt]
                imgbytes=cv2.imencode('.png', frame)[1].tobytes()
                video_image.update(data=imgbytes)
                cnt += 1
        states.reset = True
    imgbytes=cv2.imencode('.png', frame)[1].tobytes()   # Convert the image to PNG Bytes
    video_image.update(data=imgbytes)   # Change the Image Element to show the new image
		
window.close()