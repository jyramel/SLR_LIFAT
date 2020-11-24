import cv2
from GatedGCN.builder import STgraph
from GatedGCN.loader import load
from GatedGCN.utils import init_params
from GatedGCN.gated_gcn_net import GatedGCNNet
import os
import json
import glob
import torch
from scipy.special import softmax

labels = ["AVANT", "LIVRE", "BONBON", "CHAISE", "VETEMENTS", "ORDINATEUR", "BOIRE", "ALLEZ (go)", "QUI ?"]

def classification_function(window, progress_bar, progress_text, video_image, list_frame_result, keypoints):
    total_images = len(list_frame_result)
    cnt = 0
    json_data = []
    for x in range(total_images):
        frame = list_frame_result[x]
        keypoint = keypoints[x]
        event, values = window.read(timeout=20)
        if event in ('Exit', None):
            exit(0)
        if event == 'reset':
            states.reset = True
            return
        imgbytes=cv2.imencode('.png', frame)[1].tobytes()
        video_image.update(data=imgbytes)
        datum2json(cnt, keypoint, json_data)
        cnt += 1
        percent = (cnt/total_images)*100
        progress_bar.update_bar(percent)
        progress_text.update('sign classification {:3.1f}%'.format(percent))
	
    inputClassifier, object2class = loadData(json_data)
    model, device = initGCN(object2class)
    word, top3, prob = classification(model, device, inputClassifier)
    return word, top3, prob

def classification(model, device, inputClassifier):
	for iter, (batch_graphs, batch_labels) in enumerate(inputClassifier):
		batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
		batch_e = batch_graphs.edata['feat'].to(device)
		scores = model.forward(batch_graphs, batch_x, batch_e)
	indice = scores.detach().argmax(dim=1)
	word = labels[indice]
	top3 = scores.topk(3)[1].squeeze().cpu()
	prob = softmax(scores.detach().cpu().numpy())
	return word, top3, prob

def initGCN(object2class):
    params, _ = init_params(object2class, 9)
    model = GatedGCNNet(params)
    model.load_state_dict(torch.load("GatedGCN/modelSaved.model"))
    device = params["device"]
    model.to(device)
    model.eval()
    return model, device

def loadData(json_data):
	filejsonname = dict_to_json(json_data, 1, 1)
	jsonlist = glob.glob(filejsonname)
	object2class = STgraph(jsonlist)
	inputClassifier = load(object2class)
	return inputClassifier, object2class

def dict_to_json(videoJson, id_gloss, id_instance):
    """Ecrit en format JSON le contenu du dictionnaire videoJson sous le nom {id_gloss}_{id_instance}.json"""
    path = ""
    name = "{}_".format(id_gloss) + "{}.json".format(id_instance)
    filejsonname = os.path.join(path, name)
    fout = open(filejsonname, 'w')
    json.dump(videoJson , fout)
    return filejsonname

def datum2json(cnt, datum, json_data):
    json_data.append({"frame" : cnt, "keypoints" : keypoints_to_json(datum)})
	
def keypoints_to_json(datum):
    """Enregistre les keypoints obtenues avec Openpose sous une architecture JSON"""
    jsonDict = dict()
    jsonDict["pose_keypoints_2d"] = datum.poseKeypoints.tolist()
    if datum.faceKeypoints == None: #datum.faceKeypoints.size > 0 :
        jsonDict["face_keypoints_2d"] = []
    else : 
        jsonDict["face_keypoints_2d"] = datum.faceKeypoints.tolist()
    jsonDict["hand_left_keypoints_2d"] = datum.handKeypoints[0].tolist()
    jsonDict["hand_right_keypoints_2d"] = datum.handKeypoints[1].tolist()
    return jsonDict
