import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import helper

# settings and parameters
MATCH_ID = 901
VIDEO = './m-%03d.mp4'%MATCH_ID #File path to match video
START_MS = 4032000
END_MS = 4040000
FPS = 25
PER_FRAME = 1000/FPS  # 40ms per frame
MIN_DETECTOR_SCORE = 0.25
COLOR_HIST_NPZ = "./M-%d-player-npz.npz"%(MATCH_ID) #File path to NPZ file
DEBUG = True

def colorHist(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [16,16,16], [1, 256, 1, 256, 1, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

player_imgs = [[],[],[],[],[]]  # 0:team1_player, 1:team1_gk, 2:team2_player, 3:team2_gk, 4:referee
team_color_hist = np.zeros(shape=(5, 4096), dtype=np.float32)
if os.path.exists(COLOR_HIST_NPZ):
    data = np.load(COLOR_HIST_NPZ, allow_pickle=True) #Loading external numpy array/pickled values
    for i in range(5):
        if len(data.files)>i:
            player_imgs[i] = [img for img in data['arr_%d'%(i)]] # Possible to have multiple pictures for each class (0,1,2,3,4) 2 each etc
            for image in player_imgs[i]:
                print(image.shape)
                print(image[0])
            color_hist = []
            for j in range(len(player_imgs[i])):
                # cv2.imshow('player_%d_%d'%(i,j), player_imgs[i][j])
                color_hist.append(colorHist(player_imgs[i][j])) #Append color histogram for all players
        team_color_hist[i] = np.mean(np.array(color_hist).reshape(-1, 4096), axis=0).astype(np.float32) #Average of x color histogram from each team
