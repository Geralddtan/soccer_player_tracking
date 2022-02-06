import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import helper

# settings and parameters
MATCH_ID = 904
VIDEO = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/Deep Sort/detectron2-deepsort-pytorch/original_vids/m-%03d.mp4'%MATCH_ID
START_MS = 6537000   # possible sequence [58400, 67040], [67840, 72840], [75000, 82400], [91680, 106880], [108360, 118440], [120760,136040]
END_MS = 6547000
FPS = 30
PER_FRAME = 1000/FPS  # 40ms per frame
MIN_DETECTOR_SCORE = 0.25
COLOR_HIST_NPZ = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Ground Truth Player Tracking Data/M-%d-GroundTruth/soccer-player-npz/M-%d-player-npz.npz"%(MATCH_ID, MATCH_ID)
CSV_FILES = "Colorhist_optimized_intersect_csv"
OUT_CSV = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/Tracking Implementation/ALL_CSV/%s/player_detection_colorhist/m-%03d-player-dt25-team-%d-%d.csv' % (CSV_FILES,MATCH_ID, START_MS, END_MS)
DEBUG = True

# Pre-trained object detector
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MIN_DETECTOR_SCORE # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
# Alternatively, (1) download weights from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# (2) load weights locally
cfg.MODEL.DEVICE='cpu'   # cuda or cpu
detector = DefaultPredictor(cfg)

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
            color_hist = []
            for j in range(len(player_imgs[i])):
                # cv2.imshow('player_%d_%d'%(i,j), player_imgs[i][j])
                color_hist.append(colorHist(player_imgs[i][j])) #Append color histogram for all players
        team_color_hist[i] = np.mean(np.array(color_hist).reshape(-1, 4096), axis=0).astype(np.float32) #Average of x color histogram from each team

    # color histogram distance sanity check
    for i in range(5):
        for j in range(len(player_imgs[i])):
            color_hist = colorHist(player_imgs[i][j])
            distances = []
            for k in range(5):
                distances.append(cv2.compareHist(color_hist, team_color_hist[k], cv2.HISTCMP_INTERSECT))  #Check against all classes
            player_class = np.argmax(distances) #Must assign to your team
            if i != player_class:
                print('color histogram mismatch', i, j, distances)
                exit(1)
else:
    print('no color histogram file')
    exit(1)

print('time_ms,x1,y1,x2,y2,score,t1_cs,t1gk_cs,t2_cs,t2gk_cs,ref_cs')
out = open(OUT_CSV, 'a+')
out.write('time_ms,x1,y1,x2,y2,score,t1_cs,t1gk_cs,t2_cs,t2gk_cs,ref_cs\n')
out.close()

cap = cv2.VideoCapture(VIDEO)
t = START_MS
cap.set(0, t)
while t <= END_MS:
    ret, frame = cap.read()

    # player detection
    output = detector(frame)
    player_boxes = output['instances'][output['instances'].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
    player_scores = output['instances'][output['instances'].pred_classes == 0].scores.cpu().numpy()
    player_masks = output['instances'][output['instances'].pred_classes == 0].pred_masks.cpu().numpy()

    if DEBUG:
        rounded_box = np.round(player_boxes).astype(np.int)
        [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (0,255,255), 1) for box in rounded_box]
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

    for i in range(len(player_scores)):
        # player_box_img = cv2.bitwise_and(frame, frame, mask=player_masks[i].astype(np.uint8) * 255)
        player_box_img = helper.return_diff_output_images(frame, player_boxes[i], player_masks[i])
        color_hist = colorHist(player_box_img['masked_image_halved'])
        team_cs = [cv2.compareHist(color_hist, team_color_hist[k], cv2.HISTCMP_INTERSECT) for k in range(5)]
        print('%d,%0.1f,%0.1f,%0.1f,%0.1f,%0.5f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f'%(
            t, player_boxes[i,0], player_boxes[i,1], player_boxes[i,2], player_boxes[i,3], player_scores[i], team_cs[0], team_cs[1], team_cs[2], team_cs[3], team_cs[4]))
        out = open(OUT_CSV, 'a+')
        out.write('%d,%0.1f,%0.1f,%0.1f,%0.1f,%0.5f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n'%(
            t, player_boxes[i,0], player_boxes[i,1], player_boxes[i,2], player_boxes[i,3], player_scores[i], team_cs[0], team_cs[1], team_cs[2], team_cs[3], team_cs[4]))
        out.close()

    t += PER_FRAME