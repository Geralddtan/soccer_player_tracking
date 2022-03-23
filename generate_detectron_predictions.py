import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import helper

# settings and parameters
MATCH_ID = 901
START_MS = 4032000  # 61680   #91680
END_MS = 4040000 
VIDEO = './m-%03d.mp4' % MATCH_ID #File Path to Match Video
FPS = 25
PER_FRAME = 1000/FPS  # 40ms per frame
MIN_DETECTOR_SCORE = 0.25
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

cap = cv2.VideoCapture(VIDEO)
t = 4035720.0 #START_MS
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
        print(rounded_box)
        # [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (255,255,255), 2) for box in rounded_box if (box[0] > 750) & (box[0] < 800) & (box[1] > 100) & (box[1] < 250)]
        for ID, box in enumerate(rounded_box):
            if ID in [23]:
                cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (0,255,255), 2)
        # cv2.rectangle(frame, tuple([745, 385]), tuple([770, 455]), (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        print(t) 

    t += PER_FRAME