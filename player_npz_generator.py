import os
import numpy as np
import re
import cv2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import helper

MATCH_ID = 904
PLAYER_PICTURES_PATH = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Ground Truth Player Tracking Data/M-%d-GroundTruth/soccer-player-images/"%(MATCH_ID)
MASKED_PLAYERS_PATH = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Ground Truth Player Tracking Data/M-%d-GroundTruth/soccer-player-images/all_masked_players/"%(MATCH_ID)
MATCH_PICTURES_PATH = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Ground Truth Player Tracking Data/M-%d-GroundTruth/soccer-player-images/match_image/"%(MATCH_ID)
TEAM_CLASS = ['team-1/', 'team-1-gk/', 'team-2/', 'team-2-gk/', 'referee/']
OUT_CSV = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Ground Truth Player Tracking Data/M-%d-GroundTruth/soccer-player-npz/"%(MATCH_ID)
player_numpy_array = []

# Pre-trained object detector
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # Set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
# Alternatively, (1) download weights from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# (2) load weights locally
cfg.MODEL.DEVICE='cpu'   # cuda or cpu
detector = DefaultPredictor(cfg)

files  = [file for file in os.listdir(MATCH_PICTURES_PATH) if re.match('^.*.png', file)]
temp_array = []
counter = 0
for file in files:
    frame = cv2.imread(MATCH_PICTURES_PATH+file)
    # player detection
    output = detector(frame)
    player_boxes = output['instances'][output['instances'].pred_classes == 0].pred_boxes.tensor.cpu().numpy()
    player_scores = output['instances'][output['instances'].pred_classes == 0].scores.cpu().numpy()
    player_masks = output['instances'][output['instances'].pred_classes == 0].pred_masks.cpu().numpy()
    for i in range(len(player_masks)):
        player_bbox_mask = helper.return_diff_output_images(frame, player_boxes[i], player_masks[i])
        cv2.imwrite(MASKED_PLAYERS_PATH+str(counter)+".png", player_bbox_mask['masked_image_halved'])
        counter += 1

input("Please trasnfer necessary images into their respective class folders")

for classes in TEAM_CLASS:
    class_pictures_path = PLAYER_PICTURES_PATH+classes
    files  = [file for file in os.listdir(class_pictures_path) if re.match('^.*.png', file)]
    temp_array = []
    for file in files:
        _player = cv2.imread(class_pictures_path+file)
        temp_array.append(_player)
    player_numpy_array.append(temp_array)

np.savez(OUT_CSV+"M-%d-player-npz.npz"%(MATCH_ID), player_numpy_array[0], player_numpy_array[1], player_numpy_array[2], player_numpy_array[3], player_numpy_array[4])        


# npz_file = np.load(OUT_CSV+"M-%d-player-npz.npz"%(MATCH_ID), allow_pickle=True)
# print(npz_file['arr_0'])

