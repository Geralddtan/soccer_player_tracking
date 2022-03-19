import pandas as pd
import cv2
import helper_player_tracking
import numpy as np

MATCH_ID = 906
START_MS = 526000  # 61680   #91680
END_MS = 531960  # 67040   #106880
VIDEO = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/Deep Sort/detectron2-deepsort-pytorch/original_vids/m-%03d.mp4' % MATCH_ID
MOT_CHALLENGE_FOLDER = "soccer-player-test-with-team"
OUT_CSV_FOLDER = "with_reassignment/niv_tracks_with_reassignment_use_frames_last_det_inv_homog_impl_strict_niv_niv_removal_low_det_threshold"
OUT_CSV = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Tracking Implementation/Checking/TrackEval/data/trackers/mot_challenge/%s/%s/data/m-%03d.txt" % (MOT_CHALLENGE_FOLDER, OUT_CSV_FOLDER, MATCH_ID)
FPS = 25
PER_FRAME = 1000 / FPS

color_list = [[255, 255, 255], [0, 0, 255], [0, 255, 0],
        [255, 255, 0]]

court_width, court_length = 68.0, 106.3
court_points = np.array(
    [[-20.16, 0], [20.16, 0], [-20.16, 16.5], [20.16, 16.5], [-9.16, 0],
    [9.16, 0], [-9.16, 5.5], [9.16, 5.5], [-7.312, 16.5], [7.312, 16.5],
    [0, 11], [-3.66, 0], [3.66, 0], [0, 0], [-court_width / 2, 0],
    [court_width / 2, 0], [-court_width / 2, court_length / 2], [0, court_length / 2],
    [court_width / 2, court_length / 2], [-20.16, court_length],
    [20.16, court_length], [-20.16, court_length - 16.5], [20.16, court_length - 16.5], [-9.16, court_length],
    [9.16, court_length],
    [-9.16, court_length - 5.5], [9.16, court_length - 5.5], [-7.312, court_length - 16.5],
    [7.312, court_length - 16.5], [0, court_length - 11],
    [-3.66, court_length], [3.66, court_length], [0, court_length], [-court_width / 2, court_length],
    [court_width / 2, court_length],
    [-9.15, court_length / 2], [9.15, court_length / 2], [0, court_length / 2 - 9.15], [0, court_length / 2 + 9.15],
    [-court_width / 2, 16.5], [court_width / 2, 16.5], [-court_width / 2, court_length - 16.5],
    [court_width / 2, court_length - 16.5],
    [-court_width / 2, court_length / 2 + 9.15], [court_width / 2, court_length / 2 + 9.15],
    [-court_width / 2, court_length / 2 - 9.15], [court_width / 2, court_length / 2 - 9.15],
    [-court_width / 2, 5.5], [court_width / 2, 5.5], [-court_width / 2, court_length - 5.5],
    [court_width / 2, court_length - 5.5],
    [-court_width / 2, 11], [court_width / 2, 11], [-court_width / 2, court_length - 11],
    [court_width / 2, court_length - 11]])

std_img, std_court_points = helper_player_tracking.get_standard_court(court_points, line_thickness=3)

df = pd.read_csv(OUT_CSV, sep=",", header = None)
df.columns = ["frame", "id", "x1", "y1", "w", "h", "court_x", "court_y", "team"]
df[["x2", "y2"]] = df.apply(lambda x:[x[2] + x[4], x[3] + x[5]], axis = 1, result_type="expand")
print(df)

cap = cv2.VideoCapture(VIDEO)
t = START_MS
cap.set(0, t)
while t <= END_MS:
    ret, frame = cap.read()
    height, width = frame.shape[0:2]
    k = int(round((t - START_MS) / PER_FRAME)) + 1
    std_img_copy = std_img.copy()
    df_frame = df[(df["frame"] == k) & (df["team"] == 0)]
    df_frame = df_frame.astype("int")
    court_coords = np.array(df_frame[["court_x", "court_y", "team"]])
    [cv2.circle(std_img_copy, tuple((c[0], c[1])), 5, color_list[int(c[2])%len(color_list)], 2) for c in court_coords]  # Printing player court position on court outline
    image_coords = np.array(df_frame[["x1", "y1", "x2", "y2", "team"]])
    [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]),  color_list[int(box[4])%len(color_list)], 1) for box in image_coords]
    cv2.imshow('court', std_img_copy)
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    t += PER_FRAME