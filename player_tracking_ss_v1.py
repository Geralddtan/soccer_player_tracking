import pandas as pd
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from filterpy.stats import mahalanobis
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

pd.options.display.float_format = '{:.6f}'.format

def run_player_tracking_ss(match_details, to_save):
    for detail in match_details:
        MATCH_ID = detail[0]
        START_MS = detail[1]  # 61680   #91680
        END_MS = detail[2]  # 67040   #106880
        VIDEO = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/Deep Sort/detectron2-deepsort-pytorch/original_vids/m-%03d.mp4' % MATCH_ID
        FPS = detail[3]
        PER_FRAME = 1000 / FPS
        DT_CSV = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/Tracking Implementation/player_tracking_ss/csv/player_detection_colorhist/m-%03d-player-dt25-team-%d-%d.csv' % (
            MATCH_ID, START_MS, END_MS)
        DT_THRESHOLD = 0.7
        COLOR_THRESHOLD = 0.2
        MAX_P = 1000
        TEAM_OPTIONS = [0,1,2,3,4]
        OUT_CSV = "/Users/geraldtan/Desktop/NUS Modules/Dissertation/Tracking Implementation/Checking/TrackEval/data/trackers/mot_challenge/soccer-player-test/player_tracking_ss_color_hist_optimal/data/m-%03d.txt" % (MATCH_ID)
        ID0 = 1
        for team in TEAM_OPTIONS:
            TEAM = team # 0=team1, 1=team1_keeper, 2=team2, 3=team2_keeper, 4=referee
            DEBUG = True

            box_kf = KalmanFilter(6, 4)  # state contains [u, v, w, h, du, dv], measure [u, v, w, h]
            # 6 is the size of the state vector, 4 is the size of the measurement vector (there is no measurement state, 4 is just the total size of measurement)
            # u, v = x,y of center of bbox. du, dv = change in u & v (for velocity)
            '''
            State here refers to the current location + velocity of the bbox (and its uncertainty)
            Measurement is the detectron2 measurement of bbox (and its uncertainty)
            '''

            box_kf.F = np.array([[1., 0., 0., 0., 1., 0.],
                                [0., 1., 0., 0., 0., 1.],
                                [0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0.],
                                [0., 0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 0., 1.]])
            box_kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.],
                                [0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0.]])
            box_kf.Q = np.diag(
                [0., 0., 0.25, 0.25, 1.5, 1.5])  # Uncertainty for state.  Higher uncertainty is put on du, dv at the moment.
            box_kf.R = np.diag(
                [81., 81., 100., 400.])  # Uncertainty for measurement.  Higher uncertainty is put on w, h at the moment.
            # box_kf.F = state transition matrix
            # box_kf.H = measurement function
            # box_kf.Q = Process uncertainty/noise (State uncertainty)
            # box_kf.R = measurement uncertainty/noise (Detectron2 measurement uncertainty)

            BOX_KF_INIT_P = np.diag(
                [81., 81., 81., 81., 100., 400.])  # Initial covariance of state. Higher uncertainty is put on du, dv at the moment.

            '''
            What is the difference between box_kf.Q & BOX_KF_INIT_P?
            '''

            loc_kf = KalmanFilter(4,
                                2)  # state contains [x, y, dx, dy], measure [x, y] # Separate Kalman filter for soccer court coordinate
            loc_kf.F = np.array([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
            loc_kf.H = np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]])
            loc_kf.Q = np.diag([0., 0., 0.5, 0.5])  #
            loc_kf.R = np.diag([2000., 2000.])
            LOC_KF_INIT_P = np.diag([2000., 2000., 2000., 2000.])  # Equal uncertainty for x,y,dx,dy

            # A set of detected boxes on court (boxes, [k, u,v,w,h,x,y]), and their classes.
            # 3 sets of tracks, ie act_tracks, hold_tracks, delt_tracks, each track is a dictionary containing: kf, zs, xs, Ps
            # 1. Filter out person outside the court. Optionally, correct box height (for all boxes in the current time step, box_k)
            # 2. For track in act_tracks, do track.predict()
            # 3. Calculate iou(u,v,w,h) for all (act_track, box_k) pair, global nearest neighbour (GNN) assignment with min iou=0.6 Only do this for very certain tracks, ie cov of [u,v,w,h] < 200
            # 4. For remaining t and box_k, calculate mahalanobis(u,v,w,h), GNN assignment with gating=5.0 Only do this for quite certain tracks, ie cov of [u,v,w,h]<2000 (pixel coordinate)
            # 5. For remaining t and box_k, calculate euclidean(x,y), GNN assignment with max distance 50 (pixels) (soccer pitch coordinate)
            # 6. For all assigned track, do t.update(). Make w and h very smooth.
            # 7. Delete tracks with very large uncertainty cov>1e6. (move it to deleted_tracks)
            # 8. Use the remaining unassigned box_k to initial new holding_track or promote holding_track to track

            # Similar to the active tracks, but focus on the holding_tracks instead.
            # 8.1 For t in holding_tracks, do t.predict()
            # 8.2 Calculate iou(u,v,w,h) for all (t, box_k), global nearest neighbour (GNN) assignment with min iou=0.6
            # 8.3 For remaining t and box_k, calculate mahalanobis(u,v,w,h), GNN assignment with gating=5.0
            # 8.4 For all assigned track, do t.update(). Make w and h very smooth. Store the (x,P) into buffer xs, Ps
            # 8.5 If t.length >= 8, promote it to act_track. Delete holding_track if it is too uncertain.
            # 8.6 Use the remaining unassigned box_k to initial new holding_track.

            # They first perform GNN to assign boxes to each other --> then use the assigned boxes to perform kalman filter update to get the next estimate

            act_tracks, hold_tracks, delt_tracks = [], [], []
            # each track is a dictionary containing: kf, zs, xs, Ps

            player_boxes = pd.read_csv(DT_CSV).to_numpy()
            player_boxes = player_boxes[(player_boxes[:, 5] > DT_THRESHOLD)]  # Detectron confidence
            player_boxes = player_boxes[(np.max(player_boxes[:, 6:11], axis=1) > COLOR_THRESHOLD)]
            # [6:11] is the confidence score for team 1, t1 goalkeeper, t2, t2gk ,ref where lower is better (histogram similarity lower score better)
            player_boxes = player_boxes[(np.argmax(player_boxes[:, 6:11], axis=1) == TEAM)]
            # Filtering out only those rows where most similar to TEAM (only analyse that team)
            player_boxes[:, 0] = np.round(
                (player_boxes[:, 0] - START_MS) / PER_FRAME + 1)  # time_ms to k (Change time ms to frame counter)
            player_boxes[:, 3] = player_boxes[:, 3] - player_boxes[:, 1]  # x2 to w
            player_boxes[:, 4] = player_boxes[:, 4] - player_boxes[:, 2]  # y2 to h
            player_boxes[:, 1] = player_boxes[:, 1] + player_boxes[:, 3] / 2  # x1 to u
            player_boxes[:, 2] = player_boxes[:, 2] + player_boxes[:, 4] / 2  # y1 to v
            player_boxes[:, 7] = np.argmax(player_boxes[:, 6:11], axis=1)  # team
            player_boxes = player_boxes[:, 0:8]
            '''
            Final format for above is 
            [frame counter, u,v,w,h,detectron2 score, team1_similarity_score, final team number]
            '''

            COURT_CSV = '/Users/geraldtan/Desktop/NUS Modules/Dissertation/SVA/soccer/csv/court-corners-gt.csv'
            court_gt = pd.read_csv(COURT_CSV)
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


            def get_standard_court(court_points, img_size=(896, 896, 3), sport='soccer', line_thickness=2):
                if sport == 'soccer':
                    img = np.zeros(img_size, dtype=np.uint8)
                    points = np.round(
                        court_points[:, ::-1] * 8 + [(img_size[1] - court_points[19, 1] * 8) / 2, img_size[0] / 2]).astype(np.int)
                    cv2.circle(img, tuple(points[10]), 73, (255, 0, 0), line_thickness)
                    img[:, 0:points[8, 0]] = 0
                    cv2.circle(img, tuple(points[29]), 73, (255, 0, 0), line_thickness)
                    img[:, points[27, 0]:] = 0
                    cv2.rectangle(img, tuple(points[14]), tuple(points[34]), (255, 0, 0), line_thickness)
                    cv2.rectangle(img, tuple(points[4]), tuple(points[7]), (255, 0, 0), line_thickness)
                    cv2.rectangle(img, tuple(points[0]), tuple(points[3]), (255, 0, 0), line_thickness)
                    cv2.rectangle(img, tuple(points[19]), tuple(points[22]), (255, 0, 0), line_thickness)
                    cv2.rectangle(img, tuple(points[23]), tuple(points[26]), (255, 0, 0), line_thickness)
                    cv2.line(img, tuple(points[16]), tuple(points[18]), (255, 0, 0), line_thickness)
                    cv2.circle(img, tuple(points[17]), 73, (255, 0, 0), line_thickness)
                    cv2.circle(img, tuple(points[10]), 3, (0, 0, 255), -1)
                    cv2.circle(img, tuple(points[29]), 3, (0, 0, 255), -1)
                    return img, court_points[:, ::-1] * 8 + [(img_size[1] - court_points[19, 1] * 8) / 2, img_size[0] / 2]


            std_img, std_court_points = get_standard_court(court_points, line_thickness=1)
            [cv2.circle(std_img, tuple(p), 1, (0, 0, 255), -1) for p in np.round(std_court_points).astype(int)]
            cv2.imshow('std_img', std_img)


            # Plotting soccer field

            def get_H(t):
                prev = court_gt[(court_gt.match_id == MATCH_ID) & (court_gt.time_ms <= t)].to_numpy()[-1, 1:10]
                prev_t, prev_court = prev[0], prev[1:9].reshape(4, 2)
                next = court_gt[(court_gt.match_id == MATCH_ID) & (court_gt.time_ms > t)].to_numpy()[0, 1:10]
                next_t, next_court = next[0], next[1:9].reshape(4, 2)
                curr_court = prev_court + (next_court - prev_court) * (t - prev_t) / (next_t - prev_t)
                H = cv2.getPerspectiveTransform((np.array(curr_court).reshape(4, 2)).astype(np.float32),
                                                (np.array(std_court_points[[14, 33, 15, 34]]).reshape(4, 2)).astype(np.float32))
                return H


            def iou(bbox, candidates):  # copied from Deep_Sort
                """Computer intersection over union.

                Parameters
                ----------
                bbox : ndarray
                    A bounding box in format `(mid x, mid y, width, height)`.
                candidates : ndarray
                    A matrix of candidate bounding boxes (one per row) in the same format
                    as `bbox`.

                Returns
                -------
                ndarray
                    The intersection over union in [0, 1] between the `bbox` and each
                    candidate. A higher score means a larger fraction of the `bbox` is
                    occluded by the candidate.

                """
                bbox_tl, bbox_br = bbox[:2] - bbox[2:] / 2, bbox[:2] + bbox[2:] / 2
                candidates_tl = candidates[:, :2] - candidates[:, 2:] / 2
                candidates_br = candidates[:, :2] + candidates[:, 2:] / 2

                tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
                br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
                wh = np.maximum(0., br - tl)

                area_intersection = wh.prod(axis=1)
                area_bbox = bbox[2:].prod()
                area_candidates = candidates[:, 2:].prod(axis=1)
                return area_intersection / (area_bbox + area_candidates - area_intersection)


            def assign_by_iou(boxes_dt, boxes_track, min_iou=0.6):
                '''
                Returns 
                row index returns final indexes (those removed didnt survive iou = 0.6)
                col_ind returns final bbox assigned to each corresponding index 
                    [5,4,3,2,1,0] means box[0] assigned to 5 etc
                iou_matrix: Overall IOU matrix
                '''
                iou_matrix = []  # one row per tracked box, one col per detected box
                for box in boxes_track:
                    iou_matrix.append(iou(box, boxes_dt))
                iou_matrix = np.array(iou_matrix)
                cost = -1 * iou_matrix
                row_ind, col_ind = linear_sum_assignment(cost)
                to_keep = []
                for i in range(row_ind.size):
                    if iou_matrix[row_ind[i], col_ind[i]] >= min_iou:
                        to_keep.append(i)
                return row_ind[to_keep], col_ind[to_keep], iou_matrix


            def assign_by_mahalanobis(boxes_dt, kfs, gating=5.0):  # change gating to 2.0
                distance_matrix = []  # one row per tracked box, one col per kalman filter
                for f in kfs:
                    distance_matrix.append([mahalanobis(box, f.x[0:4], f.P[0:4, 0:4]) for box in boxes_dt])
                distance_matrix = np.array(distance_matrix)
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                to_keep = []
                for i in range(row_ind.size):
                    if distance_matrix[row_ind[i], col_ind[i]] <= gating:
                        to_keep.append(i)
                return row_ind[to_keep], col_ind[to_keep], distance_matrix


            def assign_by_euclidean(locs_dt, locs_track, gating=50.0):
                distance_matrix = []  # one row per tracked locations, one col per detected locations
                for p in locs_track:
                    distance_matrix.append(np.linalg.norm(locs_dt - p, axis=1))
                distance_matrix = np.array(distance_matrix)
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                to_keep = []
                for i in range(row_ind.size):
                    if distance_matrix[row_ind[i], col_ind[i]] <= gating:
                        to_keep.append(i)
                return row_ind[to_keep], col_ind[to_keep], distance_matrix


            cap = cv2.VideoCapture(VIDEO)
            t = START_MS
            cap.set(0, t)
            while t <= END_MS:
                ret, frame = cap.read()
                k = int(round((t - START_MS) / PER_FRAME)) + 1
                boxes_k = player_boxes[player_boxes[:, 0] == k]  # Filtering predictions to only be from that specific frame
                ''' boxes_k format = [frame counter, u,v,w,h,detectron2 score, team1_similarity_score, final team number]'''
                if boxes_k.size > 0:
                    player_ub = np.concatenate([boxes_k[:, 1:2], boxes_k[:, 2:3] + boxes_k[:, 4:5] / 2],
                                            axis=1)  # Player coordinate at middle top (using top of bbox as reference instead of bottom)
                    ''' Converting player location into court coordinates '''
                    player_locs = cv2.perspectiveTransform((player_ub.reshape(-1, 2)).astype(np.float32)[np.newaxis], get_H(t))[0]
                    boxes_k[:, 5:7] = player_locs
                    ''' boxes_k format here = [frame counter, u,v,w,h, court x coordinate, court y coordinate, final team number]'''

                if DEBUG:
                    rounded_box = np.round(
                        np.concatenate([boxes_k[:, 1:2] - boxes_k[:, 3:4] / 2, boxes_k[:, 2:3] - boxes_k[:, 4:5] / 2,
                                        boxes_k[:, 1:2] + boxes_k[:, 3:4] / 2, boxes_k[:, 2:3] + boxes_k[:, 4:5] / 2],
                                    axis=1)).astype(int)
                    # rounded_box = x1y1x2y2 of bboxes
                    rounded_loc = np.round(boxes_k[:, 5:7]).astype(int)
                    # rounded_loc = x,y court coordinate
                    '''Printing blue color as detectron predictions'''
                    [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (255, 0, 0), 1) for box in rounded_box]
                    std_img_copy = std_img.copy()  # Court visualisation
                    [cv2.circle(std_img_copy, tuple(c), 5, (255, 0, 0), 1) for c in
                    rounded_loc]  # Printing player court position on court outline

                # step 1, filter out persons outside the court
                if boxes_k.size > 0:
                    '''Removing player based on court coordinates'''
                    boxes_k = boxes_k[
                        (boxes_k[:, 5] >= 22.8) & (boxes_k[:, 5] <= 873.2) & (boxes_k[:, 6] >= 176) & (boxes_k[:, 6] <= 720)]
                    # optionally, correct box height based on average box height for the nearby player in the k-1 step

                # step 2, call predict for all act_tracks
                for track in act_tracks:
                    track['box_kf'].predict()  # Image pixels
                    track['loc_kf'].predict()  # Soccer court coordinate

                # step 3,
                if len(act_tracks) > 0:  # Refer to step 8.2 for similar explanation
                    candidates = [track for track in act_tracks if np.max(
                        track['box_kf'].P[0:4, 0:4]) < 200]  # If all uncertainty of u,v,w,h,du,dv is < 200, keep as candidate
                    non_candidates = [track for track in act_tracks if np.max(track['box_kf'].P[0:4, 0:4]) >= 200]
                    if len(candidates) > 0:
                        act_track_boxes = np.array([track['box_kf'].x[0:4] for track in candidates]).reshape(-1,
                                                                                                            4)  # For actual tracks where covariance < 200, get their image pixel coordinates
                        row_ind, col_ind, _ = assign_by_iou(boxes_k[:, 1:5],
                                                            act_track_boxes)  # Obtain matched box (from next frame) to these actual tracks
                        [candidates[row_ind[i]]['zs'].append(boxes_k[col_ind[i]]) for i in range(
                            row_ind.size)]  # Append these matches boxes to these candidates 'zs' which stores all of its past matches
                        act_tracks2 = [candidates[i] for i in range(len(candidates)) if
                                    i not in row_ind] + non_candidates  # If no IOU matching/covariance >200, go to next stage
                        # act_tracks2 is the remaining of those which have not been assigned
                        boxes_k2 = boxes_k[[i for i in range(boxes_k.shape[0]) if
                                            i not in col_ind]]  # All frames from next frame which have not yet been matched through IOU goes to next stage
                    else:
                        act_tracks2 = act_tracks
                        boxes_k2 = boxes_k
                else:
                    act_tracks2 = act_tracks
                    boxes_k2 = boxes_k

                # step 4, #Same as step 3 only slight differences
                if len(act_tracks2) > 0:  # For boxes not assigned in first phase (above)
                    candidates = [track for track in act_tracks2 if np.max(track['box_kf'].P[0:4, 0:4]) < 2000]
                    non_candidates = [track for track in act_tracks2 if np.max(track['box_kf'].P[0:4, 0:4]) >= 2000]

                    if len(candidates) > 0:
                        # kfs = [track['box_kf'] for track in candidates] #old
                        # row_ind, col_ind, distance_matrix = assign_by_mahalanobis(boxes_k2[:, 1:5], kfs) #old

                        act_track_boxs = np.array([track['box_kf'].x[0:2] for track in candidates]).reshape(-1, 2)
                        row_ind, col_ind, distance_matrix = assign_by_euclidean(boxes_k2[:, 1:3],
                                                                                act_track_boxs)  # [5:7] is court coordinate
                        [candidates[row_ind[i]]['zs'].append(boxes_k2[col_ind[i]]) for i in range(row_ind.size)]
                        act_tracks3 = [candidates[i] for i in range(len(candidates)) if i not in row_ind] + non_candidates
                        # Those which still haven't been assigned goes to euclidean court coordinate measurement
                        boxes_k3 = boxes_k2[[i for i in range(boxes_k2.shape[0]) if i not in col_ind]]
                    else:
                        act_tracks3 = act_tracks2
                        boxes_k3 = boxes_k2
                else:
                    act_tracks3 = act_tracks2
                    boxes_k3 = boxes_k2

                # step 5, #Same as step 3/4 just using court coordinate instead
                if len(act_tracks3) > 0:
                    act_track_locs = np.array([track['loc_kf'].x[0:2] for track in act_tracks3]).reshape(-1, 2)
                    row_ind, col_ind, distance_matrix = assign_by_euclidean(boxes_k3[:, 5:7],
                                                                            act_track_locs)  # [5:7] is court coordinate
                    [act_tracks3[row_ind[i]]['zs'].append(boxes_k3[col_ind[i]]) for i in range(row_ind.size)]
                    # print("using court coordinate")
                    # [print(tuple(np.round(boxes_k3[col_ind[i], 1:3]).astype(int)))for i in range(row_ind.size)]
                    [cv2.circle(frame, tuple(np.round(boxes_k3[col_ind[i], 1:3]).astype(int)), 20, (0, 255, 255), 1) for i in
                    range(row_ind.size)]
                    # When actual tracks don't get assigned from iou/mahalanobis, we use court coordinate.
                    # Yellow circle means using court coordinate to join
                    boxes_k = boxes_k3[[i for i in range(boxes_k3.shape[0]) if i not in col_ind]]
                else:
                    boxes_k = boxes_k3

                # step 6,
                for track in act_tracks:
                    if track['zs'][-1][0] == k:  # with assigned box  (If they have a box assigned to it at current frame k, then perform updating)
                        track['box_kf'].update(track['zs'][-1][1:5])  ##Update with pixel coordinates
                        track['loc_kf'].update(track['zs'][-1][5:7])  # Update with court coordinate
                    track['box_xs'].append(
                        track['box_kf'].x)  # Posterior pixel location from kalman filter (final prediction after updating)
                    track['box_Ps'].append(track['box_kf'].P)  # Uncertainty for pixel location (u,v,w,h,du,dv)
                    track['loc_xs'].append(track['loc_kf'].x)  # Posterior court coordinate from kalman filter
                    track['loc_Ps'].append(track['loc_kf'].P)  # Uncertainty for court coordinate (x,y,dx,dy)

                # step 7, #Discarding actual tracks if it gets very uncertain (1e6)
                if len(act_tracks) > 0:
                    to_keep = []
                    for i in range(len(act_tracks)):
                        if np.max(act_tracks[i]['box_kf'].P) > 1e6:  # Filtering out those tracks with high uncertainty
                            delt_tracks.append(act_tracks[i])
                        else:
                            to_keep.append(i)
                    act_tracks = [act_tracks[i] for i in range(len(act_tracks)) if i in to_keep]

                # step 8, use remaining boxes_k to initial or promote hold_tracks
                # step 8.1, for each hold_track, call predict
                for track in hold_tracks:
                    track['box_kf'].predict()  # Obtains prior (prediction based on initial position and velocity)
                    track['loc_kf'].predict()  # Obtains prior (prediction based on initial position and velocity)

                # step 8.2,
                if len(hold_tracks) > 0:
                    hold_track_boxes = np.array([track['box_kf'].x[0:4] for track in hold_tracks]).reshape(-1,
                                                                                                        4)  # Obtain prior prediction u,v,w,h
                    row_ind, col_ind, _ = assign_by_iou(boxes_k[:, 1:5],
                                                        hold_track_boxes)  # Assign all next frame boxes (boxes_k) to holding track as per IOU
                    [hold_tracks[row_ind[i]]['zs'].append(boxes_k[col_ind[i]]) for i in range(row_ind.size)]
                    # Assign next frame box (boxes_k[col_ind]) to holding track (hold_tracks[row_ind])
                    # 'zs' essentially saves all the new bboxes (across frames) which has been assigned to this holding track
                    hold_tracks2 = [hold_tracks[i] for i in range(len(hold_tracks)) if
                                    i not in row_ind]  # Unassigned holding track moves on to next stage
                    boxes_k2 = boxes_k[[i for i in range(boxes_k.shape[0]) if i not in col_ind]]
                else:
                    hold_tracks2 = hold_tracks
                    boxes_k2 = boxes_k

                # step 8.3, #Refer to step 8.2 for similar explanation
                if len(hold_tracks2) > 0:
                    # kfs = [track['box_kf'] for track in hold_tracks2]
                    # row_ind, col_ind, _ = assign_by_mahalanobis(boxes_k2[:, 1:5], kfs)

                    kfs = np.array([track['box_kf'].x[0:2] for track in hold_tracks2]).reshape(-1, 2)
                    row_ind, col_ind, _ = assign_by_euclidean(boxes_k2[:, 1:3], kfs)  # [5:7] is court coordinate

                    [hold_tracks2[row_ind[i]]['zs'].append(boxes_k2[col_ind[i]]) for i in range(row_ind.size)]
                    boxes_k3 = boxes_k2[[i for i in range(boxes_k2.shape[0]) if
                                        i not in col_ind]]  # Those which continue to not be assigned moves to next stage
                else:
                    boxes_k3 = boxes_k2

                # step 8.4
                for track in hold_tracks:
                    if track['zs'][-1][0] == k:  # with assigned box (k = current frame)
                        # If track has been assigned to something at this current frame, then update the kalman filter
                        track['box_kf'].update(track['zs'][-1][1:5])
                        track['loc_kf'].update(track['zs'][-1][5:7])
                    track['box_xs'].append(
                        track['box_kf'].x)  # Posterior pixel location from kalman filter (final prediction after updating)
                    track['box_Ps'].append(track['box_kf'].P)  # Uncertainty for pixel location (u,v,w,h,du,dv)
                    track['loc_xs'].append(track['loc_kf'].x)  # Posterior court coordinate from kalman filter
                    track['loc_Ps'].append(track['loc_kf'].P)  # Uncertainty for court coordinate (x,y,dx,dy)

                # step 8.5
                if len(hold_tracks) > 0:
                    to_keep = []
                    for i in range(len(hold_tracks)):
                        if len(hold_tracks[i]['zs']) >= 8:  # If have frame for more than 8
                            act_tracks.append(hold_tracks[i])  # Move hold track to act track if got 8 occurences
                        elif np.max(hold_tracks[i][
                                        'box_kf'].P) > 1000:  # If uncertainty for most current prediction (posterior) is high, delete
                            if DEBUG:
                                print('hold_track[%d] deleted' % i)
                            # no need to save such short lived track to the delt_tracks
                        else:
                            to_keep.append(i)
                    hold_tracks = [hold_tracks[i] for i in range(len(hold_tracks)) if i in to_keep]

                    # step 8.6 (Initialises holding tracks for new players)
                for box in boxes_k3:
                    new_box_kf = deepcopy(box_kf)  # New pixel coordinate kalman filter per bbox (new boxes)
                    new_box_kf.x = np.concatenate([box[1:5], np.array([0., 0.])])  # concatenating state estimate u,v,w,h,du,dv
                    new_box_kf.P = BOX_KF_INIT_P  # Initial covariance of pixel coordinate state (u,v,w,h,du,dv). Higher uncertainty is put on du, dv at the moment.
                    new_loc_kf = deepcopy(loc_kf)  # New court location kalman filter
                    new_loc_kf.x = np.concatenate([box[5:7], np.array([0., 0.])])  # concatenating x,y,dx,dy court coordinate
                    new_loc_kf.P = LOC_KF_INIT_P  # Initial covariance of court coordinate state (x,y,dx,dy). Higher uncertainty is put on dx, dy at the moment.
                    hold_tracks.append({'box_kf': new_box_kf, 'loc_kf': new_loc_kf, 'zs': [box],
                                        'box_xs': [new_box_kf.x], 'box_Ps': [new_box_kf.P], 'loc_xs': [new_loc_kf.x],
                                        'loc_Ps': [new_loc_kf.P]})

                '''Printing holding tracks in red'''
                if DEBUG:
                    if len(hold_tracks) > 0:
                        hold_track_boxes = np.array([track['box_kf'].x[0:4] for track in hold_tracks]).reshape(-1, 4)
                        rounded_box = np.round(np.concatenate([hold_track_boxes[:, 0:1] - hold_track_boxes[:, 2:3] / 2,
                                                            hold_track_boxes[:, 1:2] - hold_track_boxes[:, 3:4] / 2,
                                                            hold_track_boxes[:, 0:1] + hold_track_boxes[:, 2:3] / 2,
                                                            hold_track_boxes[:, 1:2] + hold_track_boxes[:, 3:4] / 2],
                                                            axis=1)).astype(int)
                        '''#Printing red color as holding tracks (not yet initialised to actual tracks)'''
                        [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (0, 0, 255), 1) for box in rounded_box]

                        # hold_track_xys = np.array([track['loc_kf'].x[0:2] for track in hold_tracks]).reshape(-1, 2)
                        # rounded_loc = np.round(hold_track_xys).astype(int)
                        # [cv2.circle(std_img_copy, tuple(c), 5, (0, 0, 255), 1) for c in rounded_loc]

                        # for track in hold_tracks:
                        #     [print(np.round(x)) for x in track['zs']]
                        #     [print(np.round(x)) for x in track['loc_xs']]

                    '''Printing actual tracks in yellow'''
                    if len(act_tracks) > 0:
                        act_track_boxes = np.array([track['box_kf'].x[0:4] for track in act_tracks]).reshape(-1, 4)
                        rounded_box = np.round(np.concatenate([act_track_boxes[:, 0:1] - act_track_boxes[:, 2:3] / 2,
                                                            act_track_boxes[:, 1:2] - act_track_boxes[:, 3:4] / 2,
                                                            act_track_boxes[:, 0:1] + act_track_boxes[:, 2:3] / 2,
                                                            act_track_boxes[:, 1:2] + act_track_boxes[:, 3:4] / 2],
                                                            axis=1)).astype(int)
                        '''Printing Yellow color as actual tracks'''
                        [cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]), (0, 255, 255), 1) for box in rounded_box]
                        act_track_xys = np.array([track['loc_kf'].x[0:2] for track in act_tracks]).reshape(-1, 2)
                        rounded_loc = np.round(act_track_xys).astype(int)
                        [cv2.circle(std_img_copy, tuple(c), 5, (0, 255, 255), 1) for c in
                        rounded_loc]  # Printing on court coordinate

                cv2.imshow('frame', frame)
                cv2.imshow('std_img', std_img_copy)
                # if k < 67:
                #     cv2.waitKey(10)
                # else:
                #     cv2.waitKey(0)
                cv2.waitKey(1)
                t += PER_FRAME

            # step 9, smoothing
            for track in act_tracks:
                smoothed_xs, smoothed_Ps, _, _ = track['box_kf'].rts_smoother(np.array(track['box_xs']).reshape(-1, 6, 1),
                                                                            np.array(track['box_Ps']).reshape(-1, 6, 6))
                track['smo_box_xs'] = smoothed_xs
                track['smo_box_Ps'] = smoothed_Ps

            for track in delt_tracks:
                smoothed_xs, smoothed_Ps, _, _ = track['box_kf'].rts_smoother(np.array(track['box_xs']).reshape(-1, 6, 1),
                                                                            np.array(track['box_Ps']).reshape(-1, 6, 6))
                track['smo_box_xs'] = smoothed_xs
                track['smo_box_Ps'] = smoothed_Ps

                '''
                Prints tracks in unique colors
                '''

            color_list = [[255, 0, 0], [0, 0, 255], [0, 255, 0],
                        [255, 255, 0], [255, 0, 255], [0, 255, 255],
                        [128, 128, 0], [128, 0, 128], [0, 128, 128],
                        [255, 0, 128], [255, 128, 0], [0, 255, 128],
                        [128, 255, 0], [128, 0, 255], [0, 128, 255],
                        [128, 128, 255], [128, 255, 128], [255, 128, 128],
                        [128, 255, 255], [255, 128, 255], [255, 255, 128]]

            t = START_MS
            cap.set(0, t)
            while t <= END_MS:
                ret, frame = cap.read()
                height, width = frame.shape[0:2]
                k = int(round((t - START_MS) / PER_FRAME)) + 1
                for track_i in range(len(act_tracks)):
                    k0 = int(act_tracks[track_i]['zs'][0][
                                0])  # Extracts each actual track and then does some computation to draw them out
                    k_i = k - k0
                    if k >= k0 and k_i < act_tracks[track_i]['smo_box_xs'].shape[0]:
                        box = act_tracks[track_i]['smo_box_xs'][k_i, 0:4]
                        box[0] = box[0] - box[2] / 2
                        box[1] = box[1] - box[3] / 2
                        box[2] = box[0] + box[2]
                        box[3] = box[1] + box[3]
                        max_P = np.max(act_tracks[track_i]['smo_box_Ps'][k_i, 0:4, 0:4])
                        if box[0] < width and box[2] >= 0 and box[1] < height and box[3] >= 0 and max_P < MAX_P:  # Filter if out of screen/if smoothed covariance higher than MAX_P
                            rounded_box = np.round(box).astype(int)
                            cv2.rectangle(frame, (int(rounded_box[0]), int(rounded_box[1])),
                                        (int(rounded_box[2]), int(rounded_box[3])), color_list[track_i % len(color_list)], 1)
                            # cv2.rectangle(frame, tuple(rounded_box[0:2]), tuple(rounded_box[2:4]), color_list[track_i%len(color_list)], 1)
                            print('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1' % (
                                k, track_i + ID0, box[0], box[1], box[2] - box[0], box[3] - box[1]))
                            if to_save:
                                out = open(OUT_CSV, 'a+')
                                out.write('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1\n' % (
                                    k, track_i + ID0, box[0], box[1], box[2] - box[0], box[3] - box[1]))
                                out.close()
                            
                for track_i in range(len(delt_tracks)):
                    k0 = int(delt_tracks[track_i]['zs'][0][0])
                    k_i = k - k0
                    if k >= k0 and k_i < delt_tracks[track_i]['smo_box_xs'].shape[0]:
                        box = delt_tracks[track_i]['smo_box_xs'][k_i, 0:4]
                        box[0] = box[0] - box[2] / 2
                        box[1] = box[1] - box[3] / 2
                        box[2] = box[0] + box[2]
                        box[3] = box[1] + box[3]
                        max_P = np.max(delt_tracks[track_i]['smo_box_Ps'][k_i, 0:4, 0:4])
                        if box[0] < width and box[2] >= 0 and box[1] < height and box[3] >= 0 and max_P < MAX_P:
                            rounded_box = np.round(box).astype(int)
                            cv2.rectangle(frame, (int(rounded_box[0]), int(rounded_box[1])),
                                        (int(rounded_box[2]), int(rounded_box[3])), color_list[track_i % len(color_list)], 1)
                            print("HAHOHAHOHAHOHAHOHAHOHAHO")
                            # cv2.rectangle(frame, tuple(rounded_box[0:2]), tuple(rounded_box[2:4]), color_list[track_i%len(color_list)], 1)
                            print('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1' % (
                                k, track_i + len(act_tracks) + ID0, box[0], box[1], box[2] - box[0], box[3] - box[1]))
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                t += PER_FRAME
            
            ID0 += len(act_tracks)
            print('# of act tracks = %d, # of del tracks = %d' % (len(act_tracks), len(delt_tracks)))
