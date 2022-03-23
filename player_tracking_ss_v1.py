import pandas as pd
import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from copy import deepcopy
import helper_player_tracking

pd.options.display.float_format = '{:.6f}'.format

def run_player_tracking_ss(match_details, to_save, save_all_details):
    for detail in match_details:
        # Match Details Parsed from player_tracking_compute.py [MATCH_ID, START_MS, END_MS, FPS of Video]
        MATCH_ID, START_MS, END_MS, FPS = detail[0], detail[1], detail[2], detail[3]
        PER_FRAME = 1000 / FPS

        # Tracking Parameters
        DT_THRESHOLD, COLOR_THRESHOLD, MAX_P = 0.5, 0.6, 1000
        TEAM_OPTIONS = [0,1,2,3,4]
        ID0 = 1
        TRACK_ID = 0

        # Initialised Objects for Storage
        COURT_BOUNDARIES = {}
        HOMOG_TRANSF = {}
        ALL_FRAMES = {}
        
        # File Paths
        VIDEO = './m-%03d.mp4' % MATCH_ID #File Path to Match Video
        DT_CSV = './m-%03d-player-team-%d-%d.csv' % (MATCH_ID, START_MS, END_MS) #File Path to Player Classification CSV File from player_assign_team.py
        OUT_CSV = "./m-%03d.txt" % (MATCH_ID) #File Path to final tracking data output file
        for team in TEAM_OPTIONS: # Tracking Iterations according to team
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
            '''
            This is the state transition matrix used in prior calculation in the predict step. As you can see
            apart from the diagonals which mean u,v,w,h,du,dv is the same across iterations, we have [1] for the 1st and 2nd row
            corresponding to the du,dv values. This means that in each iteration, u1 = u0 + du & v1 = v0 + dv
            Q is the associated covariance for F. 
            '''

            box_kf.H = np.array([[1., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0., 0.],
                                [0., 0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0., 0.]])
            '''
            H is the measurement function (or matrix) to convert state (u,v,w,h,du,dv) [from prior in predict step] into measurement format (u,v,w,h) since
            our detectron measurements are in (u,v,w,h). This conversion is so that we can calculate residual between
            prior and measurement in the (u,v,w,h) form.
            As can be seen, we only need diagonals for u,v,w,h and [0] for du,dv.
            '''

            # box_kf.Q = np.diag(
            #     [0., 0., 0.25, 0.25, 1.5, 1.5])  # Uncertainty for state.  Higher uncertainty is put on du, dv at the moment.
            box_kf.Q = np.diag(
                [0., 0., 0.25, 0.25, 3.0, 3.0])
            '''
            Uncertainty for state transition. We make use of state transition in the predict step when calculating 
            prior for next iteration. Statet = statet-1 * F + Q.
            This represents uncertainty in the calculations of prior in the predict step. If you think the model at which we use
            to predict the next step (prior prediction in predict step), then can increase uncertainty of u,v,w,h,du,dv depending on which
            you want to change the uncertainty for.
            
            Higher uncertainty is put on du, dv at the moment.

            '''
            box_kf.R = np.diag(
                [40., 40., 100., 400.])  # Uncertainty for measurement.  Higher uncertainty is put on w, h at the moment.
            '''
            R is the measurement covariance matrix. Can think of it as the uncertainty/variance in the measurement gaussian.
            If you believe detectron predictions are good/bad, then can change uncertainty respectively. 
            Is there any way to change R based on detecttron confidence?

            '''

            # box_kf.F = state transition matrix
            # box_kf.H = measurement function
            # box_kf.Q = Process uncertainty/noise (State uncertainty)
            # box_kf.R = measurement uncertainty/noise (Detectron2 measurement uncertainty)

            BOX_KF_INIT_P = np.diag(
                [40., 40., 81., 81., 100., 400.])  # Initial covariance of state. Higher uncertainty is put on du, dv at the moment.

            '''
            What is the difference between box_kf.Q & BOX_KF_INIT_P?

            box_kf.Q is the process uncertainty assigned to u,v,w,h,du,dv when used in the calculations of predict step. It is kinda the uncertainty
            attacked to .F matrix!
            box_kf.R is the measurement uncertainty assigned to each new measurement used in the calculation of update step
            BOX_KF_INIT_P is the initial uncertainty assigned to each new track. This is the .P matrix which is the uncertainty
            of the track itself. If im not wrong, the .P & .Q matrix are used tgt in the predict step
            '''

            loc_kf = KalmanFilter(4,
                                2)  # state contains [x, y, dx, dy], measure [x, y] # Separate Kalman filter for soccer court coordinate
            loc_kf.F = np.array([[1., 0., 1., 0.],
                                [0., 1., 0., 1.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
            loc_kf.H = np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.]])
            # loc_kf.Q = np.diag([0., 0., 0.5, 0.5]) 
            loc_kf.Q = np.diag([0., 0., 1.0, 1.0])  #
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

            act_tracks, past_act_tracks, hold_tracks, not_in_view_tracks, delt_tracks = [], [], [], [], []
            # each track is a dictionary containing: kf, zs, xs, Ps

            player_boxes = pd.read_csv(DT_CSV).to_numpy()
            player_boxes = player_boxes[(player_boxes[:, 5] > DT_THRESHOLD)]  # Detectron confidence
            player_boxes = player_boxes[(np.min(player_boxes[:, 6:11], axis=1) < COLOR_THRESHOLD)]
            # player_boxes = player_boxes[(np.argmin(player_boxes[:, 6:11], axis=1) == TEAM)]
            # # Filtering out only those rows where most similar to TEAM (only analyse that team)
            # # [6:11] is the confidence score for team 1, t1 goalkeeper, t2, t2gk ,ref where lower is better (histogram similarity lower score better)

            player_boxes[:, 0] = np.round(
                (player_boxes[:, 0] - START_MS) / PER_FRAME + 1)  # time_ms to k (Change time ms to frame counter)
            player_boxes[:, 3] = player_boxes[:, 3] - player_boxes[:, 1]  # x2 to w
            player_boxes[:, 4] = player_boxes[:, 4] - player_boxes[:, 2]  # y2 to h
            player_boxes[:, 1] = player_boxes[:, 1] + player_boxes[:, 3] / 2  # x1 to u
            player_boxes[:, 2] = player_boxes[:, 2] + player_boxes[:, 4] / 2  # y1 to v
            player_boxes[:, 7] = np.argmin(player_boxes[:, 6:11], axis=1)  # team
            player_boxes = player_boxes[:, 0:8]
            player_boxes_all_classes = player_boxes #To get all players from all classes for box height comparison
            player_boxes = player_boxes[(player_boxes[:, 7] == TEAM)]

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

            std_img, std_court_points = helper_player_tracking.get_standard_court(court_points, line_thickness=1)
            [cv2.circle(std_img, tuple(p), 1, (0, 0, 255), -1) for p in np.round(std_court_points).astype(int)]
            cv2.imshow('std_img', std_img)

            # Plotting soccer field

            cap = cv2.VideoCapture(VIDEO)
            t = START_MS
            cap.set(0, t)
            while t <= END_MS:
                ret, frame = cap.read()
                height, width = frame.shape[0:2]
                k = int(round((t - START_MS) / PER_FRAME)) + 1
                if k not in ALL_FRAMES:
                    ALL_FRAMES[k] = frame
                boxes_k = player_boxes[player_boxes[:, 0] == k]  # Filtering predictions to only be from that specific frame
                boxes_k_all_classes = player_boxes_all_classes[player_boxes_all_classes[:, 0] == k]  # Filtering predictions to only be from that specific frame
                homog_transf =  helper_player_tracking.get_H(t, MATCH_ID, court_gt, std_court_points)
                if k not in HOMOG_TRANSF:
                    HOMOG_TRANSF[k] = homog_transf
                inv_homog_transf = np.linalg.inv(homog_transf)
                std_img_copy = std_img.copy()  # Court visualisation
                
                ''' boxes_k format = [frame counter, u,v,w,h,detectron2 score, team1_similarity_score, final team number]'''
                if boxes_k.size > 0:
                    player_ub = np.concatenate([boxes_k[:, 1:2], boxes_k[:, 2:3] + boxes_k[:, 4:5] / 2],
                                            axis=1)  # Player coordinate at middle top (using top of bbox as reference instead of bottom)
                    ''' Converting player location into court coordinates '''
                    player_locs = cv2.perspectiveTransform((player_ub.reshape(-1, 2)).astype(np.float32)[np.newaxis], homog_transf)[0]
                    boxes_k[:, 5:7] = player_locs
                    ''' boxes_k format here = [frame counter, u,v,w,h, court x coordinate, court y coordinate, final team number]'''

                if boxes_k_all_classes.size > 0: #Repeat of above for all classes (Dont really like this -- maybe can abstract this out)
                    player_ub_all_classes = np.concatenate([boxes_k_all_classes[:, 1:2], boxes_k_all_classes[:, 2:3] + boxes_k_all_classes[:, 4:5] / 2],
                                            axis=1)  # Player coordinate at middle top (using top of bbox as reference instead of bottom)
                    ''' Converting player location into court coordinates '''
                    player_locs_all_classes = cv2.perspectiveTransform((player_ub_all_classes.reshape(-1, 2)).astype(np.float32)[np.newaxis], homog_transf)[0]
                    boxes_k_all_classes[:, 5:7] = player_locs_all_classes
                    ''' boxes_k_all_classes format here = [frame counter, u,v,w,h, court x coordinate, court y coordinate, final team number]'''

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
                    [cv2.circle(std_img_copy, tuple(c), 5, (255, 0, 0), 1) for c in rounded_loc]  # Printing player court position on court outline

                    '''Plotting plane of view on court image'''
                    pixel_boundaries = [[0,0], [0, height], [width, height], [width, 0]] #The homography transformation of this is wrong
                    court_boundaries = [cv2.perspectiveTransform((np.array(bound).reshape(-1, 2)).astype(np.float32)[np.newaxis], homog_transf)[0][0] for bound in pixel_boundaries]
                    court_boundaries = np.array(court_boundaries, np.int32) #plotting of polylines must be int
                    cv2.polylines(std_img_copy, [court_boundaries], True, (192, 192, 192), 2)
                    if k not in COURT_BOUNDARIES:
                        COURT_BOUNDARIES[k] = court_boundaries
                    
                # step 1, filter out persons outside the court
                if boxes_k_all_classes.size > 0:
                    '''Removing player based on court coordinates'''
                    boxes_k_all_classes = boxes_k_all_classes[
                        (boxes_k_all_classes[:, 5] >= 22.8) & (boxes_k_all_classes[:, 5] <= 873.2) & (boxes_k_all_classes[:, 6] >= 176) & (boxes_k_all_classes[:, 6] <= 720)]

                cv2.imshow('frame', frame)
                if boxes_k.size > 0:
                    '''Removing player based on court coordinates'''
                    boxes_k = boxes_k[
                        (boxes_k[:, 5] >= 22.8) & (boxes_k[:, 5] <= 873.2) & (boxes_k[:, 6] >= 176) & (boxes_k[:, 6] <= 720)]
                    # optionally, correct box height based on average box height for the nearby player in the k-1 step
                    '''Get all players nearby each player by pixel values'''
                    for player_bbox_index in range(len(boxes_k)):
                        player_bbox = boxes_k[player_bbox_index]
                        player_distances = []  # one row per tracked locations, one col per detected locations
                        for p in boxes_k_all_classes:
                            player_distances.append(np.linalg.norm(player_bbox[1:3] - p[1:3])) #Euclidean distance using pixels
                        nearby_player_candidate_index = [index for index in range(len(player_distances)) if (player_distances[index] < 200) & (player_distances[index] != 0)] #Get all those close by except itself
                        '''Calculate average height'''
                        avg_height = np.mean(boxes_k_all_classes[nearby_player_candidate_index][:, 4]) #Calculate average height of all nearby
                        if player_bbox[4] < avg_height*0.4: #Smaller than 0.6 of the average height
                            u,v,w,h = player_bbox[1:5]
                            new_height = (avg_height + h)/2 # New height is average of original height and average height of those around
                            original_coordinate_top = v - h/2 # Original top of box
                            new_v = original_coordinate_top + new_height/2 
                            '''Shift value of 'v' where top coordinate of box is equal to previously, 
                            but updated to new h (so that we shift the bbox downwards since 
                            mostly we detect the top half of ths body)'''
                            boxes_k[player_bbox_index][2] = new_v
                            boxes_k[player_bbox_index][4] = new_height
                            helper_player_tracking.print_box_uvwh(frame, boxes_k[player_bbox_index][1:5], (255, 255, 255), 5)

                        if player_bbox[4] > avg_height*1.6: # Larger than 1.4 of the average height
                            u,v,w,h = player_bbox[1:5]
                            new_height = (avg_height + h)/2 # New height is average of original height and average height of those around
                            # original_coordinate_bottom = v + h/2 # Original top of box
                            # new_v = original_coordinate_bottom - new_height/2 
                            # boxes_k[player_bbox_index][2] = new_v
                            boxes_k[player_bbox_index][4] = new_height
                            ''' Only changing height here. This is because
                                For increase in height, it can be in both directions 
                                1. 1 player detected originally then another player comes on top of him and both are detected as 1 (increase in height upwards)
                                2. 1 player detected orignally but due to occlusion, only detect small part of him (from head). Subsequently, increase back to original size (increase in height downwards)

                                Thus, cannot set top/bottom of bbox strictly. Havent thought of optimal solution yet
                            '''
                            helper_player_tracking.print_box_uvwh(frame, boxes_k[player_bbox_index][1:5], (0, 0, 0), 5)

                # step 2, call predict for all act_tracks
                for track in act_tracks:
                    track['box_kf'].predict()  # Image pixels
                    track['loc_kf'].predict()  # Soccer court coordinate

                # step 3,
                if len(act_tracks) > 0:  # Refer to step 8.2 for similar explanation
                    candidates = [track for track in act_tracks if np.max(track['box_kf'].P[0:4, 0:4]) < 200]  # If all uncertainty of u,v,w,h,du,dv is < 200, keep as candidate
                    non_candidates = [track for track in act_tracks if np.max(track['box_kf'].P[0:4, 0:4]) >= 200]
                    if len(candidates) > 0:
                        act_track_boxes = np.array([track['box_kf'].x[0:4] for track in candidates]).reshape(-1,
                                                                                                            4)  # For actual tracks where covariance < 200, get their image pixel coordinates

                        row_ind, col_ind, _ = helper_player_tracking.assign_by_iou(boxes_k[:, 1:5],
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
                        row_ind, col_ind, distance_matrix = helper_player_tracking.assign_by_euclidean(boxes_k2[:, 1:3],
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
                    row_ind, col_ind, distance_matrix = helper_player_tracking.assign_by_euclidean(boxes_k3[:, 5:7],
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
                        track['frames_since_last_detection'] = 0
                    else: #If not assigned in current frame
                        track['frames_since_last_detection'] += 1

                    track['box_xs'].append(
                        track['box_kf'].x)  # Posterior pixel location from kalman filter (final prediction after updating)
                    track['box_Ps'].append(track['box_kf'].P)  # Uncertainty for pixel location (u,v,w,h,du,dv)
                    track['loc_xs'].append(track['loc_kf'].x)  # Posterior court coordinate from kalman filter
                    track['loc_Ps'].append(track['loc_kf'].P)  # Uncertainty for court coordinate (x,y,dx,dy)

                # step 7, #Discarding actual tracks if it gets very uncertain (1e6)
                if len(act_tracks) > 0:
                    to_keep = []
                    for i in range(len(act_tracks)):
                        court_x = act_tracks[i]['loc_kf'].x[0]
                        court_y = act_tracks[i]['loc_kf'].x[1]
                        player_ub = np.array([court_x, court_y]) 
                
                        ''' Converting court coordinate into current frame pixel coordinates '''
                        player_pixel_coord_from_court = cv2.perspectiveTransform((player_ub.reshape(-1, 2)).astype(np.float32)[np.newaxis], inv_homog_transf)[0][0]              
                        if (act_tracks[i]['frames_since_last_detection'] > 20) and not (helper_player_tracking.valid_pixel_coordinate(player_pixel_coord_from_court[0], player_pixel_coord_from_court[1], width, height)):  # Filtering out those tracks with high uncertainty
                            # If any of the pixel coordinate is  < 0 (out of screen) + > 15 frames since last detection
                            target_track_past_act_track = deepcopy(act_tracks[i])
                            target_track_not_in_view_track = deepcopy(act_tracks[i])
                            past_act_tracks.append(target_track_past_act_track)
                            '''For not_in_view_tracks, keep a counter of number of consecutive frames it is inside it.
                            This value is used to remove tracks in not_in_view_tracks which have been inside for too long'''
                            target_track_not_in_view_track['number_frames_in_not_in_view_tracks'] = 0
                            not_in_view_tracks.append(target_track_not_in_view_track)
                        elif act_tracks[i]['frames_since_last_detection'] > 75:
                            '''
                            Set high value to ensure we are very certain before we delete tracks. Those tracks which should be deleted
                            but havent reach 75 frames -- are still okay since we have measures put in place (to only use those where 
                            max_p < MAX_P). 
                            If track frames > 20 but still within frame, we keep as actual track first (only delete when confident) and only
                            put it into not_in_view_tracks when its out of frame
                            '''
                            # If player uncertain (but location is still within frame) == track was probably wrong in the first place (wrong assignment to the team etc)
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
                    row_ind, col_ind, _ = helper_player_tracking.assign_by_iou(boxes_k[:, 1:5],
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
                    row_ind, col_ind, _ = helper_player_tracking.assign_by_euclidean(boxes_k2[:, 1:3], kfs)  # [5:7] is court coordinate

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
                        track['frames_since_last_detection'] = 0
                    else: #If not assigned in current frame
                        track['frames_since_last_detection'] += 1

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

                '''Check if we can perform any reassignment'''
                if len(not_in_view_tracks) > 0:
                    not_in_view_boxs = np.array([helper_player_tracking.get_track_last_acc_loc_coords(track) for track in not_in_view_tracks]).reshape(-1, 2)
                    row_ind, col_ind, distance_matrix = helper_player_tracking.assign_by_euclidean(boxes_k3[:, 5:7],
                                                                                not_in_view_boxs, gating = 75)  # [5:7] is court coordinate

                    for i in range(len(row_ind)):
                        existing_track_id = not_in_view_tracks[row_ind[i]]['id'] #Obtain old track id
                        newly_detected_box = boxes_k3[col_ind[i]] #Create new holding track with old track id
                        new_track = helper_player_tracking.create_new_track(box_kf, newly_detected_box, BOX_KF_INIT_P, loc_kf, LOC_KF_INIT_P, existing_track_id)
                        hold_tracks.append(new_track)

                    # If we bring a not_in_view tracks back to holding track, we remove this track from not_in_view tracks
                    not_in_view_tracks = [not_in_view_tracks[i] for i in range(len(not_in_view_tracks)) if i not in row_ind]

                    boxes_k4 = boxes_k3[[i for i in range(boxes_k3.shape[0]) if
                                        i not in col_ind]]  # Those which continue to not be assigned moves to next stage   
                else:
                    boxes_k4 = boxes_k3

                # Keep track of number of frames tracks in not_in_view tracks are there
                for track in not_in_view_tracks:
                    track["number_frames_in_not_in_view_tracks"] += 1

                # Remove track from not_in_view tracks is too many frames has passed
                if len(not_in_view_tracks) > 0:
                    to_keep = []
                    for track in not_in_view_tracks:
                        if track["number_frames_in_not_in_view_tracks"] <= 250:
                            to_keep.append(track)
                    not_in_view_tracks = to_keep
                                
                # step 8.6 (Initialises holding tracks for new players)
                for box in boxes_k4:                                           
                    new_track = helper_player_tracking.create_new_track(box_kf, box, BOX_KF_INIT_P, loc_kf, LOC_KF_INIT_P, TRACK_ID)
                    hold_tracks.append(new_track)
                    TRACK_ID += 1

                    '''
                    'zs': Keeps assigned detectron bounding box coordinates
                    'xs': keeps posterior location (final KF predicted location)
                    'Ps': keeps KF location uncertainty (u,v,w,h,du,dv)
                    '''

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
                        [cv2.circle(std_img_copy, tuple(c), 5, (0, 255, 255), 1) for c in rounded_loc]  # Printing on court coordinate

                cv2.imshow('frame', frame)
                cv2.imshow('std_img', std_img_copy)
                cv2.waitKey(1)
                # print(k)
                t += PER_FRAME

            # step 9, smoothing
            '''Smoothing using each tracks mean and covariance'''
            for track in act_tracks:
                smoothed_xs, smoothed_Ps, _, _ = track['box_kf'].rts_smoother(np.array(track['box_xs']).reshape(-1, 6, 1),
                                                                            np.array(track['box_Ps']).reshape(-1, 6, 6))
                track['smo_box_xs'] = smoothed_xs
                track['smo_box_Ps'] = smoothed_Ps

            for track in past_act_tracks:
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
                homog_transf = HOMOG_TRANSF[k]
                for track in act_tracks:
                    k0 = int(track['zs'][0][0])   # k0 is the first frame at which this track was generated (came into view)
                    # Extracts each actual track and then does some computation to draw them out
                    k_i = k - k0 # The number of frames from start (of its existence) till current frame
                    if k >= k0 and k_i < track['smo_box_xs'].shape[0]: #If this track exists within this current frame
                        box = track['smo_box_xs'][k_i, 0:4]
                        box[0] = box[0] - box[2] / 2
                        box[1] = box[1] - box[3] / 2
                        box[2] = box[0] + box[2]
                        box[3] = box[1] + box[3]
                        max_P = np.max(track['smo_box_Ps'][k_i, 0:4, 0:4])
                        if box[0] < width and box[2] >= 0 and box[1] < height and box[3] >= 0 and max_P < MAX_P:  # Filter if out of screen/if smoothed covariance higher than MAX_P
                            rounded_box = np.round(box).astype(int)
                            cv2.rectangle(frame, (int(rounded_box[0]), int(rounded_box[1])),
                                        (int(rounded_box[2]), int(rounded_box[3])), color_list[track['id'] % len(color_list)], 1)
                            # cv2.rectangle(frame, tuple(rounded_box[0:2]), tuple(rounded_box[2:4]), color_list[track_i%len(color_list)], 1)
                            x1, y1, w, h =  box[0], box[1], box[2] - box[0], box[3] - box[1]
                            print('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1' % (
                                k, track['id'] + ID0, x1, y1, w, h))
                            if to_save:
                                if save_all_details:
                                    out = open(OUT_CSV, 'a+')
                                    court_x, court_y = helper_player_tracking.xywh_image_to_court(x1, y1, w, h, homog_transf)
                                    out.write('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%d\n' % (
                                        k, track['id'] + ID0, x1, y1, w, h, court_x, court_y, team))
                                    out.close()
                                else:
                                    out = open(OUT_CSV, 'a+')
                                    out.write('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1\n' % (
                                        k, track['id'] + ID0, x1, y1, w, h))
                                    out.close()

                for track in past_act_tracks:
                    k0 = int(track['zs'][0][0])  # Extracts each actual track and then does some computation to draw them out
                    k_i = k - k0
                    if k >= k0 and k_i < track['smo_box_xs'].shape[0]:
                        box = track['smo_box_xs'][k_i, 0:4]
                        box[0] = box[0] - box[2] / 2
                        box[1] = box[1] - box[3] / 2
                        box[2] = box[0] + box[2]
                        box[3] = box[1] + box[3]
                        max_P = np.max(track['smo_box_Ps'][k_i, 0:4, 0:4])
                        if box[0] < width and box[2] >= 0 and box[1] < height and box[3] >= 0 and max_P < MAX_P:  # Filter if out of screen/if smoothed covariance higher than MAX_P
                            rounded_box = np.round(box).astype(int)
                            cv2.rectangle(frame, (int(rounded_box[0]), int(rounded_box[1])),
                                        (int(rounded_box[2]), int(rounded_box[3])), color_list[track['id'] % len(color_list)], 1)
                            # cv2.rectangle(frame, tuple(rounded_box[0:2]), tuple(rounded_box[2:4]), color_list[track_i%len(color_list)], 1)
                            x1, y1, w, h =  box[0], box[1], box[2] - box[0], box[3] - box[1]
                            print('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1' % (
                                k, track['id'] + ID0, x1, y1, w, h))
                            if to_save:
                                if save_all_details:
                                    out = open(OUT_CSV, 'a+')
                                    court_x, court_y = helper_player_tracking.xywh_image_to_court(x1, y1, w, h, homog_transf)
                                    out.write('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%d\n' % (
                                        k, track['id'] + ID0, x1, y1, w, h, court_x, court_y, team))
                                    out.close()  
                                else:
                                    out = open(OUT_CSV, 'a+')
                                    out.write('%d,%d,%0.3f,%0.3f,%0.3f,%0.3f,-1,-1,-1,-1\n' % (
                                        k, track['id'] + ID0, x1, y1, w, h))
                                    out.close()
                                
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                t += PER_FRAME
            
            ID0 = ID0 + len(act_tracks) + len(not_in_view_tracks)
            print('# of act tracks = %d, # of del tracks = %d' % (len(act_tracks), len(delt_tracks)))


