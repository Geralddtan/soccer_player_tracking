

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.stats import mahalanobis
from copy import deepcopy

def get_H(t, MATCH_ID, court_gt, std_court_points):
    prev = court_gt[(court_gt.match_id == MATCH_ID) & (court_gt.time_ms <= t)].to_numpy()[-1, 1:10]
    prev_t, prev_court = prev[0], prev[1:9].reshape(4, 2)
    next = court_gt[(court_gt.match_id == MATCH_ID) & (court_gt.time_ms > t)].to_numpy()[0, 1:10]
    next_t, next_court = next[0], next[1:9].reshape(4, 2)
    curr_court = prev_court + (next_court - prev_court) * (t - prev_t) / (next_t - prev_t)
    H = cv2.getPerspectiveTransform((np.array(curr_court).reshape(4, 2)).astype(np.float32),
                                    (np.array(std_court_points[[14, 33, 15, 34]]).reshape(4, 2)).astype(np.float32))
    return H

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
        cv2.circle(img, tuple(points[10]), 3, (255, 0, 0), -1)
        cv2.circle(img, tuple(points[29]), 3, (255, 0, 0), -1)
        return img, court_points[:, ::-1] * 8 + [(img_size[1] - court_points[19, 1] * 8) / 2, img_size[0] / 2]

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

def print_box_uvwh(frame, uvwh, color, thickness):
    u = uvwh[0]
    v = uvwh[1]
    w = uvwh[2]
    h = uvwh[3]
    x1 = int(u - w/2)
    x2 = int(u + w/2)
    y1 = int(v - h/2)
    y2 = int(v + h/2)
    ''' Draw rectangle of new bbox '''
    return cv2.rectangle(frame, tuple((x1, y1)), tuple((x2, y2)), color, thickness)

def valid_pixel_coordinate(x,y, frame_width, frame_height):
    is_valid_x = (x <= frame_width) and (x >= 0)
    is_valid_y = (y <= frame_height) and (y >= 0)
    return is_valid_x and is_valid_y

def get_track_last_acc_loc_coords(track):
    frames_since_last_detection = track['frames_since_last_detection']
    loc_coords = track['loc_xs'][-(frames_since_last_detection + 1)]
    return loc_coords[:2] #Only return coordinates

def create_new_track(box_kf, detected_box, BOX_KF_INIT_P, loc_kf, LOC_KF_INIT_P, track_id):
    new_box_kf = deepcopy(box_kf)  # New pixel coordinate kalman filter per bbox (new boxes)
    new_box_kf.x = np.concatenate([detected_box[1:5], np.array([0., 0.])])  # concatenating state estimate u,v,w,h,du,dv
    '''We initialise each new box with box coordinate with 0 velocity du, dv'''
    new_box_kf.P = BOX_KF_INIT_P  # Initial covariance of pixel coordinate state (u,v,w,h,du,dv). Higher uncertainty is put on du, dv at the moment.
    new_loc_kf = deepcopy(loc_kf)  # New court location kalman filter
    new_loc_kf.x = np.concatenate([detected_box[5:7], np.array([0., 0.])])  # concatenating x,y,dx,dy court coordinate
    new_loc_kf.P = LOC_KF_INIT_P  # Initial covariance of court coordinate state (x,y,dx,dy). Higher uncertainty is put on dx, dy at the moment.
    return {'id': track_id, 'box_kf': new_box_kf, 'loc_kf': new_loc_kf, 'zs': [detected_box],
                        'box_xs': [new_box_kf.x], 'box_Ps': [new_box_kf.P], 'loc_xs': [new_loc_kf.x],
                        'loc_Ps': [new_loc_kf.P], 'frames_since_last_detection': 0}

def xywh_to_uv(x1,y1,w,h):
    return [x1+w/2, y1+h]

def xywh_image_to_court(x, y, w, h, homog_transf):
    u = x + w/2
    b = y + h
    court_x, court_y = cv2.perspectiveTransform((np.array([u, b]).reshape(-1, 2)).astype(np.float32)[np.newaxis], homog_transf)[0][0]
    return court_x, court_y
