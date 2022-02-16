

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.stats import mahalanobis

def get_H(t, MATCH_ID, court_gt, std_court_points):
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