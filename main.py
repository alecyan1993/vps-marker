from util.model import superpoint, superglue
from util.feature_matcher import FeatureMatcher
from util.pnp_solver import PnpSolver
from query import QueryImage
import torch
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from marker import Marker


def get_marker(fp):
    mk = np.load(fp, allow_pickle=True).item()
    return mk


def visualize_superglue(sg_res, query_img, marker_img, match_threshold=0.3):
    h = max(query_img.shape[0], marker_img.shape[0])
    canvas = np.ones((h, query_img.shape[1] + marker_img.shape[1], 3)) * 255
    canvas[0:marker_img.shape[0], 0:marker_img.shape[1]] = marker_img
    canvas[0:query_img.shape[0], marker_img.shape[1]:] = query_img
    canvas = canvas.astype(np.uint8)

    kpts0 = sg_res['keypoints0'].squeeze().cpu().detach().numpy()
    kpts1 = sg_res['keypoints1'].squeeze().cpu().detach().numpy()
    matches0 = sg_res['matches0'].squeeze().cpu().detach().numpy()
    matches1 = sg_res['matches1'].squeeze().cpu().detach().numpy()
    matching_scores0 = sg_res['matching_scores0'].squeeze().cpu().detach().numpy()
    pointsid1 = matches0[matching_scores0 > match_threshold]
    kpts0_filter = [kpts0[matches1[i]] for i in pointsid1]
    kpts1_filter = [kpts1[i] for i in pointsid1]

    for i in range(len(kpts0_filter)):
        kpt0 = kpts0_filter[i]
        kpt1 = kpts1_filter[i]
        cv2.circle(canvas, kpt1.astype(int), 6, (0, 0, 255), -1)
        cv2.circle(canvas, (int(kpt0[0] + marker_img.shape[1]), int(kpt0[1])), 6, (0, 0, 255), -1)
        cv2.line(canvas, kpt1.astype(int), (int(kpt0[0] + marker_img.shape[1]), int(kpt0[1])), (0, 0, 255), 1)
    cv2.putText(img=canvas, text=f'Number of Matched Feature Points: {len(kpts0_filter)}', org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(0, 255, 0), thickness=2)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)


def get_pairs(sg_res, origin_features, match_threshold=0.6):
    kpts0 = sg_res['keypoints0'].squeeze().cpu().detach().numpy()
    kpts1 = sg_res['keypoints1'].squeeze().cpu().detach().numpy()
    matches0 = sg_res['matches0'].squeeze().cpu().detach().numpy()
    matches1 = sg_res['matches1'].squeeze().cpu().detach().numpy()
    matching_scores0 = sg_res['matching_scores0'].squeeze().cpu().detach().numpy()
    kpts1_3d = origin_features['keypoints_3d']

    pointsid1 = matches0[matching_scores0 > match_threshold]
    kpts0_filter = [list(kpts0[matches1[i]].astype(int)) for i in pointsid1]
    kpts1_filter = [kpts1[i] for i in pointsid1]
    kpts1_3d_filter = [list(kpts1_3d[i]) for i in pointsid1]
    return kpts0_filter, kpts1_filter, kpts1_3d_filter


if __name__ == '__main__':
    device = 'gpu:0' if torch.cuda.is_available() else 'cpu:0'
    sp = superpoint.SuperPoint({}).eval().to(device)
    sg = superglue.SuperGlue({}).eval().to(device)
    feature_matcher = FeatureMatcher(sp, sg)

    marker_features = get_marker('marker/features/cave2.npy')
    marker = cv2.imread('marker/images/cave2.jpeg')

    img = cv2.imread('query/images/query_cave2.png')
    query = {'image': img,
             'translation': [0, 0, 0],
             'rotation': [0, 0, 0],
             'camera_intrinsic': np.array([[802, 0, 468.46], [0, 802, 359.71], [0, 0, 1]])}
    qi = QueryImage(query)
    query_features = qi.get_feature(sp)

    match = feature_matcher.match_superglue(query_features, marker_features)
    # visualize_superglue(match, img, marker, match_threshold=0.6)
    kp_query_pixel, kp_marker_pixel, kp_query_3d = get_pairs(match, marker_features, match_threshold=0.6)

    # Pnp Solver
    pnp_solver = PnpSolver(min_points3d=50)
    pose_guess = {'position': [10, 10, 2],
                  'quaternion': [-0.3154602271415246, -0.40056517567031236, 0.6800333831230863, -0.5268652417556118]}
    pnp_pose, flag = pnp_solver.solve_pnp(kp_query_pixel, kp_query_3d, query['camera_intrinsic'], pose_guess)




