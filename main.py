from util.model import superpoint, superglue
from util.feature_matcher import FeatureMatcher
from query import QueryImage
import torch
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_marker(fp):
    mk = np.load(fp, allow_pickle=True).item()
    return mk


if __name__ == '__main__':
    device = 'gpu:0' if torch.cuda.is_available() else 'cpu:0'
    sp = superpoint.SuperPoint({}).eval().to(device)
    sg = superglue.SuperGlue({}).eval().to(device)
    feature_matcher = FeatureMatcher(sp, sg)

    marker_features = get_marker('marker/features/cave2.npy')

    img = cv2.imread('query/images/query_cave2.png')
    query = {'image': img,
             'translation': [0, 0, 0],
             'rotation': [0, 0, 0],
             'camera_intrinsic': np.array([[802, 0, 468.46], [0, 802, 359.71], [0, 0, 1]])}
    qi = QueryImage(query)
    query_features = qi.get_feature(sp)

    match = feature_matcher.match_superglue(query_features, marker_features)
    print(match.keys())


