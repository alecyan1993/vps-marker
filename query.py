"""
* Runtime
- Image transform for input image (resize, tensor...)
- SuperPoint to extract features for input images
- SuperGlue to match features between query image and marker candidates, generate 2d-3d pairs for input image
- PnP solver to calculate T_world_camera for the query image
- Coordinate transform for T_physical_ar
"""

from util.model import superpoint, superglue
import torch
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


class QueryImage:
    def __init__(self, query):
        self.features = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        self.img_color = query['image']
        self.img = cv2.cvtColor(query['image'], cv2.COLOR_RGB2GRAY)
        self.trans = query['translation']
        self.rot = query['rotation']
        self.k = query['camera_intrinsic']

    def get_feature(self, sp):
        img_tensor = torch.from_numpy(self.img / 255.0).float()[None, None].to(self.device)
        sp_res = sp({'image': img_tensor})
        sp_res['image_size'] = torch.tensor(self.img.shape)
        self.features = sp_res
        return sp_res

    def visualize_features(self):
        kps = self.features['keypoints'][0].cpu().detach().numpy().astype(int)
        img_color = self.img_color.copy()
        for kp in kps:
            cv2.circle(img_color, kp, 6, (0, 0, 255), -1)
        cv2.imshow('Marker Features', img_color)
        cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('query/images/query_cave2.png')
    query = {'image': img,
             'translation': [0, 0, 0],
             'rotation': [0, 0, 0],
             'camera_intrinsic': np.array([[802, 0, 468.46], [0, 802, 359.71], [0, 0, 1]])}
    qi = QueryImage(query)
    features = qi.get_feature()
    qi.visualize_features()




