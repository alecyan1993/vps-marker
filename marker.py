"""
* Preprocess
- Marker or marker image candidate with enough feature points (this case a "cave" image)
- SuperPoint to extract and store the features for the marker candidate
- Based on marker physical location config, calculate 3d position of marker candidate's feature points
"""

from util.model import superpoint
import torch
import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R


def from_Rt2T(rotation_matrix, translation_vector):
    translation_vector = np.array(translation_vector).reshape(3, 1)
    T = np.concatenate((rotation_matrix, translation_vector), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    return T


class Marker:
    def __init__(self, config):
        self.features = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sp = superpoint.SuperPoint({}).eval().to(self.device)
        self.marker_color = cv2.imread(self.config['file_path'])
        self.marker = cv2.imread(self.config['file_path'], 0)

    def get_features(self):
        marker_tensor = torch.from_numpy(self.marker / 255.0).float()[None, None].to(self.device)
        sp_res = self.sp({'image': marker_tensor})
        sp_res['image_size'] = torch.tensor(marker_tensor.shape)
        kp_3d = self.convert_2d_3d(sp_res)
        sp_res['keypoints_3d'] = kp_3d
        self.features = sp_res
        return sp_res

    def convert_2d_3d(self, sp_res):
        kp_3d = []
        r = R.from_euler('xyz', [self.config['rotation']['x'] / 180 * np.pi,
                                 self.config['rotation']['y'] / 180 * np.pi,
                                 self.config['rotation']['z'] / 180 * np.pi])
        rot_mat = r.as_matrix()
        translation = [self.config['position']['x'],
                       self.config['position']['y'],
                       self.config['position']['z']]
        T = from_Rt2T(rot_mat, translation)
        for kp in sp_res['keypoints'][0].cpu().detach().numpy().astype(int):
            trans = [kp[0] * self.config['size']['length'] / self.marker.shape[1],
                     kp[1] * self.config['size']['length'] / self.marker.shape[1],
                     0, 1]
            t = T @ np.array(trans)
            kp_3d.append(list(t)[:3])
        return np.array(kp_3d)

    def visualize_features(self):
        kps = self.features['keypoints'][0].cpu().detach().numpy().astype(int)
        marker_color = self.marker_color.copy()
        for kp in kps:
            cv2.circle(marker_color, kp, 6, (0, 0, 255), -1)
        cv2.imshow('Marker Features', marker_color)
        cv2.waitKey(0)


def main():
    marker = Marker(config_path)
    sp_res = marker.get_features()
    if marker.config['save']:
        np.save(marker.config['feature_path'], sp_res)
    if marker.config['visualize']:
        marker.visualize_features()


if __name__ == '__main__':
    config_path = 'config/marker.yml'
    main()






