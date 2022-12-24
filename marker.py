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


class Marker:
    def __init__(self, config):
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
        self.features = sp_res
        if self.config['save']:
            np.save(self.config['feature_path'], self.features)
        return sp_res

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






