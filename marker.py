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


class Marker:
    def __init__(self, config):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sp = superpoint.SuperPoint({}).eval().to(self.device)
        self.marker_color = cv2.imread(self.config['filepath'])
        self.marker = cv2.imread(self.config['filepath'], 0)

    def get_features(self):
        marker_tensor = torch.from_numpy(self.marker / 255.0).float()[None, None].to(self.device)
        sp_res = self.sp({'image': marker_tensor})
        sp_res['image_size'] = torch.tensor(marker_tensor.shape)
        return sp_res

    def visualize_features(self):
        pass


if __name__ == '__main__':
    marker = Marker('config/marker.yml')
    sp_res = marker.get_features()
    print(sp_res)







