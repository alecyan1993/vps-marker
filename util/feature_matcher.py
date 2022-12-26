import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import cv2
import torch

"""
Feature Matcher
- Use SuperPoint Model to Extract Feature Points from the Image
- Use SuperGlue Model to Match the Two Images
"""


class FeatureMatcher:

    def __init__(self, superpoint_model, superglue_model):
        self.processer = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        self.superpoint_model = superpoint_model
        self.superglue_model = superglue_model

    def get_superpoint(self, img, resize_ratio=0.5):
        img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
        img_tensor = torch.from_numpy(img / 255.0).float()[None, None].to(self.processer)
        superpoint_res = self.superpoint_model({'image': img_tensor})
        superpoint_res['image_size'] = torch.tensor(img.shape)
        return superpoint_res

    def convertmap2superpoint(self, data):
        data['keypoints'] = torch.FloatTensor(data['keypoints'])[None]
        data['descriptors'] = torch.FloatTensor(data['descriptors'])[None]
        data['scores'] = torch.FloatTensor(data['scores'])[None]
        data['image_size'] = torch.tensor(data['image_size'])

    @torch.no_grad()
    def match_superglue(self, query, origin):
        pred = {}

        pred['image0'] = query['image_size']
        pred['keypoints0'] = query['keypoints']
        pred['descriptors0'] = query['descriptors']
        pred['scores0'] = query['scores']

        pred['image1'] = origin['image_size']
        pred['keypoints1'] = origin['keypoints']
        pred['descriptors1'] = origin['descriptors']
        pred['scores1'] = origin['scores']

        for k in pred:
            if isinstance(pred[k], (list, tuple)):
                pred[k] = torch.stack(pred[k])
            elif isinstance(pred[k], (np.ndarray)):
                pred[k] = torch.from_numpy(pred[k])
            pred[k] = pred[k].to(self.processer)

        pred = {**pred, **self.superglue_model(pred)}
        return pred

    @staticmethod
    def get_superpoint_candidates(candidate_list, visual_map):
        superpoint_candidates = []
        for candidate in candidate_list:
            superpoint_candidates.append(visual_map[candidate])
        return superpoint_candidates

    @torch.no_grad()
    def extract_allcloudpoint(self, superpoint_query, superpoint_candidates, match_threshold=0.3):
        points_pixel = []
        points_3d = []
        for superpoint_candidate in superpoint_candidates:
            superglue_result = self.match_superglue(superpoint_query, superpoint_candidate)
            kpts0 = superglue_result['keypoints0'].squeeze()
            matches0 = superglue_result['matches0'].squeeze()
            matches1 = superglue_result['matches1'].squeeze()
            matching_score0 = superglue_result['matching_scores0'].squeeze()
            kp_pose1 = superpoint_candidate['kp_pose']
            # Filter Valid Points ID
            pointsid = matches0[matching_score0 > match_threshold]
            pointsid = pointsid[pointsid > -1]
            # Extract Matched Points' 3D Position and Pixel Position
            if len(pointsid) > 0:
                for id in pointsid.cpu():
                    if kp_pose1[id][0] is not None and list(kpts0[matches1[id]].numpy()) not in points_pixel:
                            points_3d.append(list(kp_pose1[id]))
                            points_pixel.append(list(kpts0[matches1[id]].numpy()))

        points_pixel = np.array(points_pixel, dtype=np.double)
        points_3d = np.array(points_3d, dtype=np.double)
        return points_pixel, points_3d

    @staticmethod
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
        cv2.imshow('canvas', canvas)
        cv2.waitKey(0)




