import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import cv2
from pyquaternion import Quaternion

"""
PnP Solver:
- Get the pose of the world origin in camera coordinate
"""


class PnpSolver:
    def __init__(self, min_points3d=50):
        self.min_points3d = min_points3d

    def solve_pnp(self, points_pixel, points_3d, camera_intrinsic_matrix, pose_guess):
        camera_intrinsic_matrix = np.array(camera_intrinsic_matrix).reshape((3, 3))
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        pnp_pose = {
            'translation': np.zeros((3,)),
            'inliers': None,
            'rotation_matrix': np.eye(4)
        }

        q_vec = Quaternion(pose_guess['quaternion'])
        rvec_guess = np.ones(shape=(3, 1))
        cv2.Rodrigues(q_vec.rotation_matrix, rvec_guess)
        tvec_guess = np.array(pose_guess['position'])

        if len(points_3d) > self.min_points3d:
            try:
                if np.linalg.norm(np.array(pose_guess['position'])) < 0.01:
                    success, rotation_vector, translation_vector, inliers=cv2.solvePnPRansac(
                        points_3d, points_pixel, camera_intrinsic_matrix, dist_coeffs,
                        reprojectionError=8, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    success, rotation_vector, translation_vector, inliers=cv2.solvePnPRansac(
                        points_3d, points_pixel, camera_intrinsic_matrix, dist_coeffs,
                        reprojectionError=8, flags=cv2.SOLVEPNP_ITERATIVE, rvec=rvec_guess,
                        tvec=tvec_guess, useExtrinsicGuess=True
                    )
            except:
                print("Error on Solving PNP")
            else:
                rotation_matrix = np.zeros(shape=(3, 3))
                cv2.Rodrigues(rotation_vector, rotation_matrix)
                pnp_pose = {
                    'translation': translation_vector,
                    'inliers': inliers,
                    'points3d': points_3d,
                    'rotation_matrix': rotation_matrix
                }
            return pnp_pose, True
        else:
            return pnp_pose, False



