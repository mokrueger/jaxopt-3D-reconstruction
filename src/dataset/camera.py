from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np

from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import TransformationDirection
from src.dataset.point import Point3D


@dataclass
class CameraIntrinsics:
    camera_intrinsics_matrix: Union[np.ndarray, None]
    focal_length: Union[float, Tuple[float, float], None]
    skew_factor: Union[float, None]
    center_x: Union[float, None]
    center_y: Union[float, None]

    @property
    def focal_x(self):
        return self.focal_length if type(self.focal_length) == float else self.focal_length[0]

    @property
    def focal_y(self):
        return self.focal_length if type(self.focal_length) == float else self.focal_length[1]

    def apply_noise(self, noise_matrix, masked=True):
        if self.camera_intrinsics_matrix is not None:
            if masked:
                self.camera_intrinsics_matrix[0, 0] += noise_matrix[0, 0]  # fx
                self.camera_intrinsics_matrix[1, 1] += noise_matrix[1, 1]  # fy
                self.camera_intrinsics_matrix[0, 2] += noise_matrix[0, 2]  # cx
                self.camera_intrinsics_matrix[1, 2] += noise_matrix[1, 2]  # cy
                self.camera_intrinsics_matrix[0, 1] += noise_matrix[0, 1]  # skew
            else:
                self.camera_intrinsics_matrix += noise_matrix
            if type(self.focal_length) == float:
                self.focal_length += noise_matrix[0, 0]
            else:
                self.focal_length = (self.focal_length[0] + noise_matrix[0, 0],
                                     self.focal_length[1] + noise_matrix[1, 1])
            self.skew_factor += noise_matrix[0, 1]
            self.center_x += noise_matrix[0, 2]
            self.center_y += noise_matrix[1, 2]


@dataclass
class Camera:
    camera_pose: CameraPose
    camera_intrinsics: Union[CameraIntrinsics, None]
    width: int
    height: int

    def project(self, point3D: Union[Point3D, np.ndarray]):
        xyz = point3D.xyz if type(point3D) == Point3D else point3D
        p = self.camera_intrinsics.camera_intrinsics_matrix.dot(
            self.camera_pose.in_direction(TransformationDirection.W2C).transformation_translation_matrix.dot(
                np.array([*xyz, 1])
            )[0:3]
        )
        return np.array([p[0] / p[2], p[1] / p[2]])

    def compute_inlier_mask(self, p2d, p3d, max_error):
        return [np.linalg.norm(p2 - self.project(p3)) <= max_error for p2, p3 in zip(p2d, p3d)]

    def compute_inlier_mask_mod(self, p2d, p3d):
        errors = [np.linalg.norm(p2 - self.project(p3)) for p2, p3 in zip(p2d, p3d)]
        max_error = np.median(errors)
        return [np.linalg.norm(p2 - self.project(p3)) <= max_error for p2, p3 in zip(p2d, p3d)]

    def compute_projection_errors(self, p2d, p3d):  # TODO: this is technically duplicate code
        return [np.linalg.norm(p2 - self.project(p3)) for p2, p3 in zip(p2d, p3d)]
