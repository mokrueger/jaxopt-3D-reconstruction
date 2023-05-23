from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from dataset.datacontainers.camera_pose import CameraPose


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
