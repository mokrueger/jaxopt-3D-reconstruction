from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from dataset.datacontainers.camera_pose import CameraPose


@dataclass
class CameraIntrinsics:
    camera_intrinsics_matrix: Union[np.ndarray, None]
    focal_length: Union[float, Tuple[float, float]]
    skew_factor: Union[float, None]
    center_x: Union[float, None]
    center_y: Union[float, None]

    @property
    def focal_x(self):
        return self.focal_length if type(self.focal_length) == float else self.focal_length[0]

    @property
    def focal_y(self):
        return self.focal_length if type(self.focal_length) == float else self.focal_length[1]


@dataclass
class Camera:
    camera_pose: CameraPose
    camera_intrinsics: Union[CameraIntrinsics, None]
    width: int
    height: int
