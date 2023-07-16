import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import List

from src.config import DATASETS_PATH
from src.dataset.loaders.colmap_dataset_loader.read_write_model import \
    read_cameras_text


@dataclass
class Camera:
    camera_id: int
    camera_model_type: "CameraModelType"
    width: int
    height: int
    params: List


class CameraModelType(Enum):
    SIMPLE_PINHOLE = 0
    PINHOLE = 1
    SIMPLE_RADIAL = 2
    RADIAL = 3
    OPENCV = 4
    OPENCV_FISHEYE = 5
    FULL_OPENCV = 6
    FOV = 7
    SIMPLE_RADIAL_FISHEYE = 8
    RADIAL_FISHEYE = 9
    THIN_PRISM_FISHEYE = 10


class CameraModelParamNums(Enum):
    SIMPLE_PINHOLE = 3
    PINHOLE = 4
    SIMPLE_RADIAL = 4
    RADIAL = 5
    OPENCV = 8
    OPENCV_FISHEYE = 8
    FULL_OPENCV = 12
    FOV = 5
    SIMPLE_RADIAL_FISHEYE = 4
    RADIAL_FISHEYE = 5
    THIN_PRISM_FISHEYE = 12


def read_cameras_bin(file):
    cameras = []
    with open(file, "rb") as f:
        num_cameras = struct.unpack_from("<L", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, camera_model_type, width, height = struct.unpack_from("<IiQQ", f.read(24))
            num_camera_params = CameraModelParamNums[CameraModelType(camera_model_type).name].value
            params = list(struct.unpack_from(f"<{num_camera_params}d", f.read(num_camera_params * 8)))
            cameras.append(Camera(camera_id, CameraModelType(camera_model_type), width, height, params))
    return {cc.camera_id: cc for cc in cameras}


def read_cameras_txt(file):
    c = read_cameras_text(file)
    c = list(map(lambda x: Camera(x.id, CameraModelType[x.model], x.width, x.height, x.params), c.values()))
    return {cc.camera_id: cc for cc in c}


if __name__ == "__main__":
    c1 = read_cameras_txt(os.path.join(DATASETS_PATH, "reichstag/sparse/TXT" + "/cameras.txt"))
    c2 = read_cameras_bin(os.path.join(DATASETS_PATH, "reichstag/sparse" + "/cameras.bin"))
