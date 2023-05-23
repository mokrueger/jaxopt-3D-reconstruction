import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from dataset.datacontainers import Camera

SEED = 0
np.random.seed(SEED)


# (random) TODO: perhaps move all of classes in __init__ files into own files for better overview
@dataclass
class ImageMetadata:
    identifier: str
    image_path: Optional[str]
    width: int
    height: int


@dataclass
class Point2D:
    identifier: int
    x: float
    y: float
    point3D_identifier: Union[int, None]
    metadata: Dict

    @property
    def xy(self):
        return np.array([self.x, self.y])


@dataclass
class Point3D:
    identifier: int
    x: float
    y: float
    z: float
    metadata: Dict

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    def translate(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z

    def translated(self, x, y, z):
        new = copy.deepcopy(self)
        new.translate(x, y, z)
        return new

    def translate_np(self, xyz):
        self.x += xyz[0]
        self.y += xyz[1]
        self.z += xyz[2]

    def translated_np(self, xyz):
        new = copy.deepcopy(self)
        new.translate_np(xyz)
        return new


@dataclass
class DatasetEntry:
    image_metadata: ImageMetadata
    points2D: List[Point2D]
    points2D_mapped: Dict = field(init=False)
    camera: Camera

    def __post_init__(self):
        self.points2D_mapped = {p.identifier: p for p in self.points2D}

    def points_with_3d(self):
        return list(filter(lambda x: x.point3D_identifier, self.points2D))

    def refresh_mapping(self):
        self.points2D_mapped = {p.identifier: p for p in self.points2D}

    def map2d_3d(self, points3D_mapped, zipped=True):
        if zipped:
            return [(p, points3D_mapped.get(p.point3D_identifier)) for p in self.points_with_3d()]
        points_with_3d = self.points_with_3d()
        return points_with_3d, [points3D_mapped.get(p.point3D_identifier) for p in points_with_3d]

    def map2d_3d_np(self, points3D_mapped, zipped=True):
        if zipped:
            return list(
                map(lambda p2d_p3d: (p2d_p3d[0].xy, p2d_p3d[1].xyz), self.map2d_3d(points3D_mapped, zipped=zipped))
            )
        p2d, p3d = self.map2d_3d(points3D_mapped, zipped=zipped)
        return list(map(lambda p: p.xy, p2d)), list(map(lambda p: p.xyz, p3d))

    @property
    def num_3d_points(self):
        return len(self.points_with_3d())

    @property
    def num_2d_points(self):
        return len(self.points2D)


@dataclass
class Dataset:
    points3D: List[Point3D]
    points3D_mapped: Dict = field(init=False)
    datasetEntries: List[DatasetEntry]

    def __post_init__(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}

    def refresh_mapping(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}

    @staticmethod
    def _random_direction():  # TODO: helper methods
        r = np.random.rand(3)
        r /= np.linalg.norm(r)
        return r

    @staticmethod
    def with_noise(dataset: "Dataset", point3d_noise=3e-2, camera_rotation_noise=5e-2, camera_translation_noise=5e-2,
                   camera_intrinsics_noise=10):
        new_dataset = copy.deepcopy(dataset)  # TODO: this is not performant
        for p in new_dataset.points3D:
            p.translate_np(Dataset._random_direction() * point3d_noise)  # * np.random.randn(3))
        for d in new_dataset.datasetEntries:
            d.camera.camera_pose.apply_move(  # TODO: check which distributions to use for random
                Dataset._random_direction() * camera_translation_noise  # * np.random.randn(3)
            )
            d.camera.camera_pose.apply_transform_3d(
                Rotation.from_rotvec(
                    Dataset._random_direction() * camera_rotation_noise
                ).as_matrix()  # * np.random.randn(3)
            )
            d.camera.camera_intrinsics.apply_noise(np.random.rand(3, 3) * camera_intrinsics_noise)
            new_dataset.refresh_mapping()
        return new_dataset

    #  @property
    def num_3d_points(self):
        return len(self.points3D)

    #  @property
    def num_images(self):
        return len(self.datasetEntries)

    #  @property
    def avg_num_3d_points_per_image(self):  # TODO: avg or median(?)
        return np.average([de.num_3d_points for de in self.datasetEntries])

    #  @property
    def avg_num_2d_points_per_image(self):
        return np.average([de.num_2d_points for de in self.datasetEntries])
