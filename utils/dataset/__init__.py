from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional

import numpy as np

from dataset.datacontainers import Camera


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


@dataclass
class Dataset:
    points3D: List[Point3D]
    points3D_mapped: Dict = field(init=False)
    datasetEntries: List[DatasetEntry]

    def __post_init__(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}

    def refresh_mapping(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}
