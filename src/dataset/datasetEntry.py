from dataclasses import dataclass, field
from typing import List, Dict

from src.dataset.camera import Camera
from src.dataset.imageMetadata import ImageMetadata
from src.dataset.point import Point2D


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

    def refresh_mapping(self):  # TODO: Technically using @property would be better but slows down debugging
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

    @property
    def identifier(self):
        return self.image_metadata.identifier
