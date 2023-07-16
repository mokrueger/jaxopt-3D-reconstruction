import os
import struct
from dataclasses import dataclass
from typing import List

from src.config import DATASETS_PATH
from src.dataset.loaders.colmap_dataset_loader.read_write_model import \
    read_points3D_text


@dataclass
class PointInformation:
    point3d_id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float


@dataclass
class TrackEntry:
    image_id: int
    point2d_idx: int


@dataclass
class Point:
    point_information: PointInformation
    track_entries: List[TrackEntry]


def read_points3d_bin(file):
    points = []
    with open(file, "rb") as f:
        num_points = struct.unpack_from("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3d_id, x, y, z, r, g, b, error = struct.unpack_from("<Q3d3Bd", f.read(43))
            point_information = PointInformation(point3d_id, x, y, z, r, g, b, error)
            track_length = struct.unpack_from("<Q", f.read(8))[0]
            track_entries = []
            for _ in range(track_length):
                track_entries.append(TrackEntry(*struct.unpack_from("<2I", f.read(8))))
            points.append(Point(point_information, track_entries))
    return {pp.point_information.point3d_id: pp for pp in points}


def read_points3d_txt(file):
    points = []
    po = read_points3D_text(file)
    for p in po.values():
        point_information = PointInformation(p.id, p.xyz[0], p.xyz[1], p.xyz[2], p.rgb[0], p.rgb[1], p.rgb[2], p.error)
        track_entries = []
        for index, point2d_idx in enumerate(start=0, iterable=p.point2D_idxs):
            track_entries.append(TrackEntry(p.image_ids[index], point2d_idx))
        points.append(Point(point_information, track_entries))
    return {pp.point_information.point3d_id: pp for pp in points}


if __name__ == "__main__":
    pt1 = read_points3d_bin(os.path.join(DATASETS_PATH, "reichstag/sparse" + "/points3D.bin"))
    pt2 = read_points3d_txt(os.path.join(DATASETS_PATH, "reichstag/sparse/TXT" + "/points3D.txt"))
