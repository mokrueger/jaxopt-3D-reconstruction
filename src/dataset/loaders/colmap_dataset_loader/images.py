import struct
from dataclasses import dataclass
from typing import List, Union

from src.dataset.loaders.colmap_dataset_loader.read_write_model import read_images_text


@dataclass
class ImageInformation:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class Point2dEntry:
    id: int
    x: int
    y: int
    point3d_id: Union[int, None]


@dataclass
class Image:
    image_information: ImageInformation
    point2d_entries: List[Point2dEntry]


def read_images_bin(file):
    images = []
    with open(file, "rb") as f:
        num_images = struct.unpack_from("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack_from("<I7dI", f.read(64))
            name = ''.join(iter(lambda: f.read(1).decode('ascii'), '\x00'))
            image_information = ImageInformation(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)
            num_points = struct.unpack_from("<Q", f.read(8))[0]
            point2d_entries = []
            for i in range(num_points):
                x, y, point3d_id = struct.unpack_from("<2dQ", f.read(24))
                if point3d_id == 18446744073709551615:
                    point3d_id = None
                point2d_entries.append(Point2dEntry(i, x, y, point3d_id))
            images.append(Image(image_information, point2d_entries))
    return {i.image_information.image_id: i for i in images}


def read_images_txt(file):
    im = read_images_text(file)
    images = []
    for i in im.values():
        image_information = ImageInformation(i.id, i.qvec[0], i.qvec[1], i.qvec[2], i.qvec[3], i.tvec[0],
                                             i.tvec[1], i.tvec[2], i.camera_id, i.name)
        point2d_entries = []
        for index, p in enumerate(start=0, iterable=i.xys):
            point2d_entries.append(Point2dEntry(index, p[0], p[1], i.point3D_ids[index] if i.point3D_ids[index] != -1 else None))
        images.append(Image(image_information, point2d_entries))
    return {i.image_information.image_id: i for i in images}


if __name__ == "__main__":
    im1 = read_images_bin("/home/morkru/Downloads/reichstag/dense/sparse" + "/images.bin")
    im2 = read_images_txt("/home/morkru/Downloads/reichstag/dense/sparse/TXT" + "/images.txt")
