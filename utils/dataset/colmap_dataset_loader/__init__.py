import os
from pathlib import Path

import numpy as np

from PIL import Image

from dataset import DatasetEntry, ImageMetadata, Point2D, Point3D, Camera, Dataset
from dataset.colmap_dataset_loader.cameras import read_cameras_bin, read_cameras_txt
from dataset.colmap_dataset_loader.images import read_images_bin, read_images_txt
from dataset.colmap_dataset_loader.points import read_points3d_bin, read_points3d_txt
from dataset.datacontainers import CameraIntrinsics
from dataset.datacontainers.camera_pose import CoordinateSystem, TransformationDirection
from dataset.datacontainers.camera_pose.camera_pose import CameraPose


def params_to_intrinsics(fx, fy, cx, cy, s=None):
    return CameraIntrinsics(camera_intrinsics_matrix=np.array([[fx, s if s else 0, cx], [0, fy, cy], [0, 0, 1]]),
                            focal_length=(fx, fy), skew_factor=s if s else 0, center_x=cx, center_y=cy)


def get_intrinsics(camera):
    pars = camera.params
    if camera.camera_model_type.name == 'SIMPLE_RADIAL':
        return CameraIntrinsics(camera_intrinsics_matrix=np.array([
            [pars[0], 0, pars[1]],
            [0, pars[0], pars[2]],
            [0, 0, 1],
        ]), focal_length=pars[0], skew_factor=0, center_x=pars[1], center_y=pars[2])
    else:
        return params_to_intrinsics(*pars)


def _get_image_width_height(image_path):
    im = Image.open(image_path)
    width, height = im.size
    return width, height


def load_colmap_dataset(path_to_sparse_folder, path_to_images):
    files = os.listdir(path_to_sparse_folder)
    file_format = Path(files[0]).suffix
    if file_format.lower() == ".bin":
        points = read_points3d_bin(path_to_sparse_folder + "/points3D.bin")
        images = read_images_bin(path_to_sparse_folder + "/images.bin")
        cameras = read_cameras_bin(path_to_sparse_folder + "/cameras.bin")
    else:
        points = read_points3d_txt(path_to_sparse_folder + "/points3D.txt")
        images = read_images_txt(path_to_sparse_folder + "/images.txt")
        cameras = read_cameras_txt(path_to_sparse_folder + "/cameras.txt")

    points3D = list(map(lambda p: Point3D(
        p.point_information.point3d_id,
        p.point_information.x,
        p.point_information.y,
        p.point_information.z,
        metadata={
            "rgb": np.array([p.point_information.r, p.point_information.g, p.point_information.b]),
            "error": p.point_information.error,
            "track_entries": p.track_entries
        }
    ), points.values()))

    datasetEntries = []
    for im in images.values():
        image_path = os.path.join(path_to_images, im.image_information.name)
        width, height = _get_image_width_height(image_path)
        image_metadata = ImageMetadata(identifier=im.image_information.name,
                                       image_path=image_path,
                                       width=width,
                                       height=height)
        points2D = list(map(lambda p: Point2D(p.id, p.x, p.y, p.point3d_id, {}), im.point2d_entries))

        camera_pose = CameraPose.from_string_wxyz_quaternion_translation(f"{im.image_information.qw} "
                                                                         f"{im.image_information.qx} "
                                                                         f"{im.image_information.qy} "
                                                                         f"{im.image_information.qz} "
                                                                         f"{im.image_information.tx} "
                                                                         f"{im.image_information.ty} "
                                                                         f"{im.image_information.tz}",
                                                                         identifier=Path(
                                                                             im.image_information.name).name,
                                                                         coordinate_system=CoordinateSystem.COLMAP,
                                                                         direction=TransformationDirection.C2W
                                                                         )
        camera_intrinsics = get_intrinsics(cameras.get(im.image_information.camera_id))
        camera = Camera(camera_pose=camera_pose,
                        camera_intrinsics=camera_intrinsics,
                        width=width, height=height)
        datasetEntries.append(DatasetEntry(image_metadata, points2D, camera))
    return Dataset(points3D, datasetEntries)


if __name__ == "__main__":
    path = "/home/morkru/Downloads/reichstag/dense/sparse/"
    image_path = "/home/morkru/Downloads/reichstag/dense/images"
    dataset = load_colmap_dataset(path, image_path)
