import os
import subprocess

import numpy as np
from pathlib import Path

from PIL import Image

from src.config import DATASETS_PATH
from src.dataset.camera import Camera, CameraIntrinsics
from src.dataset.dataset import Dataset
from src.dataset.datasetEntry import DatasetEntry
from src.dataset.imageMetadata import ImageMetadata
from src.dataset.point import Point2D, Point3D

from src.dataset.loaders.colmap_dataset_loader.cameras import read_cameras_bin, read_cameras_txt, CameraModelType
from src.dataset.loaders.colmap_dataset_loader.images import read_images_bin, read_images_txt
from src.dataset.loaders.colmap_dataset_loader.points import read_points3d_bin, read_points3d_txt

from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import CoordinateSystem, TransformationDirection


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


def _parse_points(points):
    return list(map(lambda p: Point3D(
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


def _parse_dataset_entries(images, cameras, path_to_images):
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
                                                                         direction=TransformationDirection.W2C
                                                                         # !!! W2C !!!
                                                                         )
        camera_intrinsics = get_intrinsics(cameras.get(im.image_information.camera_id))
        camera = Camera(camera_pose=camera_pose,
                        camera_intrinsics=camera_intrinsics,
                        width=width, height=height)
        datasetEntries.append(DatasetEntry(image_metadata, points2D, camera))
    return datasetEntries


def _parse_cameras_only(images, cameras, path_to_images):  # Note: this is mainly here to evaluate colmap benchmark
    parsed_cameras = {}
    for im in images.values():
        image_path = os.path.join(path_to_images, im.image_information.name)
        width, height = _get_image_width_height(image_path)
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
                                                                         direction=TransformationDirection.W2C
                                                                         # !!! W2C !!!
                                                                         )
        camera_intrinsics = get_intrinsics(cameras.get(im.image_information.camera_id))
        parsed_cameras.update(
            {
                im.image_information.image_id: Camera(camera_pose=camera_pose,
                                                      camera_intrinsics=camera_intrinsics,
                                                      width=width, height=height)
            }
        )
    return parsed_cameras


def load_colmap_cameras(path_to_sparse_folder, path_to_images, binary=False):
    if binary:
        images = read_images_bin(os.path.join(path_to_sparse_folder, "images.bin"))
        cameras = read_cameras_bin(os.path.join(path_to_sparse_folder, "cameras.bin"))
    else:
        images = read_images_txt(os.path.join(path_to_sparse_folder, "images.txt"))
        cameras = read_cameras_txt(os.path.join(path_to_sparse_folder, "cameras.txt"))

    parsed_cameras = _parse_cameras_only(images, cameras, path_to_images)
    return parsed_cameras


def load_colmap_dataset(path_to_sparse_folder, path_to_images, binary=False, name=None):
    if binary:
        points = read_points3d_bin(os.path.join(path_to_sparse_folder, "points3D.bin"))
        images = read_images_bin(os.path.join(path_to_sparse_folder, "images.bin"))
        cameras = read_cameras_bin(os.path.join(path_to_sparse_folder, "cameras.bin"))
    else:
        points = read_points3d_txt(os.path.join(path_to_sparse_folder, "points3D.txt"))
        images = read_images_txt(os.path.join(path_to_sparse_folder, "images.txt"))
        cameras = read_cameras_txt(os.path.join(path_to_sparse_folder, "cameras.txt"))

    points3D = _parse_points(points)
    datasetEntries = _parse_dataset_entries(images, cameras, path_to_images)

    return Dataset(points3D, datasetEntries, name=name)


def export_in_colmap_format(ds: Dataset, output_path, binary=False):
    from src.dataset.loaders.colmap_dataset_loader.read_write_model import Camera, BaseImage, Point3D, \
        write_cameras_binary, write_points3D_binary, write_images_binary, write_cameras_text, write_points3D_text, \
        write_images_text
    cameras = []
    base_images = []
    points3D = []

    os.makedirs(output_path, exist_ok=True)

    for index, d in enumerate(ds.datasetEntries, start=1):
        cameras.append(
            Camera(index,
                   model=CameraModelType.PINHOLE.name,  # TODO: maybe not always pinhole
                   width=d.camera.width,
                   height=d.camera.height,
                   params=[d.camera.camera_intrinsics.focal_x, d.camera.camera_intrinsics.focal_y,
                           d.camera.camera_intrinsics.center_x, d.camera.camera_intrinsics.center_y])
        )
        scipy_qvec = d.camera.camera_pose.in_direction(TransformationDirection.W2C).rotation.as_quat()
        base_images.append(
            BaseImage(index,
                      qvec=np.array([scipy_qvec[3], scipy_qvec[0], scipy_qvec[1], scipy_qvec[2]]),
                      tvec=d.camera.camera_pose.in_direction(TransformationDirection.W2C).translation,
                      camera_id=index,
                      name=d.image_metadata.identifier,
                      xys=np.array(list(map(lambda p: list(p.xy), d.points2D))),
                      point3D_ids=np.array(list(map(
                          lambda p: p.point3D_identifier if p.point3D_identifier is not None else -1,
                          d.points2D))
                      ))
        )
    img_id_point2d = [(i, p) for i, d in enumerate(ds.datasetEntries, start=1) for p in d.points_with_3d()]
    auxiliary_mapping = {}
    for img_id, point2 in img_id_point2d:
        if auxiliary_mapping.get(point2.point3D_identifier) is None:
            auxiliary_mapping[point2.point3D_identifier] = []
        auxiliary_mapping[point2.point3D_identifier].append((img_id, point2.identifier))
    for p in ds.points3D:
        image_ids, point2d_idxs = list(zip(*auxiliary_mapping.get(p.identifier)))
        points3D.append(
            Point3D(id=p.identifier,
                    xyz=p.xyz,
                    rgb=p.metadata.get("rgb") if p.metadata.get("rgb") is not None else np.array([255, 255, 255]),
                    error=p.metadata.get("error") if p.metadata.get("error") is not None else 999,
                    image_ids=np.array(image_ids),
                    point2D_idxs=np.array(point2d_idxs))
        )
    cameras = {c.id: c for c in cameras}
    base_images = {b.id: b for b in base_images}
    points3D = {p.id: p for p in points3D}
    if binary:
        write_cameras_binary(cameras, os.path.join(output_path, "cameras.bin"))
        write_images_binary(base_images, os.path.join(output_path, "images.bin"))
        write_points3D_binary(points3D, os.path.join(output_path, "points3D.bin"))
    else:
        write_cameras_text(cameras, os.path.join(output_path, "cameras.txt"))
        write_images_text(base_images, os.path.join(output_path, "images.txt"))
        write_points3D_text(points3D, os.path.join(output_path, "points3D.txt"))


# TODO: Decide where this goes
def show_in_colmap(sparse_path, image_path, database_path="/tmp/tmp.db", block=False):
    COLMAP_CMD = "colmap"
    if block:
        p = subprocess.run([COLMAP_CMD, "gui",
                            "--import_path", sparse_path,
                            "--database_path", database_path,
                            "--image_path", image_path,
                            ], stdout=subprocess.PIPE)
    else:
        subprocess.Popen([COLMAP_CMD, "gui",
                          "--import_path", sparse_path,
                          "--database_path", database_path,
                          "--image_path", image_path,
                          ], stdout=subprocess.PIPE)


if __name__ == "__main__":
    # path = "/home/morkru/Desktop/Github/jaxopt-3D-reconstruction/datasets/reichstag/sparse/"
    # image_path = "/home/morkru/Desktop/Github/jaxopt-3D-reconstruction/datasets/reichstag/images"
    # dataset = load_colmap_dataset(path, image_path)

    reichstag_sparse = os.path.join(DATASETS_PATH, "reichstag/sparse")
    reichstag_images = os.path.join(DATASETS_PATH, "reichstag/images")

    sacre_coeur_sparse = os.path.join(DATASETS_PATH, "sacre_coeur/sparse")
    sacre_coeur_images = os.path.join(DATASETS_PATH, "sacre_coeur/images")

    st_peters_square_sparse = os.path.join(DATASETS_PATH, "st_peters_square/sparse")
    st_peters_square_images = os.path.join(DATASETS_PATH, "st_peters_square/images")

    print("reichstag")
    ds = load_colmap_dataset(reichstag_sparse, reichstag_images, binary=True)
    ds = Dataset.with_noise_mp(ds)
    export_in_colmap_format(ds, os.path.join(DATASETS_PATH, "reichstag/sparse_noised"), binary=True)

    print("sacre")
    ds = load_colmap_dataset(sacre_coeur_sparse, sacre_coeur_images, binary=True)
    ds = Dataset.with_noise_mp(ds)
    export_in_colmap_format(ds, os.path.join(DATASETS_PATH, "sacre_coeur/sparse_noised"), binary=True)

    print("peter")
    ds = load_colmap_dataset(st_peters_square_sparse, st_peters_square_images, binary=True)
    ds = Dataset.with_noise_mp(ds)
    export_in_colmap_format(ds, os.path.join(DATASETS_PATH, "st_peters_square/sparse_noised"), binary=True)
