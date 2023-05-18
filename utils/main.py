from dataset.colmap_dataset_loader.points import read_points3d_bin
from dataset.colmap_dataset_loader.images import read_images_bin
from dataset.colmap_dataset_loader.cameras import read_cameras_bin

import pycolmap
import numpy as np
from pycolmap import AbsolutePoseEstimationOptions, AbsolutePoseRefinementOptions


def map_2d_3d(images_dict, points_3d_dict):
    def _to_numpy(p):
        if hasattr(p, "z"):
            return np.array([p.x, p.y, p.z])
        return np.array([p.x, p.y])

    id_to_2d = {k: [pe for pe in v.point2d_entries if pe.point3d_id] for k, v in images_dict.items()}
    return {k: ([_to_numpy(vv) for vv in v], [_to_numpy(points_3d_dict.get(vv.point3d_id)) for vv in v]) for k, v in
            id_to_2d.items()}


if __name__ == "__main__":
    points = read_points3d_bin("/home/morkru/Downloads/reichstag/dense/sparse" + "/points3D.bin")
    images = read_images_bin("/home/morkru/Downloads/reichstag/dense/sparse" + "/images.bin")
    cameras = read_cameras_bin("/home/morkru/Downloads/reichstag/dense/sparse" + "/cameras.bin")
    points_3d = [p.point_information for p in points]

    points_3d_dict = {p.point3d_id: p for p in points_3d}
    images_dict = {i.image_information.image_id: i for i in images}
    cameras_dict = {c.camera_id: c for c in cameras}

    # List[numpy.ndarray[numpy.float64[2, 1]]], points3D: List[numpy.ndarray[numpy.float64[3, 1]]], camera: colmap::Camera, max_error_px: float = 12.0, min_inlier_ratio: float = 0.01, min_num_trials: int = 1000, max_num_trials: int = 100000, confidence: float = 0.9999, return_covariance: bool = False)
    x = map_2d_3d(images_dict, points_3d_dict)
    image_id = 1

    c1 = pycolmap.Camera()
    focal_length = 1.2 * max(1048, 628)
    c1.initialize_with_name("SIMPLE_RADIAL", focal_length, 1048, 628)
    p1 = x.get(1)

    ae = AbsolutePoseEstimationOptions()
    ae.estimate_focal_length = True
    ar = AbsolutePoseRefinementOptions()
    ar.refine_extra_params = True
    ar.refine_focal_length = True
    ar.print_summary = True
    x = pycolmap.absolute_pose_estimation(p1[0], p1[1], c1, ae, ar)
    print("hi")
