import sys
import os
import time

src_path = "/mnt/c/Users/ktyyl/cop/jaxopt-3D-reconstruction/"
if src_path not in sys.path:
    sys.path.append(src_path)

from src.benchmark.benchmark import Benchmark

from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.dataset.dataset import Dataset

from src.reconstruction.bundle_adjustment.optimization import JaxBundleAdjustment


import numpy as np


class JaxoptBenchmark(Benchmark):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.optimizer = JaxBundleAdjustment()

        (
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
        ) = self._prepare_dataset()

        self.cam_poses_gpu = JaxBundleAdjustment.to_gpu(self.cam_poses)
        self.intrinsics_gpu = JaxBundleAdjustment.to_gpu(self.intrinsics)
        self.points_gpu = JaxBundleAdjustment.to_gpu(self.points)
        self.observations_gpu = JaxBundleAdjustment.to_gpu(self.observations)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []
        points_3d = []
        points_2d = []

        for index, e in enumerate(self.dataset.datasetEntries):
            cam = e.camera

            mapped_points = e.map2d_3d_np(self.dataset.points3D_mapped, zipped=False)
            p_2d, p_3d = [np.array(l) for l in mapped_points]
            p_3d = np.concatenate([p_3d, np.ones((p_3d.shape[0], 1))], axis=1)

            points_2d.append(p_2d)
            points_3d.append(p_3d)

            cam_poses.append(cam.camera_pose.rotation_translation_matrix)
            intrinsics.append(cam.camera_intrinsics.camera_intrinsics_matrix)

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)

        return cam_poses, intrinsics, points_3d, points_2d

    def compile(self, index: int) -> None:
        self.optimizer.compile_pose_opt(len(self.cam_poses), 1400, (1600, 2))

    def optimize(
        self, index: int, initial_pose: np.array, initial_intrinsics: np.array
    ):
        return self.optimizer.run_pose_opt(
            initial_pose,
            initial_intrinsics,
            self.points_gpu[index],
            self.observations_gpu[index],
        )


from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    ds_path = os.path.join(src_path, "dataset", "reichstag", "dense")

    config = {
        "add_noise": True,
        "path": os.path.join(ds_path, "sparse"),
        "image_path": os.path.join(ds_path, "images"),
    }

    ds = load_colmap_dataset(config["path"], config["image_path"], binary=True)
    ds = Dataset.with_noise(ds) if config["add_noise"] else ds

    jaxopt_benchmark = JaxoptBenchmark(ds)

    camera_index = 1

    print("=== compilation ===")
    start = time.process_time()
    jaxopt_benchmark.compile(camera_index)
    print("compilation time:", time.process_time() - start, "s")

    # # fx fy cx cy skew
    # intr = jaxopt_benchmark.intrinsics[camera_index]
    # initial_intrinsics = [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], intr[0, 1]]

    # print("=== optimization ===")
    # start = time.process_time()
    # params, state = jaxopt_benchmark.optimize(
    #     camera_index,
    #     jaxopt_benchmark.cam_poses_gpu[camera_index + 3],
    #     initial_intrinsics,
    # )
    # print("optimization time:", time.process_time() - start, "s")

    # print("Loss:", state.loss, "in", state.iter_num, "iterations")

    # print(
    #     "Pose error:",
    #     JaxPoseOptimizer.pose_mat_to_vec(jaxopt_benchmark.cam_poses[camera_index])
    #     - np.array(params[:6]),
    # )

    # print(
    #     "Intrinsics error:",
    #     np.array(initial_intrinsics) - np.array(params[6:]),
    # )
