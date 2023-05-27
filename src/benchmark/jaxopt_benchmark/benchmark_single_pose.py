import sys
import os
import time

src_path = "/mnt/c/Users/ktyyl/cop/jaxopt-3D-reconstruction/"
if src_path not in sys.path:
    sys.path.append(src_path)

from src.benchmark.benchmark import Benchmark

from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.dataset.dataset import Dataset

from src.reconstruction.bundle_adjustment.pose_optimization import (
    JaxPoseOptimizer,
    get_reprojection_residuals_cpu,
)

import numpy as np

from triangulation_relaxations.se3 import Se3
from triangulation_relaxations.so3 import rotvec_to_r, r_to_rotvec


class JaxoptBenchmark(Benchmark):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.optimizer = JaxPoseOptimizer()

        (
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
        ) = self._prepare_dataset()

        self.cam_poses_gpu = JaxPoseOptimizer.to_gpu(self.cam_poses)
        self.intrinsics_gpu = JaxPoseOptimizer.to_gpu(self.intrinsics)
        self.points_gpu = JaxPoseOptimizer.to_gpu(self.points)
        self.observations_gpu = JaxPoseOptimizer.to_gpu(self.observations)

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

            cam_pose = np.identity(4)
            cam_pose[:3, :4] = cam.camera_pose.rotation_translation_matrix
            cam_poses.append(cam_pose)

            intr = cam.camera_intrinsics
            _intrinsics = np.identity(4)
            _intrinsics[:3, :3] = intr.camera_intrinsics_matrix
            intrinsics.append(_intrinsics)

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)

        return cam_poses, intrinsics, points_3d, points_2d

    def compile(self, index: int) -> None:
        self.optimizer.compile_pose_opt(
            self.points[index].shape,
            self.observations[index].shape,
            self.intrinsics[index].shape,
        )

    def optimize(self, index: int, initial_pose: np.array):
        return self.optimizer.run_pose_opt(
            initial_pose,
            self.points_gpu[index],
            self.observations_gpu[index],
            self.intrinsics_gpu[index],
        )


from scipy.spatial.transform import Rotation as R


def plot_costs(
    ax,
    pose0,
    pose1,
    points,
    observations,
    intrinsics,
    eps=0.1,
    n=1000,
    label0="",
    label1="",
):
    """Plot cost function when interpolating between pose0 and pose1"""
    taus = np.linspace(-eps, 1 + eps, n)
    index_0, index_1 = np.searchsorted(taus, [0, 1])
    taus = np.insert(taus, [index_0, index_1], [0, 1])
    index_1 += 1  # compensate for the insertion of 0

    p0 = Se3(pose0[:3, :3], pose0[:3, 3])
    p1 = Se3(rotvec_to_r(pose1[:3]), pose1[3:])

    objective_values = []
    for tau in taus:
        p_int = Se3(
            (p0.q ** (1 - tau) * p1.q**tau).R,
            p0.t * (1 - tau) + p1.t * tau,
        )

        objective_values.append(
            get_reprojection_residuals_cpu(
                p_int.T, points, observations, intrinsics
            ).sum()
        )

    ax.plot(taus, objective_values)
    ax.plot(0, objective_values[index_0], "o", color="red", label=label0)
    ax.plot(1, objective_values[index_1], "o", color="blue", label=label1)


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

    print("=== optimization ===")
    start = time.process_time()
    params, state = jaxopt_benchmark.optimize(
        camera_index, jaxopt_benchmark.cam_poses_gpu[camera_index + 3]
    )
    print("optimization time:", time.process_time() - start, "s")

    print("Loss:", state.loss)

    print(
        "Pose error:",
        JaxPoseOptimizer.pose_mat_to_vec(jaxopt_benchmark.cam_poses[camera_index])
        - np.array(params),
    )

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 1)
    plot_costs(
        ax,
        jaxopt_benchmark.cam_poses[camera_index + 1],
        np.array(params),
        jaxopt_benchmark.points[camera_index],
        jaxopt_benchmark.observations[camera_index],
        jaxopt_benchmark.intrinsics[camera_index],
        label0="initial pose",
        label1="optimized pose",
        n=100,
    )
    # ax.axhline(results_gt.cost, color='k', linestyle='--')
    ax.set_xlabel("distance (normalized)")
    ax.set_ylabel("cost function")
    ax.legend()
    fig.savefig("test_jaxopt.png")
