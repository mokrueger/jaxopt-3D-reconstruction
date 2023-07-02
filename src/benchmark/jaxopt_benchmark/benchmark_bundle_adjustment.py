import os
import sys
import time

src_path = "/home/kuti/py_ws/gsu_jaxopt/jaxopt-3D-reconstruction/"
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np

from src.benchmark.benchmark import Benchmark
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.reconstruction.bundle_adjustment.bundle_adjustment import JaxBundleAdjustment


class JaxoptBundleAdjustmentBenchmark(Benchmark):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        (
            self.points_2d_all,
            self.points_3d_all,
            self.p3d_indices_all,
            self.cam_poses,
            self.intrinsics,
        ) = self._prepare_dataset()

        self.points_2d_all_gpu = JaxBundleAdjustment.to_gpu(self.points_2d_all)
        self.points_3d_all_gpu = JaxBundleAdjustment.to_gpu(self.points_3d_all)
        self.p3d_indices_all_gpu = JaxBundleAdjustment.to_gpu(self.p3d_indices_all)
        self.cam_poses_gpu = JaxBundleAdjustment.to_gpu(self.cam_poses)
        self.intrinsics_gpu = JaxBundleAdjustment.to_gpu(self.intrinsics)

        self.optimizer = JaxBundleAdjustment(len(self.cam_poses))

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []

        points_2d_all = []
        p3d_indices_all = []

        p3d = sorted(
            [(p.identifier, p.xyz) for p in self.dataset.points3D], key=lambda x: x[0]
        )
        p3d_ids = {j[0]: i for i, j in enumerate(p3d)}
        points_3d_all = [p[1] for p in p3d]

        map_2d_3d_list = []

        for d_entry in self.dataset.datasetEntries[:2]:
            cam_poses.append(d_entry.camera.camera_pose.rotation_translation_matrix)
            intrinsics.append(d_entry.camera.camera_intrinsics.camera_intrinsics_matrix)
            map_2d_3d_list.append(d_entry.map2d_3d(self.dataset.points3D_mapped))

        max_2d_indices = max(len(i) for i in map_2d_3d_list)

        for map2d_3d in map_2d_3d_list:
            points_2d = []
            p3d_indices = []

            points_2d, p3d_indices = zip(
                *[(p2.xy, p3d_ids[p3.identifier]) for p2, p3 in map2d_3d]
            )

            points_2d_all.append(
                points_2d + (points_2d[0],) * (max_2d_indices - len(points_2d))
            )  # pad with recurring data

            p3d_indices_all.append(
                p3d_indices + (p3d_indices[0],) * (max_2d_indices - len(p3d_indices))
            )  # pad with recurring data

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)
        p3d_indices_all = np.array(p3d_indices_all)[..., :200]
        points_3d_all = np.array(points_3d_all)
        points_2d_all = np.array(points_2d_all)[..., :200, :]

        print(
            "points_2d_all: ",
            points_2d_all.shape,
            "\npoints_3d_all: ",
            points_3d_all.shape,
            "\np3d_indices_all: ",
            p3d_indices_all.shape,
            "\nposes: ",
            cam_poses.shape,
            "\nintrinsics: ",
            intrinsics.shape,
        )

        return (
            points_2d_all,
            points_3d_all,
            p3d_indices_all,
            cam_poses,
            intrinsics,
        )

    def compile(self, index: int) -> None:
        self.optimizer.compile(self.points_num)

    def optimize(self, initial_pose: np.array, initial_intrinsics: np.array):
        return self.optimizer.optimize(
            initial_pose,
            initial_intrinsics,
            self.points_2d_all_gpu,
            self.points_3d_all_gpu,
            self.p3d_indices_all_gpu,
        )


from src.reconstruction.bundle_adjustment.pose_optimization import (
    get_reprojection_residuals_cpu,
)

if __name__ == "__main__":
    #
    # configuration
    #

    ds_path = os.path.join(src_path, "dataset", "reichstag", "dense")

    config = {
        "add_noise": True,
        "path": os.path.join(ds_path, "sparse"),
        "image_path": os.path.join(ds_path, "images"),
    }

    #
    # dataset & benchmark generation
    #

    ds = load_colmap_dataset(config["path"], config["image_path"], binary=True)

    # ds = (
    #     Dataset.with_noise(
    #         ds, camera_translation_noise=0.01, camera_rotation_noise=0.01
    #     )
    #     if config["add_noise"]
    #     else ds
    # )

    jaxopt_benchmark = JaxoptBundleAdjustmentBenchmark(ds)

    #
    # compilation
    #

    # print("=== compilation ===")
    # start = time.process_time()
    # jaxopt_benchmark.compile(0)
    # print("compilation time: %.5fs" % (time.process_time() - start))

    #
    # optimization
    #

    initial_intrinsics = np.array(
        [
            [
                intr[0, 0],
                intr[1, 1],
                intr[0, 2],
                intr[1, 2],
                intr[0, 1],
            ]
            for intr in jaxopt_benchmark.intrinsics
        ]
    )

    start = time.process_time()
    params, state = jaxopt_benchmark.optimize(
        jaxopt_benchmark.cam_poses, initial_intrinsics
    )

    print(state)
    # print("=== optimization ===")
    # cam_num = len(jaxopt_benchmark)

    # losses = []
    # start = time.process_time()
    # for camera_index in range(cam_num):
    #     # fx fy cx cy skew
    #     intr = jaxopt_benchmark.intrinsics[camera_index]
    #     initial_intrinsics = [
    #         intr[0, 0],
    #         intr[1, 1],
    #         intr[0, 2],
    #         intr[1, 2],
    #         intr[0, 1],
    #     ]

    #     params, state = jaxopt_benchmark.optimize(
    #         camera_index,
    #         jaxopt_benchmark.cam_poses_gpu[camera_index],
    #         initial_intrinsics,
    #     )

    #     losses.append(state.loss)

    print("optimization time: %.5fs" % (time.process_time() - start))

    #
    # initial losses
    #

    initial_losses = []
    for camera_index in range(cam_num):
        loss = get_reprojection_residuals_cpu(
            jaxopt_benchmark.cam_poses[camera_index],
            jaxopt_benchmark.points[camera_index],
            jaxopt_benchmark.observations[camera_index],
            jaxopt_benchmark.intrinsics[camera_index],
            jaxopt_benchmark.masks[camera_index],
        ).sum()

        initial_losses.append(loss)

    #
    # plot
    #

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=400)

    ax.bar(np.arange(cam_num), initial_losses, color="r", label="initial")
    ax.bar(np.arange(cam_num), losses, color="g", label="optimized")

    ax.set_xlabel("camera number")
    ax.set_ylabel("cost function")
    ax.legend()
    fig.savefig("test_jaxopt.png")
    fig.show()
