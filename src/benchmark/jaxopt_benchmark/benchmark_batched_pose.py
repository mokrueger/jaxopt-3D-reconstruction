import os
import sys
import time
from datetime import datetime

src_path = "/home/kuti/py_ws/gsu_jaxopt/jaxopt-3D-reconstruction/"
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np

from src.benchmark.benchmark import (
    Benchmark,
    SinglePoseBenchmark,
    SinglePoseBenchmarkResults,
)
from src.benchmark.jaxopt_benchmark.helpers import _parse_output_params
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.reconstruction.bundle_adjustment.pose_optimization import JaxPoseOptimizer
from src.reconstruction.bundle_adjustment.utils import get_reprojection_residuals_cpu


class JaxoptSinglePoseBenchmarkBatched(SinglePoseBenchmark):
    FRAMEWORK = "JAX"
    NAME = "Single Pose Benchmark Batched"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        (
            self.optimizer,
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
            self.initial_point_sizes,
            self.points_num,
            self.masks,
            self.cam_poses_gpu,
            self.intrinsics_gpu,
            self.points_gpu,
            self.observations_gpu,
            self.masks_gpu,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def setup(self):
        self.optimizer = JaxPoseOptimizer()

        (
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
        ) = self._prepare_dataset()

        self.initial_point_sizes = [len(p) for p in self.points]
        self.points_num = max(self.points, key=lambda x: x.shape[0]).shape[0]

        self.masks = self._pad_points_and_create_masks()

        self.cam_poses_gpu = JaxPoseOptimizer.to_gpu(self.cam_poses)
        self.intrinsics_gpu = JaxPoseOptimizer.to_gpu(self.intrinsics)
        self.points_gpu = JaxPoseOptimizer.to_gpu(self.points)
        self.observations_gpu = JaxPoseOptimizer.to_gpu(self.observations)
        self.masks_gpu = JaxPoseOptimizer.to_gpu(self.masks)

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []
        points_3d = []
        points_2d = []

        for _, e in enumerate(self.dataset.datasetEntries):
            cam = e.camera

            mapped_points = e.map2d_3d(
                self.dataset.points3D_mapped, zipped=False, np=True
            )
            p_2d, p_3d = [np.array(l) for l in mapped_points]
            p_3d = np.concatenate([p_3d, np.ones((p_3d.shape[0], 1))], axis=1)

            points_2d.append(p_2d)
            points_3d.append(p_3d)

            cam_poses.append(cam.camera_pose.rotation_translation_matrix)
            intrinsics.append(cam.camera_intrinsics.camera_intrinsics_matrix)

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)

        return cam_poses, intrinsics, points_3d, points_2d

    def _pad_points_and_create_masks(self):
        masks = []
        for i in range(len(self)):
            curr_num = self.points[i].shape[0]
            diff = self.points_num - curr_num

            masks.append(
                np.concatenate((np.full(curr_num, True), np.full(diff, False)))
            )

            pad = np.block([np.zeros((diff, 3)), np.ones((diff, 1))])  # np.zeros(diff)
            self.points[i] = np.concatenate((self.points[i], pad))
            self.observations[i] = np.concatenate(
                (self.observations[i], np.zeros((diff, 2)))
            )

        return np.array(masks)

    def compile(self, batch_size=8):
        self.optimizer.compile(self.points_num, batch_size=batch_size)

    def optimize(
        self,
        index: int,
        initial_poses: np.array,
        initial_intrinsics: np.array,
        batch_size=8,
    ):
        return self.optimizer.optimize(
            initial_poses,
            initial_intrinsics,
            self.points_gpu[index : index + batch_size],
            self.observations_gpu[index : index + batch_size],
            self.masks_gpu[index : index + batch_size],
        )

    def optimize_single_pose_batched(self, camera_index, batch_size, verbose):
        # fx fy cx cy skew
        intr = self.intrinsics[camera_index : camera_index + batch_size]

        initial_intrinsics_batch = np.array(
            [
                intr[:, 0, 0],
                intr[:, 1, 1],
                intr[:, 0, 2],
                intr[:, 1, 2],
                intr[:, 0, 1],
            ]
        ).T

        if verbose:
            print("=== compilation ===")
        start = time.perf_counter()
        self.compile(batch_size)
        compilation_time = time.perf_counter() - start

        if verbose:
            print("compilation time:", compilation_time, "s")
            print("=== optimization ===")

        start = time.perf_counter()
        params, state = self.optimize(
            index=camera_index,
            initial_poses=self.cam_poses_gpu[camera_index : camera_index + batch_size],
            initial_intrinsics=initial_intrinsics_batch,
            batch_size=batch_size,
        )
        optimization_time = time.perf_counter() - start

        if verbose:
            print("optimization time:", optimization_time, "s")
            print("Loss:", state.loss, "in", state.iter_num, "iterations")
            print("Gradient:", np.mean(np.abs(state.gradient)))
        return compilation_time, optimization_time, params, state

    def benchmark(self, *args, **kwargs):
        """
        Args:
            @parameter verbose (bool, default: True): specify verbosity of output
            @parameter batch_size (int, default: 1): specify num of entries processed in parallel.
            Must be divisible by length of datasetEntries.
        """
        self.setup()
        verbose = kwargs.get("verbose", True)
        batch_size = kwargs.get("batch_size", 1)
        c_times, o_times, param_list, state_list = [], [], [], []
        for i in range(0, len(self.cam_poses), batch_size):
            (
                compilation_time,
                optimization_time,
                params,
                state,
            ) = self.optimize_single_pose_batched(
                i, batch_size=batch_size, verbose=verbose
            )
            c_times.append(compilation_time),
            o_times.append(optimization_time)
            param_list += list(params)
            state_list += state

        total_c = sum(c_times)
        total_o = sum(o_times)

        total_t = total_c + total_o

        self._results = SinglePoseBenchmarkResults(
            camera_mapping=_parse_output_params(param_list, self.dataset)
        )
        self._time = c_times, o_times, total_t
        self._single_times = list(
            map(lambda x: x[0] + x[1], list(zip(c_times, o_times)))
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
    ds = (
        Dataset.with_noise(
            ds, camera_translation_noise=0.01, camera_rotation_noise=0.01
        )
        if config["add_noise"]
        else ds
    )
    batch_size = 75

    jaxopt_benchmark = JaxoptSinglePoseBenchmarkBatched(ds)

    #
    # compilation
    #

    print("=== compilation ===")
    start = time.process_time()
    jaxopt_benchmark.compile(batch_size=batch_size)
    print("compilation time: %.5fs" % (time.process_time() - start))

    #
    # optimization
    #

    print("=== optimization ===")
    cam_num = len(jaxopt_benchmark)

    losses = []
    times = []
    start = time.process_time()
    for camera_index in range(0, cam_num, batch_size):
        # fx fy cx cy skew
        intr = jaxopt_benchmark.intrinsics[camera_index : camera_index + batch_size]

        initial_intrinsics = np.array(
            [
                intr[:, 0, 0],
                intr[:, 1, 1],
                intr[:, 0, 2],
                intr[:, 1, 2],
                intr[:, 0, 1],
            ]
        ).T

        opt_start = time.process_time()
        params, state = jaxopt_benchmark.optimize(
            camera_index,
            jaxopt_benchmark.cam_poses_gpu[camera_index : camera_index + batch_size],
            initial_intrinsics,
            batch_size=batch_size,
        )

        times.extend([(time.process_time() - opt_start) / batch_size] * batch_size)
        losses.extend(state.loss)

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
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=400)
    ax2 = ax1.twinx()

    p_num = ax1.bar(
        np.arange(cam_num),
        jaxopt_benchmark.initial_point_sizes,
        color="r",
        label="number of points",
    )
    opt_time = ax2.scatter(
        np.arange(cam_num), times, color="g", label="optimization time"
    )

    ax1.set_xlabel("camera number")
    ax1.set_ylabel(p_num.get_label(), color="r")
    ax2.set_ylabel(opt_time.get_label(), color="g")
    ax1.legend([p_num, opt_time], [p_num.get_label(), opt_time.get_label()])
    date_format = r"%m%d_%H%M%S"
    fig.savefig(f"figures/test_jaxopt_{datetime.now().strftime(date_format)}.png")
    # fig.show()
