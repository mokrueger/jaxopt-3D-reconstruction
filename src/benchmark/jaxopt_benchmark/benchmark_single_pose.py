import sys

src_path = "/home/kuti/py_ws/gsu_jaxopt/jaxopt-3D-reconstruction"
if src_path not in sys.path:
    sys.path.append(src_path)

import os
import time

import numpy as np

from src.benchmark.benchmark import SinglePoseBenchmark, SinglePoseBenchmarkResults
from src.benchmark.jaxopt_benchmark.helpers import _parse_output_params
from src.config import DATASETS_PATH
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.dataset.loss_functions import LossFunction
from src.reconstruction.bundle_adjustment.loss import JaxLossFunction
from src.reconstruction.bundle_adjustment.pose_optimization import JaxPoseOptimizer
from src.reconstruction.bundle_adjustment.utils import pose_mat_to_vec, to_gpu


class JaxoptSinglePoseBenchmark(SinglePoseBenchmark):
    FRAMEWORK = "JAX"
    NAME = "Single Pose Benchmark"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.optimizer = None
        self.cam_poses = None
        self.intrinsics = None
        self.points = None
        self.observations = None
        self.initial_point_sizes = None
        self.points_num = None
        self.cam_poses_gpu = None
        self.intrinsics_gpu = None

    def setup(self):
        (
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
            self.avg_cam_width,
        ) = self._prepare_dataset()

        self.optimizer = JaxPoseOptimizer(
            self.avg_cam_width, loss_fn=JaxLossFunction.CAUCHY
        )

        self.initial_point_sizes = [len(p) for p in self.points]
        self.points_num = max(self.points, key=lambda x: x.shape[0]).shape[0]

        self.cam_poses_gpu = to_gpu(self.cam_poses)
        self.intrinsics_gpu = to_gpu(self.intrinsics)

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []
        points_3d = []
        points_2d = []

        avg_cam_width = 0

        for _, e in enumerate(self.dataset.datasetEntries):
            cam = e.camera
            avg_cam_width += cam.width

            mapped_points = e.map2d_3d(
                self.dataset.points3D_mapped, zipped=False, np=True
            )
            p_2d, p_3d = [np.array(l) for l in mapped_points]

            points_2d.append(p_2d)
            points_3d.append(p_3d)

            cam_poses.append(cam.camera_pose.rotation_translation_matrix)
            intrinsics.append(cam.camera_intrinsics.camera_intrinsics_matrix)

        avg_cam_width /= len(self.dataset.datasetEntries)

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)

        return cam_poses, intrinsics, points_3d, points_2d, avg_cam_width

    def compile(self, index):
        self.optimizer.compile(self.initial_point_sizes[index])

    def optimize(
        self, index: int, initial_pose: np.array, initial_intrinsics: np.array
    ):
        return self.optimizer.optimize(
            initial_pose,
            initial_intrinsics,
            to_gpu(self.points[index]),
            to_gpu(self.observations[index]),
        )

    def optimize_single_pose(self, camera_index, verbose):
        # fx fy cx cy skew
        intrinsics = self.intrinsics[camera_index]
        initial_intrinsics = [intrinsics[0, 0], intrinsics[1, 1]]

        if verbose:
            print("=== compilation ===")
        start = time.perf_counter()
        self.compile(camera_index)
        compilation_time = time.perf_counter() - start

        if verbose:
            print("compilation time:", compilation_time, "s")
            print("=== optimization ===")

        start = time.perf_counter()
        params, state = self.optimize(
            index=camera_index,
            initial_pose=self.cam_poses_gpu[camera_index],
            initial_intrinsics=initial_intrinsics,
        )
        optimization_time = time.perf_counter() - start

        if verbose:
            print("optimization time:", optimization_time, "s")
            print("Loss:", state.loss, "in", state.iter_num, "iterations")
            print("Gradient:", np.mean(np.abs(state.gradient)))
            print(
                "Pose error:",
                pose_mat_to_vec(self.cam_poses[camera_index]) - np.array(params[:6]),
            )
            print(
                "Intrinsics error:",
                np.array(initial_intrinsics) - np.array(params[6:]),
            )
        return compilation_time, optimization_time, params, state

    def benchmark(self, *args, **kwargs):
        self.setup()
        verbose = kwargs.get("verbose", True)
        c_times, o_times, param_list, state_list = [], [], [], []
        for i in range(len(self.cam_poses)):
            (
                compilation_time,
                optimization_time,
                params,
                state,
            ) = self.optimize_single_pose(i, verbose=verbose)
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
    ds_path = os.path.join(DATASETS_PATH, "reichstag/")

    config = {
        "add_noise": True,
        "path": os.path.join(ds_path, "sparse"),
        "image_path": os.path.join(ds_path, "images"),
    }

    ds = load_colmap_dataset(config["path"], config["image_path"], binary=True)
    gt_errors = ds.compute_reprojection_errors_alt(
        loss_function=LossFunction.CAUCHY_LOSS
    )
    ds_noise = (
        Dataset.with_noise(ds, point2d_noise=0, point3d_noise=0)
        if config["add_noise"]
        else ds
    )
    jaxopt_benchmark = JaxoptSinglePoseBenchmark(ds)
    jaxopt_benchmark.benchmark()

    initial_errors = (
        jaxopt_benchmark.shallow_results_dataset().compute_reprojection_errors()
    )
    print("finished")
