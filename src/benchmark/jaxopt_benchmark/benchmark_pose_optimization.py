import time

import numpy as np
from tqdm import tqdm

from src.benchmark.benchmark import SinglePoseBenchmark, SinglePoseBenchmarkResults
from src.benchmark.jaxopt_benchmark.helpers import _parse_output_params
from src.dataset.dataset import Dataset
from src.reconstruction.bundle_adjustment.loss import JaxLossFunction
from src.reconstruction.bundle_adjustment.pose_optimization import JaxPoseOptimizer
from src.reconstruction.bundle_adjustment.utils import to_gpu


class JaxoptSinglePoseBenchmarkBatched(SinglePoseBenchmark):
    FRAMEWORK = "JAX Batched"
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
        ) = (None, None, None, None, None, None, None, None, None)

    def setup(self):
        (
            self.cam_poses,
            self.intrinsics,
            self.points,
            self.observations,
            self.avg_cam_width,
        ) = self._prepare_dataset()

        self.optimizer = JaxPoseOptimizer(
            avg_cam_width=self.avg_cam_width, loss_fn=JaxLossFunction.CAUCHY
        )

        self.initial_point_sizes = [len(p) for p in self.points]
        self.points_num = max(self.points, key=lambda x: x.shape[0]).shape[0]

        self.masks = self._create_masks()

        self.cam_poses_gpu = to_gpu(self.cam_poses)

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

    def _create_masks(self):
        return np.array(
            [
                np.concatenate(
                    [
                        np.full(len(points), 1.0),
                        np.full(self.points_num - len(points), 0.0),
                    ]
                )
                for points in self.points
            ]
        )

    def _pad_points(self, points, observations):
        len_diff = self.points_num - len(points)

        return (
            np.concatenate([points, np.zeros((len_diff, 3))]),
            np.concatenate([observations, np.zeros((len_diff, 2))]),
        )

    def _get_mask(self, index, batch_size=1):
        if batch_size == 1:
            return to_gpu(np.ones((1, len(self.points[index]))))

        return to_gpu(self.masks[index : index + batch_size])

    def _prepare_points(self, index, batch_size=1):
        if batch_size == 1:
            return (
                to_gpu(self.points[index][None, ...]),
                to_gpu(self.observations[index][None, ...]),
            )

        points_gpu = []
        observations_gpu = []
        for points, observations in zip(
            self.points[index : index + batch_size],
            self.observations[index : index + batch_size],
        ):
            points_padded, observations_padded = self._pad_points(points, observations)
            points_gpu.append(points_padded)
            observations_gpu.append(observations_padded)

        return (to_gpu(points_gpu), to_gpu(observations_gpu))

    def compile(self, points_num, batch_size=8):
        self.optimizer.compile(points_num, batch_size=batch_size)

    def optimize(
        self,
        opt_params: np.array,
        points_gpu: np.array,
        observations_gpu: np.array,
        cx_cy_skew: np.array,
        masks: np.array,
    ):
        return self.optimizer.optimize(
            opt_params, points_gpu, observations_gpu, cx_cy_skew, masks
        )

    def optimize_single_pose_batched(self, camera_index, batch_size, verbose):
        intr = self.intrinsics[camera_index : camera_index + batch_size]
        intrinsics0 = np.array(  # fx fy cx cy skew
            [intr[:, 0, 0], intr[:, 1, 1], intr[:, 0, 2], intr[:, 1, 2], intr[:, 0, 1]]
        ).T

        poses0 = self.cam_poses_gpu[camera_index : camera_index + batch_size]
        opt_params, cx_cy_skew = self.optimizer.prepare_params(poses0, intrinsics0)

        opt_params = to_gpu(opt_params)
        cx_cy_skew = to_gpu(cx_cy_skew)

        points_gpu, observations_gpu = self._prepare_points(
            camera_index, batch_size=batch_size
        )

        masks_gpu = self._get_mask(camera_index, batch_size=batch_size)

        start = time.perf_counter()
        self.compile(
            len(self.points[camera_index]) if batch_size == 1 else self.points_num,
            batch_size=batch_size,
        )
        initial_time = time.perf_counter() - start

        start = time.perf_counter()
        params, state = self.optimize(
            opt_params,
            points_gpu,
            observations_gpu,
            cx_cy_skew,
            masks_gpu,
        )
        optimization_time = time.perf_counter() - start
        compilation_time = max(0.0, initial_time - optimization_time)

        params = np.concatenate([params, cx_cy_skew], axis=1)

        if verbose:
            print("=== Camera %d ===" % camera_index)
            print("compilation time:", compilation_time, "s")
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

        if verbose:
            print("Warming up...")
        self.compile(3000, batch_size=batch_size)

        for i in tqdm(
            range(0, len(self.cam_poses), batch_size),
            total=len(self.cam_poses) // batch_size,
            desc="Camera pose: ",
        ):
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
            state_list.append(state)

        total_c = c_times[0]  # we can do this instead of sum(c_times)
        # because we use masks that only should need to compile once.
        total_o = sum(o_times)

        total_t = total_c + total_o
        iterations = (
            list(map(lambda s: int(s.iter_num), state_list))
            if batch_size == 1
            else [int(a) for l in [list(s.iter_num) for s in state_list] for a in l]
        )
        if verbose:
            if batch_size == 1:
                print(
                    f"Average iterations: {np.round(np.average(iterations), decimals=2)}"
                )
            else:
                print(
                    f"Average iterations: {np.round(np.average(iterations), decimals=2)}"
                )

        self._results = SinglePoseBenchmarkResults(
            camera_mapping=_parse_output_params(param_list, self.dataset)
        )
        self._time = c_times, o_times, total_t
        #  self._single_times = list(map(lambda x: x[0] + x[1], list(zip(c_times, o_times))))
        self._single_times = [o_times[0] + total_c, *o_times[1:]]
        self._iterations = iterations
