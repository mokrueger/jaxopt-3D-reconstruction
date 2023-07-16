import sys

src_path = "/home/kuti/py_ws/gsu_jaxopt/jaxopt-3D-reconstruction"
if src_path not in sys.path:
    sys.path.append(src_path)

import os
import time

import jax
import numpy as np

from src.benchmark.jaxopt_benchmark.helpers import _parse_output_params_bundle
from src.config import DATASETS_PATH
from src.dataset.loss_functions import LossFunction

jax.config.update("jax_platform_name", "gpu")

from src.benchmark.benchmark import (
    BundleAdjustmentBenchmark,
    BundleAdjustmentBenchmarkResults,
)
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import (
    load_colmap_dataset,
    open_dataset_in_colmap,
)
from src.reconstruction.bundle_adjustment.bundle_adjustment import JaxBundleAdjustment


class JaxoptBundleAdjustmentBenchmark(BundleAdjustmentBenchmark):
    FRAMEWORK = "JAX"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.points_limit = 100
        self.camera_limit = 5

        (
            self.points_2d_all,
            self.points_3d_all,
            self.p3d_indices_all,
            self.cam_poses,
            self.intrinsics,
            self.benchmark_index_to_point_identifier_mapping,
            self.avg_cam_width,
        ) = self._prepare_dataset()

        self.points_2d_all_gpu = to_gpu(self.points_2d_all)
        self.points_3d_all_gpu = to_gpu(self.points_3d_all)
        self.p3d_indices_all_gpu = to_gpu(self.p3d_indices_all)
        self.cam_poses_gpu = to_gpu(self.cam_poses)
        self.intrinsics_gpu = to_gpu(self.intrinsics)

        self.optimizer = JaxBundleAdjustment(len(self.cam_poses), self.avg_cam_width)

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []

        points_2d_all = []
        p3d_indices_all = []

        map_2d_3d_list = []
        p3d = {}

        avg_cam_width = 0
        for d_entry in self.dataset.datasetEntries[: self.camera_limit]:
            avg_cam_width += d_entry.camera.width
            cam_poses.append(d_entry.camera.camera_pose.rotation_translation_matrix)
            intrinsics.append(d_entry.camera.camera_intrinsics.camera_intrinsics_matrix)
            map_2d_3d_list.append(d_entry.map2d_3d(self.dataset.points3D_mapped))
            p3d.update({p3.identifier: p3.xyz for _, p3 in map_2d_3d_list[-1]})

        max_2d_indices = max(len(i) for i in map_2d_3d_list)

        p3d = sorted(p3d.items(), key=lambda x: x[0])
        p3d_ids = {j[0]: i for i, j in enumerate(p3d)}
        points_3d_all = [p[1] for p in p3d]

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
        p3d_indices_all = np.array(p3d_indices_all)[..., : self.points_limit]
        points_3d_all = np.array(points_3d_all)
        points_2d_all = np.array(points_2d_all)[..., : self.points_limit, :]

        benchmark_index_to_point_identifier_mapping = {v: k for k, v in p3d_ids.items()}
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
            benchmark_index_to_point_identifier_mapping,
            avg_cam_width,
        )

    def compile(self) -> None:
        self.optimizer.compile(len(self.points_3d_all), len(self.p3d_indices_all[0]))

    def optimize(self, opt_params: np.array, cx_cy_skew: np.array):
        return self.optimizer.optimize(
            opt_params,
            self.points_2d_all_gpu,
            self.p3d_indices_all_gpu,
            cx_cy_skew,
        )

    def benchmark(self, *args, **kwargs):
        verbose = kwargs.get("verbose", False)
        initial_intrinsics = np.array(
            [
                [intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2], intr[0, 1]]
                for intr in self.intrinsics
            ]
        )

        opt_params, cx_cy_skew = self.optimizer.prepare_params(
            self.cam_poses, initial_intrinsics, self.points_3d_all
        )

        opt_params = to_gpu(opt_params)
        cx_cy_skew = to_gpu(cx_cy_skew)

        start = time.perf_counter()
        self.compile()
        compile_time = time.perf_counter() - start

        print("compile: ", compile_time)

        start = time.perf_counter()
        params, state = self.optimize(opt_params, cx_cy_skew)
        total_time = time.perf_counter() - start

        print("run: ", total_time)

        # cam, points = _parse_output_params_bundle(
        #     params,
        #     self.dataset,
        #     num_3d_points=len(self.points_3d_all),
        #     num_cams=len(self.cam_poses),
        #     benchmark_index_to_point_identifier_mapping=self.benchmark_index_to_point_identifier_mapping,
        # )
        # self._results = BundleAdjustmentBenchmarkResults(
        #     camera_mapping=cam, point_mapping=points
        # )
        # self._time = total_time
        # self._iterations = int(state.iter_num)


from src.reconstruction.bundle_adjustment.utils import (
    get_reprojection_residuals_cpu,
    to_gpu,
)

if __name__ == "__main__":
    #
    # configuration
    #

    ds_path = os.path.join(src_path, "datasets", "reichstag")
    REICHSTAG_SPARSE_NOISED = os.path.join(DATASETS_PATH, "reichstag/sparse_noised")
    REICHSTAG_SPARSE = os.path.join(DATASETS_PATH, "reichstag/sparse")
    REICHSTAG_IMAGES = os.path.join(DATASETS_PATH, "reichstag/images")

    config = {
        "add_noise": True,
        "path": REICHSTAG_SPARSE_NOISED,
        "image_path": REICHSTAG_IMAGES,
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

    jaxopt_benchmark.benchmark()

    exit()
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

    cam, points = _parse_output_params_bundle(
        params,
        ds,
        num_3d_points=len(jaxopt_benchmark.points_3d_all),
        num_cams=len(jaxopt_benchmark.cam_poses),
        benchmark_index_to_point_identifier_mapping=jaxopt_benchmark.benchmark_index_to_point_identifier_mapping,
    )
    results = BundleAdjustmentBenchmarkResults(camera_mapping=cam, point_mapping=points)
    jaxopt_benchmark._results = results  # preliminary
    jaxopt_benchmark.export_results_in_colmap_format(open_in_colmap=True)
    open_dataset_in_colmap(jaxopt_benchmark.shallow_results_trimmed_original_dataset())
    #  open_dataset_in_colmap(jaxopt_benchmark.dataset)
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

    # initial_losses = []
    # for camera_index in range(cam_num):
    #     loss = get_reprojection_residuals_cpu(
    #         jaxopt_benchmark.cam_poses[camera_index],
    #         jaxopt_benchmark.points[camera_index],
    #         jaxopt_benchmark.observations[camera_index],
    #         jaxopt_benchmark.intrinsics[camera_index],
    #         jaxopt_benchmark.masks[camera_index],
    #     ).sum()

    #     initial_losses.append(loss)

    # #
    # # plot
    # #

    # import matplotlib.pyplot as plt

    # plt.rcParams.update({"font.size": 14})
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=400)

    # ax.bar(np.arange(cam_num), initial_losses, color="r", label="initial")
    # ax.bar(np.arange(cam_num), losses, color="g", label="optimized")

    # ax.set_xlabel("camera number")
    # ax.set_ylabel("cost function")
    # ax.legend()
    # fig.savefig("test_jaxopt.png")
    # fig.show()
