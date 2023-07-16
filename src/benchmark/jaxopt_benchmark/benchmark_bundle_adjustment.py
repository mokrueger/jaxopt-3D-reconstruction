import os
import time

import jax
import numpy as np

from src.benchmark.jaxopt_benchmark.helpers import _parse_output_params_bundle
from src.config import DATASETS_PATH

jax.config.update("jax_platform_name", "gpu")

from src.benchmark.benchmark import (
    BundleAdjustmentBenchmark,
    BundleAdjustmentBenchmarkResults,
)
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.reconstruction.bundle_adjustment.bundle_adjustment import JaxBundleAdjustment


class JaxoptBundleAdjustmentBenchmark(BundleAdjustmentBenchmark):
    FRAMEWORK = "JAX"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

        self.points_limit = 300
        self.camera_limit = 5

        (
            self.points_2d_all,
            self.points_3d_all,
            self.p3d_indices_all,
            self.masks_all,
            self.cam_poses,
            self.intrinsics,
            self.benchmark_index_to_point_identifier_mapping,
            self.avg_cam_width,
        ) = self._prepare_dataset()

        self.points_2d_all_gpu = to_gpu(self.points_2d_all)
        self.points_3d_all_gpu = to_gpu(self.points_3d_all)
        self.p3d_indices_all_gpu = to_gpu(self.p3d_indices_all)
        self.masks_all_gpu = to_gpu(self.masks_all)
        self.cam_poses_gpu = to_gpu(self.cam_poses)

        self.optimizer = JaxBundleAdjustment(len(self.cam_poses), self.avg_cam_width)

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []

        points_2d_all = []
        p3d_indices_all = []
        masks_all = []

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

            pad_len = max_2d_indices - len(points_2d)
            points_2d_all.append(points_2d + (points_2d[0],) * pad_len)
            p3d_indices_all.append(p3d_indices + (p3d_indices[0],) * pad_len)
            masks_all.append([1.0] * len(points_2d) + [0.0] * pad_len)

        cam_poses = np.array(cam_poses)
        intrinsics = np.array(intrinsics)
        p3d_indices_all = np.array(p3d_indices_all)[..., : self.points_limit]
        points_3d_all = np.array(points_3d_all)
        points_2d_all = np.array(points_2d_all)[..., : self.points_limit, :]
        masks_all = np.array(masks_all)[..., : self.points_limit]

        benchmark_index_to_point_identifier_mapping = {v: k for k, v in p3d_ids.items()}

        return (
            points_2d_all,
            points_3d_all,
            p3d_indices_all,
            masks_all,
            cam_poses,
            intrinsics,
            benchmark_index_to_point_identifier_mapping,
            avg_cam_width,
        )

    def compile(self) -> None:
        self.optimizer.compile(
            len(self.points_3d_all_gpu), len(self.p3d_indices_all[0])
        )

    def optimize(self, opt_params: np.array, cx_cy_skew: np.array):
        return self.optimizer.optimize(
            opt_params,
            self.points_2d_all_gpu,
            self.p3d_indices_all_gpu,
            cx_cy_skew,
            self.masks_all_gpu,
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

        # print("opt_params:", opt_params.shape, opt_params.dtype)
        # print("points_2d_all:", points_2d_all.shape, points_2d_all.dtype)
        # print("p3d_indices_all:", p3d_indices_all.shape, p3d_indices_all.dtype)
        # print("cx_cy_skew:", cx_cy_skew.shape, cx_cy_skew.dtype)
        # print("masks_all:", masks_all.shape, masks_all.dtype)

        start = time.perf_counter()
        params, state = self.optimize(opt_params, cx_cy_skew)
        total_time = time.perf_counter() - start

        print("run: ", total_time)

        cam, points = _parse_output_params_bundle(
            params,
            self.dataset,
            cx_cy_skew=cx_cy_skew,
            num_3d_points=len(self.points_3d_all),
            num_cams=len(self.cam_poses),
            benchmark_index_to_point_identifier_mapping=self.benchmark_index_to_point_identifier_mapping,
        )
        self._results = BundleAdjustmentBenchmarkResults(
            camera_mapping=cam, point_mapping=points
        )
        self._time = total_time
        self._iterations = int(state.iter_num)


from src.reconstruction.bundle_adjustment.utils import (
    get_reprojection_residuals_cpu,
    to_gpu,
)

if __name__ == "__main__":
    #
    # configuration
    #
    src_path = ""
    ds_path = os.path.join(src_path, "datasets", "reichstag")
    REICHSTAG_SPARSE_NOISED = os.path.join(DATASETS_PATH, "reichstag/sparse_noised")
    REICHSTAG_SPARSE = os.path.join(DATASETS_PATH, "reichstag/sparse")
    REICHSTAG_IMAGES = os.path.join(DATASETS_PATH, "reichstag/images")

    config = {
        "add_noise": True,
        "path": REICHSTAG_SPARSE_NOISED,
        "image_path": REICHSTAG_IMAGES,
    }

    ds = load_colmap_dataset(config["path"], config["image_path"], binary=True)

    jaxopt_benchmark = JaxoptBundleAdjustmentBenchmark(ds)

    jaxopt_benchmark.benchmark()
