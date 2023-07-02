import os
import sys
import time
from datetime import datetime

src_path = "/home/kuti/py_ws/gsu_jaxopt/jaxopt-3D-reconstruction/"
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np

from src.benchmark.benchmark import Benchmark
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset
from src.reconstruction.bundle_adjustment.pose_optimization import JaxPoseOptimizer


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

        self.initial_point_sizes = [len(p) for p in self.points]
        self.points_num = max(self.points, key=lambda x: x.shape[0]).shape[0]

        self.cam_poses_gpu = JaxPoseOptimizer.to_gpu(self.cam_poses)
        self.intrinsics_gpu = JaxPoseOptimizer.to_gpu(self.intrinsics)

    def __len__(self):
        return len(self.cam_poses)

    def _prepare_dataset(self):
        cam_poses = []
        intrinsics = []
        points_3d = []
        points_2d = []

        for _, e in enumerate(self.dataset.datasetEntries):
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

    def compile(self, index):
        self.optimizer.compile(self.initial_point_sizes[index], batch_size=1)

    def optimize(
        self, index: int, initial_poses: np.array, initial_intrinsics: np.array
    ):
        return self.optimizer.optimize(
            np.expand_dims(initial_poses, 0),
            np.expand_dims(initial_intrinsics, 0),
            JaxPoseOptimizer.to_gpu(np.expand_dims(self.points[index], 0)),
            JaxPoseOptimizer.to_gpu(np.expand_dims(self.observations[index], 0)),
            JaxPoseOptimizer.to_gpu(
                np.ones((1, self.points[index].shape[0]), dtype=bool)
            ),
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
    ds = (
        Dataset.with_noise(
            ds, camera_translation_noise=0.01, camera_rotation_noise=0.01
        )
        if config["add_noise"]
        else ds
    )

    jaxopt_benchmark = JaxoptBenchmark(ds)

    #
    # optimization
    #

    cam_num = len(jaxopt_benchmark)

    # losses = []
    # times = []
    # start = time.process_time()
    # for camera_index in range(0, cam_num):
    #     print("=== cam_pose %d ===" % camera_index)

    #     #
    #     # compilation
    #     #
    #     start = time.process_time()
    #     jaxopt_benchmark.compile(camera_index)
    #     print("compilation time: %.5fs" % (time.process_time() - start))

    #     # fx fy cx cy skew
    #     intr = jaxopt_benchmark.intrinsics[camera_index]

    #     initial_intrinsics = np.array(
    #         [
    #             intr[0, 0],
    #             intr[1, 1],
    #             intr[0, 2],
    #             intr[1, 2],
    #             intr[0, 1],
    #         ]
    #     ).T

    #     opt_start = time.process_time()
    #     params, state = jaxopt_benchmark.optimize(
    #         camera_index,
    #         jaxopt_benchmark.cam_poses_gpu[camera_index],
    #         initial_intrinsics,
    #     )
    #     opt_time = time.process_time() - opt_start
    #     print("optimization time: %.5fs" % opt_time)

    #     times.append(opt_time)
    #     losses.extend(state.loss)

    # #
    # # initial losses
    # #

    # initial_losses = []
    # for camera_index in range(cam_num):
    #     loss = get_reprojection_residuals_cpu(
    #         jaxopt_benchmark.cam_poses[camera_index],
    #         jaxopt_benchmark.points[camera_index],
    #         jaxopt_benchmark.observations[camera_index],
    #         jaxopt_benchmark.intrinsics[camera_index],
    #         np.ones((1, jaxopt_benchmark.points[camera_index].shape[0]), dtype=bool),
    #     ).sum()

    #     initial_losses.append(loss)

    times = [
        0.12289790000000167,
        0.034958199999998385,
        0.042888999999998845,
        0.03152279999999763,
        0.0354005000000015,
        0.036324699999994436,
        0.04218620000000328,
        0.0343726999999987,
        0.03283899999999562,
        0.03755979999999681,
        0.03211530000000096,
        0.04322919999999897,
        0.06484880000000715,
        0.03787780000000396,
        0.040019599999993716,
        0.06047460000000626,
        0.035520300000001725,
        0.029966400000006388,
        0.03895650000001183,
        0.03421450000001869,
        0.03538810000000581,
        0.0637911000000031,
        0.03443980000000124,
        0.035793799999993325,
        0.1048613999999759,
        0.10773370000001137,
        0.03658329999998955,
        0.04327520000001073,
        0.037269699999995964,
        0.029898599999995668,
        0.03586990000002288,
        0.040505800000005365,
        0.035998900000009826,
        0.03422040000000948,
        0.0341158999999891,
        0.09519289999997227,
        0.0521577000000093,
        0.03754340000000411,
        0.03642200000001594,
        0.03382300000001237,
        0.05190479999998843,
        0.03412500000001728,
        0.03534630000001471,
        0.046209799999985535,
        0.03734380000003057,
        0.042855599999995775,
        0.03539159999996855,
        0.04151080000002594,
        0.03830769999996164,
        0.03931729999999334,
        0.032594100000039816,
        0.0335886999999957,
        0.03593179999995755,
        0.0425850999999966,
        0.03962360000002718,
        0.033366099999966536,
        0.03683139999998275,
        0.03394780000002129,
        0.03344390000000885,
        0.03845339999998032,
        0.03647620000003826,
        0.07497430000000804,
        0.0368090000000052,
        0.038333700000009685,
        0.04939110000003666,
        0.03704779999998209,
        0.042280399999981455,
        0.035512999999980366,
        0.03917529999995395,
        0.0550117999999884,
        0.03532010000003538,
        0.0415787000000023,
        0.034933200000011766,
        0.03632460000000037,
        0.03786689999998316,
    ]
    print(times)
    #
    # plot
    #

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=400)
    ax2 = ax1.twinx()

    p_sizes, ts = zip(*sorted(zip(jaxopt_benchmark.initial_point_sizes, times)))

    p_num = ax1.bar(
        np.arange(cam_num),
        p_sizes,
        edgecolor="green",
        width=1.0,
        color="black",
        label="number of points",
    )
    opt_time = ax2.bar(
        np.arange(cam_num),
        ts,
        width=0.5,
        color="r",
        label="optimization time",
    )

    ax1.set_xlabel("camera configuration")
    ax1.set_ylabel(p_num.get_label(), color="g")
    ax2.set_ylabel(opt_time.get_label() + " (s)", color="r")
    ax1.legend([p_num, opt_time], [p_num.get_label(), opt_time.get_label()])
    date_format = r"%m%d_%H%M%S"
    fig.savefig(
        f"figures/test_jaxopt_single_{datetime.now().strftime(date_format)}.png"
    )
