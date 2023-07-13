"""
This is where the code for the comparison between the three methods goes
"""
import contextlib
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List

from src.benchmark.benchmark import Benchmark
from src.benchmark.benchmark_visualization import single_pose_statistics
from src.benchmark.colmap_benchmark.benchmark_bundle_adjustment import (
    ColmapBundleAdjustmentBenchmark,
)
from src.benchmark.colmap_benchmark.benchmark_single_pose import (
    ColmapSinglePoseBenchmark,
)
from src.benchmark.gtsam_benchmark.benchmark_bundle_adjustment import (
    GtsamBundleAdjustmentBenchmark,
)
from src.benchmark.jaxopt_benchmark.benchmark_pose_optimization import JaxoptSinglePoseBenchmarkBatched
from src.benchmark.jaxopt_benchmark.benchmark_bundle_adjustment import JaxoptBundleAdjustmentBenchmark
from src.config import DATASETS_PATH, BENCHMARK_SINGLE_POSE_RESULTS_PATH
#  from src.benchmark.gtsam_benchmark.benchmark_single_pose import import benchmark_gtsam_single_pose
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset

REICHSTAG_SPARSE_NOISED = os.path.join(DATASETS_PATH, "reichstag/sparse_noised")
REICHSTAG_SPARSE = os.path.join(DATASETS_PATH, "reichstag/sparse")
REICHSTAG_IMAGES = os.path.join(DATASETS_PATH, "reichstag/images")

SACRE_COEUR_SPARSE_NOISED = os.path.join(DATASETS_PATH, "sacre_coeur/sparse_noised")
SACRE_COEUR_IMAGES = os.path.join(DATASETS_PATH, "sacre_coeur/images")

ST_PETERS_SQUARE_SPARSE_NOISED = os.path.join(DATASETS_PATH, "st_peters_square/sparse_noised")
ST_PETERS_SQUARE_IMAGES = os.path.join(DATASETS_PATH, "st_peters_square/images")


def save_benchmarks(list_of_benchmarks: List[Benchmark], parent_dir, override_latest=True):
    current_time_formatted = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    benchmarks_path = os.path.join(parent_dir, current_time_formatted)
    latest_path = os.path.join(parent_dir, "latest")

    os.makedirs(benchmarks_path, exist_ok=True)
    for b in list_of_benchmarks:
        # Note: Default filename can lead to overrides e.g. when same benchmark class twice
        f = b.export_pickle(benchmarks_path)
        if override_latest:
            os.makedirs(latest_path, exist_ok=True)
            path_in_latest = os.path.join(latest_path, str(Path(f).name))
            with contextlib.suppress(FileNotFoundError):
                os.remove(path_in_latest)
            shutil.copy(f, path_in_latest)


def benchmark_single_pose(dataset, **kwargs):
    # jaxopt_benchmark = JaxoptSinglePoseBenchmark(dataset)

    print("Benchmarking JAX")
    jaxopt_benchmark_batched = JaxoptSinglePoseBenchmarkBatched(dataset)
    jaxopt_benchmark_batched.subprocess_benchmark(verbose=False, batch_size=kwargs.get("batch_size"))
    total_c, total_o, total_t = jaxopt_benchmark_batched.time
    total_e = sum(total_o)

    # jaxopt_benchmark_batched.subprocess_benchmark(verbose=False, batch_size=75)
    # timee2 = jaxopt_benchmark_batched.time
    # results2 = jaxopt_benchmark_batched.results
    # jaxopt_benchmark.subprocess_benchmark(verbose=False)
    # timee3 = jaxopt_benchmark.time
    # results3 = jaxopt_benchmark.results
    print("Benchmarking Colmap")
    colmapSinglePoseBenchmark = ColmapSinglePoseBenchmark(dataset)
    colmapSinglePoseBenchmark.benchmark()

    save_benchmarks(
        [jaxopt_benchmark_batched, colmapSinglePoseBenchmark],
        os.path.join(BENCHMARK_SINGLE_POSE_RESULTS_PATH),
        override_latest=True
    )
    single_pose_statistics([jaxopt_benchmark_batched, colmapSinglePoseBenchmark])
    return {
        "colmap": colmapSinglePoseBenchmark.time,
        "jax": (total_c, total_o, total_t, total_e),
    }


def benchmark_bundle_adjustment(dataset):
    jaxopt_benchmark = JaxoptBundleAdjustmentBenchmark(dataset)
    jaxopt_benchmark.benchmark()

    colmap_benchmark = ColmapBundleAdjustmentBenchmark(dataset)
    colmap_benchmark.benchmark()

    gtsam_benchmark = GtsamBundleAdjustmentBenchmark(dataset)
    gtsam_benchmark.benchmark()

    return {
        "colmap_time": colmap_benchmark.time,
        "colmap_results": colmap_benchmark.results,
        "gtsam_time": gtsam_benchmark.time,
        "gtsam_results": gtsam_benchmark.results
    }


if __name__ == "__main__":

    # ds0 = load_colmap_dataset(REICHSTAG_SPARSE, REICHSTAG_IMAGES, binary=True, name="Reichstag Original")
    # ds0.compute_reprojection_errors_alt(loss_function=LossFunction.CAUCHY_LOSS)
    # ds00 = Dataset.with_noise(ds0, point2d_noise=0, point3d_noise=0)
    # benchmark_single_pose(ds00)
    ###################################
    print("Loading datasets")
    datasets = [
        {"dataset": partial(load_colmap_dataset, REICHSTAG_SPARSE_NOISED, REICHSTAG_IMAGES, binary=True,
                            name="Reichstag"),
         "batch_size": 75},
        {"dataset": partial(load_colmap_dataset, SACRE_COEUR_SPARSE_NOISED, SACRE_COEUR_IMAGES, binary=True,
                            name="Sacre Coeur"),
         "batch_size": 131},
        {"dataset": partial(load_colmap_dataset, ST_PETERS_SQUARE_SPARSE_NOISED, ST_PETERS_SQUARE_IMAGES, binary=True,
                            name="St. Peters Square"),
         "batch_size": 313}
    ]

    print("Adding noise")
    noisy_datasets = datasets  # list(map(lambda d: Dataset.with_noise_mp(d), datasets))

    evaluation = []
    for nd in noisy_datasets:
        dataset = nd["dataset"]()
        batch_size = nd["batch_size"]
        print(f"Benchmarking {str(dataset.name)}")
        problem_metadata = {
            "points2D_per_image": dataset.avg_num_2d_points_per_image(),
            "points3D_per_image": dataset.avg_num_3d_points_per_image(),
            "num_images": dataset.num_images(),
            "num_points3D": dataset.num_3d_points(),
        }
        #  statistics = benchmark_bundle_adjustment(dataset)
        statistics = benchmark_single_pose(dataset, batch_size=batch_size)
        eval = {**problem_metadata, **statistics}
        print("Evaluation:")
        print(eval)
        evaluation.append(eval)
        del dataset

    print(evaluation)
