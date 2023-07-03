"""
This is where the code for the comparison between the three methods goes
"""
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from matplotlib import pyplot as plt

from src.benchmark.benchmark import SinglePoseBenchmark, Benchmark
from src.benchmark.colmap_benchmark.benchmark_bundle_adjustment import (
    ColmapBundleAdjustmentBenchmark,
)
from src.benchmark.colmap_benchmark.benchmark_single_pose import (
    ColmapSinglePoseBenchmark,
)
from src.benchmark.gtsam_benchmark.benchmark_bundle_adjustment import (
    GtsamBundleAdjustmentBenchmark,
)
from src.benchmark.jaxopt_benchmark.benchmark_batched_pose import JaxoptSinglePoseBenchmarkBatched
from src.benchmark.jaxopt_benchmark.benchmark_single_pose import JaxoptSinglePoseBenchmark
from src.dataset.camera import Camera
from src.dataset.dataset import Dataset

#  from src.benchmark.gtsam_benchmark.benchmark_single_pose import import benchmark_gtsam_single_pose
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset

DATASET_FOLDER = os.path.join(str(Path(__file__).parent.parent.parent), "datasets")
REICHSTAG_SPARSE_NOISED = os.path.join(DATASET_FOLDER, "reichstag/sparse_noised")
REICHSTAG_SPARSE = os.path.join(DATASET_FOLDER, "reichstag/sparse")
REICHSTAG_IMAGES = os.path.join(DATASET_FOLDER, "reichstag/images")

SACRE_COEUR_SPARSE_NOISED = os.path.join(DATASET_FOLDER, "sacre_coeur/sparse_noised")
SACRE_COEUR_IMAGES = os.path.join(DATASET_FOLDER, "sacre_coeur/images")

ST_PETERS_SQUARE_SPARSE_NOISED = os.path.join(DATASET_FOLDER, "st_peters_square/sparse_noised")
ST_PETERS_SQUARE_IMAGES = os.path.join(DATASET_FOLDER, "st_peters_square/images")


def save_reprojection_error_histogram(list_of_benchmarks):
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}", exist_ok=True)
    reprojection_errors = []
    for benchmark in list_of_benchmarks:
        _, _, reprojection_error = benchmark.reprojection_error_histogram()
        reprojection_errors.append(reprojection_error)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    hist_data = np.histogram(reprojection_errors, bins="auto")
    # Filter counts of below 1% of top height to get new bins
    new_hist_data = [[], []]
    for index in range(len(hist_data[0])):
        if hist_data[0][index] < max(hist_data[0]) * 0.01:
            continue
        new_hist_data[0].append(hist_data[0][index])
        new_hist_data[1].append(hist_data[1][index])
    new_hist_data = (np.array(new_hist_data[0]), np.array(new_hist_data[1]))
    bins = new_hist_data[1]

    filtered_reprojection_errors = []
    for re in reprojection_errors:
        filtered_reprojection_errors.append(list(filter(lambda x: x <= bins[-1] + 5e-01, re)))

    for re, b in list(zip(filtered_reprojection_errors, list_of_benchmarks)):
        ax.hist(re, bins=bins, alpha=1 / len(list_of_benchmarks), label=b.FRAMEWORK)
    ax.set_xlabel(f"Squared reprojection error (meters)")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"reprojection_error_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )


def save_runtime_plot(list_of_benchmarks):
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs(f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}", exist_ok=True)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    """ Full runtime """
    cams = list(range(len(list_of_benchmarks[0].dataset.datasetEntries)))
    for index, b in enumerate(list_of_benchmarks):
        ax.bar(np.array(cams) + 0.25 * index, b.single_times, label=b.FRAMEWORK, width=0.25)
    ax.set_xlabel(f"Cameras")
    ax.set_ylabel("Execution time in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"runtime_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )

    """ mean runtime """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    names = list(map(lambda b: f"{b.FRAMEWORK}", list_of_benchmarks))
    ax.bar(names, list(map(lambda b: np.mean(b.single_times), list_of_benchmarks)))

    ax.set_xlabel(f"Frameworks")
    ax.set_ylabel("Mean execution time per camera in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"mean_runtime_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )

    """ Optimization time """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    cams = list(range(len(list_of_benchmarks[0].dataset.datasetEntries)))
    for index, b in enumerate(list_of_benchmarks):
        # This has to be adjusted according to JAX
        optimization_time = b.single_times if type(b.time) == float else b.time[1]
        ax.bar(np.array(cams) + 0.25 * index, optimization_time, label=b.FRAMEWORK, width=0.25)
    ax.set_xlabel(f"Cameras")
    ax.set_ylabel("Execution time in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"optimization_time_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )

    """ mean optimization time """
    """ mean runtime """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    names = list(map(lambda b: f"{b.FRAMEWORK}", list_of_benchmarks))
    ax.bar(names, list(
        map(lambda b: np.mean(b.single_times) if type(b.time) == float else np.mean(b.time[1]), list_of_benchmarks)))

    ax.set_xlabel(f"Frameworks")
    ax.set_ylabel("Mean execution time per camera in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"mean_optimization_time_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}"
        f".png"
    )


def single_pose_statistics(list_of_benchmarks: List[Benchmark]):
    save_runtime_plot(list_of_benchmarks)
    save_reprojection_error_histogram(list_of_benchmarks)

    #  Camera.difference(list(cr.camera_mapping.values())[0], list(jr.camera_mapping.values())[0])
    #  colmapSinglePoseBenchmark.export_results_in_colmap_format(open_in_colmap=True)
    #  jaxopt_benchmark.export_results_in_colmap_format(open_in_colmap=True)
    pass


def benchmark_single_pose(dataset):
    # jaxopt_benchmark = JaxoptSinglePoseBenchmark(dataset)

    print("Benchmarking JAX")
    jaxopt_benchmark_batched = JaxoptSinglePoseBenchmarkBatched(dataset)
    jaxopt_benchmark_batched.subprocess_benchmark(verbose=False)
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

    single_pose_statistics([jaxopt_benchmark_batched, colmapSinglePoseBenchmark])
    return {
        "colmap": colmapSinglePoseBenchmark.time,
        "jax": (total_c, total_o, total_t, total_e),
    }


def benchmark_bundle_adjustment(dataset):
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
    # ds00 = Dataset.with_noise(ds0, point2d_noise=0, point3d_noise=0)
    # benchmark_single_pose(ds00)
    ###################################
    print("Loading datasets")
    datasets = [
        partial(load_colmap_dataset, REICHSTAG_SPARSE_NOISED, REICHSTAG_IMAGES, binary=True, name="Reichstag"),
        partial(load_colmap_dataset, SACRE_COEUR_SPARSE_NOISED, SACRE_COEUR_IMAGES, binary=True, name="Sacre Coeur"),
        partial(load_colmap_dataset, ST_PETERS_SQUARE_SPARSE_NOISED, ST_PETERS_SQUARE_IMAGES, binary=True,
                name="St. Peters Square")
    ]

    print("Adding noise")
    noisy_datasets = datasets  # list(map(lambda d: Dataset.with_noise_mp(d), datasets))

    evaluation = []
    for nd in noisy_datasets:
        dataset = nd()
        print(f"Benchmarking {str(dataset.name)}")
        problem_metadata = {
            "points2D_per_image": dataset.avg_num_2d_points_per_image(),
            "points3D_per_image": dataset.avg_num_3d_points_per_image(),
            "num_images": dataset.num_images(),
            "num_points3D": dataset.num_3d_points(),
        }
        #  statistics = benchmark_bundle_adjustment(dataset)
        statistics = benchmark_single_pose(dataset)
        eval = {**problem_metadata, **statistics}
        print("Evaluation:")
        print(eval)
        evaluation.append(eval)
        del dataset

    print(evaluation)
