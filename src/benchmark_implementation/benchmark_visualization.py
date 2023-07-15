import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.benchmark.benchmark import Benchmark, SinglePoseBenchmark
from src.config import BENCHMARK_SINGLE_POSE_RESULTS_PATH
from src.dataset.loss_functions import LossFunction


def save_reprojection_error_histogram(list_of_benchmarks):
    os.makedirs("../benchmark/evaluation", exist_ok=True)
    os.makedirs(f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}", exist_ok=True)
    reprojection_errors = []
    for benchmark in list_of_benchmarks:
        reprojection_error = benchmark.reprojection_errors(loss_function=LossFunction.CAUCHY_LOSS)
        reprojection_errors.append(reprojection_error)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    hist_data = np.histogram(reprojection_errors, bins="auto")
    # Filter counts of below 1% of top height to get new bins
    # threshold = np.max(hist_data[0]) * 0.005
    # indices = np.where(hist_data[0] >= threshold)[0]
    # bins = hist_data[1][indices]
    bins = hist_data[1]

    # filtered_reprojection_errors = []
    # for re in reprojection_errors:
    #     filtered_reprojection_errors.append(re[np.where(re <= bins[-1] + 5e-01)])
    filtered_reprojection_errors = reprojection_errors

    for re, b in list(zip(filtered_reprojection_errors, list_of_benchmarks)):
        ax.hist(re, bins=bins, alpha=1 / len(list_of_benchmarks), label=b.FRAMEWORK)
        # ax.axvline(re.mean(), color='k', linestyle='dashed', linewidth=1)
        # min_ylim, max_ylim = ax.get_ylim()
        # plt.text(re.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(re.mean()))
    ax.set_xlabel(f"Squared reprojection error")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"reprojection_error_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )


def save_runtime_plot(list_of_benchmarks):
    os.makedirs("../benchmark/evaluation", exist_ok=True)
    os.makedirs(f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}", exist_ok=True)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    # """ Full runtime """
    # cams = list(range(len(list_of_benchmarks[0].dataset.datasetEntries)))
    # for index, b in enumerate(list_of_benchmarks):  # Note this does not work if batch_size != 1 for now
    #     ax.bar(np.array(cams) + 0.25 * index, b.single_times, label=b.FRAMEWORK, width=0.25)
    # ax.set_xlabel(f"Cameras")
    # ax.set_ylabel("Execution time in s")
    # ax.legend(loc='upper right')
    # ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")
    #
    # fig.savefig(
    #     f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
    #     f"runtime_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
    #     f".png"
    # )

    """ mean runtime """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    num_cams = len(list_of_benchmarks[0].dataset.datasetEntries)
    names = list(map(lambda b: f"{b.FRAMEWORK}", list_of_benchmarks))
    ax.bar(names, list(map(lambda b: np.sum(b.single_times) / num_cams, list_of_benchmarks)))

    ax.set_xlabel(f"Frameworks")
    ax.set_ylabel("Mean execution time per camera in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"mean_runtime_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )

    # """ Optimization time """
    # fig: plt.Figure
    # ax: plt.Axes
    # fig, ax = plt.subplots()
    # cams = list(range(len(list_of_benchmarks[0].dataset.datasetEntries)))
    # for index, b in enumerate(list_of_benchmarks):
    #     # This has to be adjusted according to JAX
    #     optimization_time = b.single_times if type(b.time) == float else b.time[1]
    #     ax.bar(np.array(cams) + 0.25 * index, optimization_time, label=b.FRAMEWORK, width=0.25)
    # ax.set_xlabel(f"Cameras")
    # ax.set_ylabel("Execution time in s")
    # ax.legend(loc='upper right')
    # ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")
    #
    # fig.savefig(
    #     f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
    #     f"optimization_time_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
    #     f".png"
    # )

    """ mean optimization time """
    """ mean runtime """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    num_cams = len(list_of_benchmarks[0].dataset.datasetEntries)
    names = list(map(lambda b: f"{b.FRAMEWORK}", list_of_benchmarks))
    ax.bar(
        names,
        list(
            map(
                lambda b: np.sum(b.single_times) / num_cams if type(b.time) == float else np.sum(b.time[1]) / num_cams,
                list_of_benchmarks
            )
        )
    )

    ax.set_xlabel(f"Frameworks")
    ax.set_ylabel("Mean execution time per camera in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"mean_optimization_time_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}"
        f".png"
    )


def save_scatter_plot(list_of_benchmarks: List[SinglePoseBenchmark]):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    for index, b in enumerate(list_of_benchmarks):
        # This has to be adjusted according to JAX
        optimization_time = b.single_times if type(b.time) == float else b.time[1]
        num_3d_points = list(map(lambda dse: dse.num_3d_points, b.dataset.datasetEntries))
        a, _b = np.polyfit(num_3d_points, optimization_time, 1)
        ax.scatter(x=num_3d_points, y=optimization_time, alpha=1 / len(list_of_benchmarks), label=b.FRAMEWORK)
        ax.plot(num_3d_points, a * np.array(num_3d_points) + _b, alpha=1 / len(list_of_benchmarks))

    ax.set_xlabel(f"Number of 2d-3d correspondences")
    ax.set_ylabel("Optimization time in s")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"scatter_plot_optimization_time_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )


def save_iteration_plot(list_of_benchmarks: List[SinglePoseBenchmark]):
    os.makedirs("../benchmark/evaluation", exist_ok=True)
    os.makedirs(f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}", exist_ok=True)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    names = list(map(lambda b: f"{b.FRAMEWORK}", list_of_benchmarks))
    values = list(map(lambda b: np.mean(b.iterations), list_of_benchmarks))

    bars = ax.bar(names, values)
    ax.bar_label(bars)

    ax.set_xlabel(f"Frameworks")
    ax.set_ylabel("Mean number of iterations time per camera")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"mean_num_iterations_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}"
        f".png"
    )

    """ per camera """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    cams = list(range(len(list_of_benchmarks[0].dataset.datasetEntries)))
    for index, b in enumerate(list_of_benchmarks):  # Note this does not work if batch_size != 1 for now
        ax.bar(np.array(cams) + 0.25 * index, b.iterations, label=b.FRAMEWORK, width=0.25)
    ax.set_xlabel(f"Cameras")
    ax.set_ylabel("Number of iterations")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"num_iterations_plot_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )

    """ per camera histogram """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()

    iterations_total = [i for li in list_of_benchmarks for i in li.iterations]
    unique = set(iterations_total)
    hist_data = np.histogram(np.array(iterations_total), bins=len(unique))
    for b in list_of_benchmarks:
        ax.hist(b.iterations, bins=hist_data[1], alpha=1 / len(list_of_benchmarks),
                label=b.FRAMEWORK + f" (Median: {np.median(b.iterations)})")
        # ax.axvline(re.mean(), color='k', linestyle='dashed', linewidth=1)
        # min_ylim, max_ylim = ax.get_ylim()
        # plt.text(re.mean() * 1.1, max_ylim * 0.9, 'Mean: {:.2f}'.format(re.mean()))
    ax.set_xlabel(f"Number of iterations")
    ax.set_ylabel("Count")
    ax.legend(loc='upper right')
    ax.set_title(f"SinglePoseBenchmark ({list_of_benchmarks[0].dataset.name})")

    fig.savefig(
        f"evaluation/{list_of_benchmarks[0].dataset.name.replace(' ', '_').lower()}/{list_of_benchmarks[0].NAME.replace(' ', '_').lower() + '_'}"
        f"num_iterations_histogram_{list_of_benchmarks[0].dataset.name.replace(' ', '').lower()}"
        f".png"
    )


def single_pose_statistics(list_of_benchmarks: List[SinglePoseBenchmark]):
    save_reprojection_error_histogram(list_of_benchmarks)
    save_runtime_plot(list_of_benchmarks)
    save_iteration_plot(list_of_benchmarks)
    #  if any([isinstance(b, JaxoptSinglePoseBenchmark) for b in list_of_benchmarks]):
    #  save_scatter_plot(list_of_benchmarks)

    #  Camera.difference(list(cr.camera_mapping.values())[0], list(jr.camera_mapping.values())[0])
    #  colmapSinglePoseBenchmark.export_results_in_colmap_format(open_in_colmap=True)
    #  jaxopt_benchmark.export_results_in_colmap_format(open_in_colmap=True)


if __name__ == "__main__":
    latest_single_pose_benchmarks = Benchmark.load_pickle_folder(
        os.path.join(BENCHMARK_SINGLE_POSE_RESULTS_PATH, "latest")
    )
    single_pose_statistics(latest_single_pose_benchmarks)
