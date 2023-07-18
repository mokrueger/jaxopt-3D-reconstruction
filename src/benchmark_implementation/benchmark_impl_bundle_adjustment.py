"""
This is where the code for the comparison between the three methods goes
"""
import os

from src.benchmark.colmap_benchmark.benchmark_bundle_adjustment import (
    ColmapBundleAdjustmentBenchmark,
)
from src.benchmark.gtsam_benchmark.benchmark_bundle_adjustment import (
    GtsamBundleAdjustmentBenchmark,
)
from src.benchmark.jaxopt_benchmark.benchmark_bundle_adjustment import (
    JaxoptBundleAdjustmentBenchmark,
)
from src.benchmark_implementation.benchmark_datasets import (
    REICHSTAG_NOISED_LOADER,
    SACRE_COEUR_NOISED_LOADER,
    ST_PETERS_SQUARE_NOISED_LOADER,
)
from src.benchmark_implementation.benchmark_impl_shared import save_benchmarks
from src.config import BENCHMARK_BUNDLE_ADJUSTMENT_RESULTS_PATH

#  from src.benchmark.gtsam_benchmark.benchmark_single_pose import import benchmark_gtsam_single_pose
from src.dataset.loss_functions import LossFunction


def benchmark_bundle_adjustment(dataset):
    points_limit = 400
    camera_limit = 15

    jaxopt_benchmark = JaxoptBundleAdjustmentBenchmark(dataset)
    jaxopt_benchmark.benchmark(points_limit=points_limit, camera_limit=camera_limit)

    colmap_benchmark = ColmapBundleAdjustmentBenchmark(dataset)
    colmap_benchmark.benchmark(verbose=True, points_limit=points_limit, camera_limit=camera_limit)

    gtsam_benchmark = GtsamBundleAdjustmentBenchmark(dataset)
    gtsam_benchmark.benchmark()

    """ DEBUG STUFF (remove later) """
    import numpy as np

    jds_errors = (
        jaxopt_benchmark.shallow_results_dataset(
            point_limit=jaxopt_benchmark.points_limit,
            only_trimmed_2d_points=True
        ).compute_reprojection_errors_alt(
            LossFunction.TRIVIAL_LOSS
        )
    )
    cds_errors = colmap_benchmark.shallow_results_dataset(
        points_limit=points_limit,
        camera_limit=camera_limit,
        only_trimmed_2d_points=True
    ).compute_reprojection_errors_alt(
        LossFunction.TRIVIAL_LOSS
    )
    gds_errors = {
        k: v
        for k, v in gtsam_benchmark.shallow_results_dataset()
        .compute_reprojection_errors_alt(LossFunction.TRIVIAL_LOSS)
        .items()
        if k in jds_errors.keys()
    }
    jds_errors_avg = {k: np.mean(v) for k, v in jds_errors.items()}
    cds_errors_avg = {k: np.mean(v) for k, v in cds_errors.items()}
    gds_errors_avg = {k: np.mean(v) for k, v in gds_errors.items()}

    # jaxopt_benchmark.export_results_in_colmap_format(points_limit=jaxopt_benchmark.points_limit,
    #                                                  cameras_limit=camera_limit, open_in_colmap=True)
    # colmap_benchmark.export_results_in_colmap_format(points_limit=points_limit,
    #                                                  cameras_limit=camera_limit, open_in_colmap=True)
    # gtsam_benchmark.export_results_in_colmap_format(open_in_colmap=True)
    """ DEBUG END"""

    """ Save benchmarks """
    save_benchmarks(
        [jaxopt_benchmark, colmap_benchmark, gtsam_benchmark],
        os.path.join(BENCHMARK_BUNDLE_ADJUSTMENT_RESULTS_PATH),
        override_latest=True,
    )
    return {
        "colmap_time": colmap_benchmark.time,
        "colmap_results": colmap_benchmark.results,
        # "gtsam_time": gtsam_benchmark.time,
        # "gtsam_results": gtsam_benchmark.results,
    }


if __name__ == "__main__":
    print("Loading datasets")
    noisy_datasets = [
        REICHSTAG_NOISED_LOADER,
        #  SACRE_COEUR_NOISED_LOADER,
        #  ST_PETERS_SQUARE_NOISED_LOADER,
    ]

    evaluation = []
    for nd in noisy_datasets:
        dataset = nd()
        print(f"Benchmarking {str(dataset.name)}")
        statistics = benchmark_bundle_adjustment(dataset)
        eval = {**statistics}
        print("Evaluation:")
        print(eval)
        evaluation.append(eval)
        del dataset

    print(evaluation)
