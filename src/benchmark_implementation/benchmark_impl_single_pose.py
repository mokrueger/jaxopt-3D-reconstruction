"""
This is where the code for the comparison between the three methods goes
"""
import os

from src.benchmark.colmap_benchmark.benchmark_single_pose import (
    ColmapSinglePoseBenchmark,
)
from src.benchmark.jaxopt_benchmark.benchmark_pose_optimization import JaxoptSinglePoseBenchmarkBatched
from src.benchmark_implementation.benchmark_datasets import REICHSTAG_NOISED_LOADER, SACRE_COEUR_NOISED_LOADER, \
    ST_PETERS_SQUARE_NOISED_LOADER
from src.benchmark_implementation.benchmark_impl_shared import save_benchmarks
from src.benchmark_implementation.benchmark_visualization import single_pose_statistics
from src.config import BENCHMARK_SINGLE_POSE_RESULTS_PATH


#  from src.benchmark.gtsam_benchmark.benchmark_single_pose import import benchmark_gtsam_single_pose


def benchmark_single_pose(dataset, **kwargs):
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


if __name__ == "__main__":

    # ds0 = load_colmap_dataset(REICHSTAG_SPARSE, REICHSTAG_IMAGES, binary=True, name="Reichstag Original")
    # ds0.compute_reprojection_errors_alt(loss_function=LossFunction.CAUCHY_LOSS)
    # ds00 = Dataset.with_noise(ds0, point2d_noise=0, point3d_noise=0)
    # benchmark_single_pose(ds00)
    ###################################
    print("Loading datasets")
    noisy_datasets = [
        {"dataset": REICHSTAG_NOISED_LOADER, "batch_size": 1},
        {"dataset": SACRE_COEUR_NOISED_LOADER, "batch_size": 1},
        {"dataset": ST_PETERS_SQUARE_NOISED_LOADER, "batch_size": 1}
    ]

    evaluation = []
    for nd in noisy_datasets:
        dataset = nd["dataset"]()
        batch_size = nd["batch_size"]
        print(f"Benchmarking {str(dataset.name)}")
        statistics = benchmark_single_pose(dataset, batch_size=batch_size)
        eval = {**statistics}
        print("Evaluation:")
        print(eval)
        evaluation.append(eval)
        del dataset

    print(evaluation)
