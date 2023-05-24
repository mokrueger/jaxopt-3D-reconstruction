"""
This is where the code for the comparison between the three methods goes
"""
from src.benchmark.colmap_benchmark.benchmark_single_pose import benchmark_colmap_absolute_pose
from src.benchmark.colmap_benchmark.benchmark_bundle_adjustment import benchmark_colmap_bundle_adjustment
from src.benchmark.gtsam_benchmark.benchmark_bundle_adjustment import benchmark_gtsam_bundle_adjustment
from src.dataset.dataset import Dataset

#  from src.benchmark.gtsam_benchmark.benchmark_single_pose import import benchmark_gtsam_single_pose
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset

REICHSTAG_SPARSE = "/home/morkru/Downloads/reichstag/dense/sparse/"
REICHSTAG_IMAGES = "/home/morkru/Downloads/reichstag/dense/images"

SACRE_COEUR_SPARSE = "/home/morkru/Downloads/sacre_coeur/dense/sparse/"
SACRE_COEUR_IMAGES = "/home/morkru/Downloads/sacre_coeur/dense/images/"


def benchmark_single_pose(dataset):
    colmap_time = benchmark_colmap_absolute_pose(dataset, add_noise=False)
    #  gtsam_time = benchmark_gtsam_single_pose(dataset, add_noise=False)
    return {
        "colmap_time": colmap_time,
        #  "gtsam_time": gtsam_time,
    }


def benchmark_bundle_adjustment(dataset):
    colmap_time = benchmark_colmap_bundle_adjustment(dataset, add_noise=False, validate_result=False)
    #  gtsam_time = benchmark_gtsam_bundle_adjustment(dataset, add_noise=False)
    return {
        "colmap_time": colmap_time,
        #  "gtsam_time": gtsam_time,
    }


if __name__ == "__main__":
    print("Loading datasets")
    datasets = [
        #  load_colmap_dataset(REICHSTAG_SPARSE, REICHSTAG_IMAGES, binary=True, name="Reichstag"),
        load_colmap_dataset(SACRE_COEUR_SPARSE, SACRE_COEUR_IMAGES, binary=True, name="Sacre Coeur")
    ]

    print("Adding noise")
    noisy_datasets = list(map(lambda d: Dataset.with_noise(d), datasets))

    evaluation = []
    for nd in noisy_datasets:
        print(f"Benchmarking {str(nd.name)}")
        problem_metadata = {
            "points2D_per_image": nd.avg_num_2d_points_per_image(),
            "points3D_per_image": nd.avg_num_3d_points_per_image(),
            "num_images": nd.num_images(),
            "num_points3D": nd.num_3d_points()
        }
        statistics = benchmark_bundle_adjustment(nd)
        evaluation.append({
            **problem_metadata,
            **statistics
        })

    print("finished")
