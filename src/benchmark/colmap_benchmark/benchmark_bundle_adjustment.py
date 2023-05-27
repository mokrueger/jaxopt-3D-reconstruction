import os
import shutil

from src.benchmark.colmap_benchmark.bundle_adjuster import perform_bundle_adjustment, _process_std_out
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset, export_in_colmap_format


def benchmark_colmap_bundle_adjustment(dataset: Dataset, add_noise=True, refine_focal_length=False,
                                       validate_result=True, validation_error_position=5e-02,
                                       validation_error_rotation=1e-02):
    noise_dataset = Dataset.with_noise(dataset) if add_noise else dataset
    os.makedirs("benchmark_input", exist_ok=True)
    export_in_colmap_format(noise_dataset, "benchmark_input")

    time = 0.0
    number = 5
    for n in range(number):
        os.makedirs("benchmark_output", exist_ok=True)
        std_out = perform_bundle_adjustment("benchmark_input", "benchmark_output", )
        bundle_adjustment_report = _process_std_out(std_out)
        time += bundle_adjustment_report.time
        shutil.rmtree("benchmark_output")

    shutil.rmtree("benchmark_input")
    return time / number


if __name__ == "__main__":
    path = "/home/morkru/Downloads/reichstag/dense/sparse/TXT"
    image_path = "/home/morkru/Downloads/reichstag/dense/images"
    ds = load_colmap_dataset(path, image_path, binary=False)
    time1 = benchmark_colmap_bundle_adjustment(ds)
    print("finished")
