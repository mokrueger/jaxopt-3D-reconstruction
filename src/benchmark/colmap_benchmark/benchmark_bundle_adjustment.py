import os
import shutil

from src.benchmark.benchmark import BundleAdjustmentBenchmark, BundleAdjustmentBenchmarkResults
from src.benchmark.colmap_benchmark.bundle_adjuster import perform_bundle_adjustment, _process_std_out
from src.dataset.loaders.colmap_dataset_loader.cameras import read_cameras_bin
from src.dataset.loaders.colmap_dataset_loader.images import read_images_bin
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset, export_in_colmap_format, \
    _parse_cameras_only, _parse_points
from src.dataset.loaders.colmap_dataset_loader.points import read_points3d_bin


class ColmapBundleAdjustmentBenchmark(BundleAdjustmentBenchmark):
    FRAMEWORK = "Colmap"

    def benchmark(self, refine_focal_length=False,
                  validate_result=True, validation_error_position=5e-02,
                  validation_error_rotation=1e-02):
        os.makedirs("benchmark_input", exist_ok=True)
        export_in_colmap_format(self.dataset, "benchmark_input", binary=True)

        self_reported_time = 0.0
        measured_time = 0.0
        os.makedirs("benchmark_output", exist_ok=True)
        std_out, t = perform_bundle_adjustment("benchmark_input", "benchmark_output", )
        bundle_adjustment_report = _process_std_out(std_out)
        self_reported_time += bundle_adjustment_report.time
        measured_time += t
        result_points = {p.identifier: p for p in
                         _parse_points(read_points3d_bin(os.path.join("benchmark_output", "points3D.bin")))}
        result_cameras = _parse_cameras_only(
            images=read_images_bin(os.path.join("benchmark_output", "images.bin")),
            cameras=read_cameras_bin(os.path.join("benchmark_output", "cameras.bin")),
            path_to_images=self.dataset.images_path
        )
        shutil.rmtree("benchmark_output")

        shutil.rmtree("benchmark_input")
        self._results = BundleAdjustmentBenchmarkResults(camera_mapping=result_cameras,
                                                         point_mapping=result_points)
        self._time = self_reported_time


if __name__ == "__main__":
    path = "/home/morkru/Desktop/Github/jaxopt-3D-reconstruction/datasets/reichstag/sparse/TXT"
    image_path = "/home/morkru/Desktop/Github/jaxopt-3D-reconstruction/datasets/reichstag/images"
    ds = load_colmap_dataset(path, image_path, binary=False)

    colmapBundleAdjustmentBenchmark = ColmapBundleAdjustmentBenchmark(ds)
    colmapBundleAdjustmentBenchmark.benchmark()
    time1 = colmapBundleAdjustmentBenchmark.time
    print("finished")
