import os
import shutil

from src.benchmark.benchmark import (
    BundleAdjustmentBenchmark,
    BundleAdjustmentBenchmarkResults,
)
from src.benchmark.colmap_benchmark.bundle_adjuster import (
    _process_std_out,
    perform_bundle_adjustment,
)
from src.config import DATASETS_PATH
from src.dataset.loaders.colmap_dataset_loader.cameras import read_cameras_bin
from src.dataset.loaders.colmap_dataset_loader.images import read_images_bin
from src.dataset.loaders.colmap_dataset_loader.loader import (
    _parse_cameras_only,
    _parse_points,
    export_in_colmap_format,
    load_colmap_dataset,
)
from src.dataset.loaders.colmap_dataset_loader.points import read_points3d_bin


class ColmapBundleAdjustmentBenchmark(BundleAdjustmentBenchmark):
    FRAMEWORK = "Colmap"

    def benchmark(self, *args, **kwargs):
        os.makedirs("benchmark_input", exist_ok=True)
        export_in_colmap_format(self.dataset, "benchmark_input", binary=True)

        self_reported_time = 0.0
        measured_time = 0.0
        os.makedirs("benchmark_output", exist_ok=True)
        std_out, t = perform_bundle_adjustment(
            "benchmark_input",
            "benchmark_output",
        )
        bundle_adjustment_report = _process_std_out(std_out)
        self_reported_time += bundle_adjustment_report.time
        measured_time += t

        result_points = {
            p.identifier: p
            for p in _parse_points(
                read_points3d_bin(os.path.join("benchmark_output", "points3D.bin"))
            )
        }
        result_cameras = _parse_cameras_only(
            images=read_images_bin(os.path.join("benchmark_output", "images.bin")),
            cameras=read_cameras_bin(os.path.join("benchmark_output", "cameras.bin")),
            path_to_images=self.dataset.images_path,
        )
        # Since colmap starts with non-zero indices we have to undo that by subtracting 1 from each id
        result_cameras = {k - 1: v for k, v in result_cameras.items()}

        shutil.rmtree("benchmark_output")

        shutil.rmtree("benchmark_input")
        self._results = BundleAdjustmentBenchmarkResults(
            camera_mapping=result_cameras, point_mapping=result_points
        )
        self._time = self_reported_time
        self._iterations = bundle_adjustment_report.iterations


if __name__ == "__main__":
    path = os.path.join(DATASETS_PATH, "reichstag/sparse/TXT")
    image_path = os.path.join(DATASETS_PATH, "reichstag/images")
    ds = load_colmap_dataset(path, image_path, binary=False)

    colmapBundleAdjustmentBenchmark = ColmapBundleAdjustmentBenchmark(ds)
    colmapBundleAdjustmentBenchmark.benchmark()
    time1 = colmapBundleAdjustmentBenchmark.time
    print("finished")
