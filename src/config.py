import os
from pathlib import Path

DATASETS_PATH = os.path.join(str(Path(__file__).parent.parent), "datasets")
BENCHMARK_RESULTS_PATH = os.path.join(
    str(Path(__file__).parent.parent), "benchmark_results"
)
BENCHMARK_SINGLE_POSE_RESULTS_PATH = os.path.join(BENCHMARK_RESULTS_PATH, "single_pose")
BENCHMARK_BUNDLE_ADJUSTMENT_RESULTS_PATH = os.path.join(
    BENCHMARK_RESULTS_PATH, "bundle_adjustment"
)
# TODO: Here also colmap cmd path
