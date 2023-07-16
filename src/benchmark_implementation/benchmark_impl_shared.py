import contextlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from src.benchmark.benchmark import Benchmark


def save_benchmarks(
    list_of_benchmarks: List[Benchmark], parent_dir, override_latest=True
):
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
