import copy
import multiprocessing
import os
import time

from matplotlib import pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union
from uuid import uuid4

import numpy as np

from src.dataset.camera import Camera
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import export_in_colmap_format, show_in_colmap
from src.dataset.point import Point3D


class Benchmark(ABC):
    NAME = "Benchmark"
    FRAMEWORK = "Framework"

    @abstractmethod
    def benchmark(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def results(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def time(self):
        raise NotImplementedError


@dataclass
class SinglePoseBenchmarkResults:
    camera_mapping: Dict[Union[str, int], Camera]


class SinglePoseBenchmark(Benchmark, ABC):
    NAME = "Single Pose Benchmark"

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._results = None
        self._time = None
        self._single_times = None

    @property
    def results(self) -> SinglePoseBenchmarkResults:
        if self._results:
            return self._results
        raise AttributeError

    @property
    def time(self):
        if self._time:
            return self._time
        raise AttributeError

    @property
    def single_times(self):
        if self._single_times:
            return self._single_times
        raise AttributeError

    def subprocess_benchmark(self, benchmark_function_name="benchmark", *args, **kwargs, ):
        """
        Args:
            benchmark_function_name: function name of to-be-called benchmark function (for e.g. in-dev functions)
            *args (object): args passed to benchmark function
            **kwargs (object): kwargs passed to benchmark function
        """

        def execute_subprocess_benchmark(dataset, queue: multiprocessing.Queue, function_name: str,
                                         *ar, **kw):
            subprocess_benchmark_class = self.__class__(copy.copy(dataset))
            benchmark_function = getattr(subprocess_benchmark_class, function_name)
            benchmark_function(*ar, **kw)
            queue.put(subprocess_benchmark_class.results)
            queue.put(subprocess_benchmark_class.time)
            queue.put(subprocess_benchmark_class.single_times)
            print("Process exiting")
            exit(0)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=execute_subprocess_benchmark,
                                    args=args,
                                    kwargs={
                                        "queue": q,
                                        "dataset": self.dataset,
                                        "function_name": benchmark_function_name,
                                        **kwargs,
                                    })
        p.start()
        item_count = 0
        items = []
        while item_count != 3:  # Join does not work when putting large objects in queue.
            if q.empty():
                time.sleep(5)
            else:
                if item_count == 0:
                    print("transferring items")
                items.append(q.get())
                item_count += 1
        p.join()  # Now it can exit and join properly.
        if p.exitcode != 0:
            raise Exception("An unknown exception happened.")
        self._results = copy.deepcopy(items[0])
        self._time = copy.deepcopy(items[1])
        self._single_times = copy.deepcopy(items[2])

    def shallow_results_dataset(self):  # Note: everything (excluding cameras) points to the original dataset(!!)
        if self._results:
            copied_dataset = copy.copy(self.dataset)  # Shallow copy(!) is enough to export, but be really careful here
            copied_dataset.datasetEntries = list(map(lambda x: copy.copy(x), copy.copy(copied_dataset.datasetEntries)))
            # Since only the camera changes we can substitute it with the by the new cameras
            camera_mapping = self.results.camera_mapping
            for index, de in enumerate(copied_dataset.datasetEntries):
                de.camera = camera_mapping.get(index)
            return copied_dataset
        raise AttributeError

    def export_results_in_colmap_format(self, output_path="export_results/" + str(uuid4()), open_in_colmap=False):
        os.makedirs(output_path, exist_ok=True)
        shallow_results_dataset = self.shallow_results_dataset()
        export_in_colmap_format(shallow_results_dataset, output_path, binary=True)
        if open_in_colmap:
            show_in_colmap(output_path, shallow_results_dataset.images_path, block=False)

    # def reprojection_error_histogram(self, loss_function=lambda x: x, show=False): # TODO: DEPRECATED!!
    #     if self._results:
    #         dataset = self.shallow_results_dataset()
    #         #  reprojection_errors = dataset.compute_reprojection_errors()  # TODO: here loss_function
    #         reprojection_errors = dataset.compute_reprojection_errors_alt()
    #         reprojection_errors_list = np.array(
    #             [item for sublist in list(reprojection_errors.values()) for item in sublist]
    #         )
    #         fig: plt.Figure;
    #         ax: plt.Axes
    #         fig, ax = plt.subplots()
    #         ax.hist(reprojection_errors_list, bins="auto")
    #         ax.set_xlabel(f"Squared reprojection error (meters)")
    #         ax.set_ylabel("Count")
    #         ax.set_title(f"{self.NAME} ({self.dataset.name})")
    #         if show:
    #             plt.show()
    #         return fig, ax, reprojection_errors_list
    #     raise AttributeError

    def reprojection_errors(self, loss_function=lambda x: x, show=False):
        if self._results:
            dataset = self.shallow_results_dataset()
            #  reprojection_errors = dataset.compute_reprojection_errors()  # TODO: here loss_function
            reprojection_errors = dataset.compute_reprojection_errors_alt()
            reprojection_errors_list = np.array(
                [item for sublist in list(reprojection_errors.values()) for item in sublist]
            )
            return reprojection_errors_list
        raise AttributeError


@dataclass
class BundleAdjustmentBenchmarkResults:
    camera_mapping: Dict[Union[str, int], Camera]
    point_mapping: Dict[Union[str, int], Point3D]


class BundleAdjustmentBenchmark(Benchmark, ABC):
    NAME = "Bundle Adjustment Benchmark"

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._results = None
        self._time = None

    @property
    def results(self) -> BundleAdjustmentBenchmarkResults:
        if self._results:
            return self.results
        raise AttributeError

    @property
    def time(self):
        if self._time:
            return self._time
        raise AttributeError
