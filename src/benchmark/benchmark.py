import copy
import multiprocessing
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, Tuple, List
from uuid import uuid4

import numpy as np
from matplotlib import pyplot as plt

from src.dataset.camera import Camera
from src.dataset.dataset import Dataset
from src.dataset.loaders.colmap_dataset_loader.loader import (
    export_in_colmap_format,
    show_in_colmap,
)
from src.dataset.loss_functions import LossFunction
from src.dataset.point import Point3D


class Benchmark(ABC):
    NAME = "Benchmark"
    FRAMEWORK = "Framework"

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.benchmark_args_kwargs: Tuple[List, Dict] = ([], {})

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

    def export_pickle(self, full_path_to_folder, filename=None) -> str:
        os.makedirs(full_path_to_folder, exist_ok=True)
        if not filename:
            filename = (
                    self.__class__.__name__
                    + f"_{self.dataset.name.replace(' ', '_')}"
                    + ".pkl"
            )
        full_filename = os.path.abspath(os.path.join(full_path_to_folder, filename))
        with open(full_filename, "wb") as f:
            pickle.dump(self, f)
        return full_filename

    @staticmethod
    def load_pickle(path_to_pickle_obj):
        with open(path_to_pickle_obj, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_pickle_folder(path_to_pickle_folder):
        files = [
            os.path.join(path_to_pickle_folder, f)
            for f in os.listdir(path_to_pickle_folder)
        ]
        return list(map(Benchmark.load_pickle, files))


@dataclass
class SinglePoseBenchmarkResults:
    camera_mapping: Dict[Union[str, int], Camera]


class SinglePoseBenchmark(Benchmark, ABC):
    NAME = "Single Pose Benchmark"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._results = None
        self._time = None
        self._single_times = None
        self._iterations = None

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

    @property
    def iterations(self):
        if self._iterations:
            return self._iterations
        raise AttributeError

    def subprocess_benchmark(
            self,
            benchmark_function_name="benchmark",
            *args,
            **kwargs,
    ):
        """
        Args:
            benchmark_function_name: function name of to-be-called benchmark function (for e.g. in-dev functions)
            *args (object): args passed to benchmark function
            **kwargs (object): kwargs passed to benchmark function
        """

        def execute_subprocess_benchmark(
                dataset, queue: multiprocessing.Queue, function_name: str, *ar, **kw
        ):
            subprocess_benchmark_class = self.__class__(copy.copy(dataset))
            benchmark_function = getattr(subprocess_benchmark_class, function_name)
            benchmark_function(*ar, **kw)
            queue.put(subprocess_benchmark_class.results)
            queue.put(subprocess_benchmark_class.time)
            queue.put(subprocess_benchmark_class.single_times)
            queue.put(subprocess_benchmark_class.iterations)
            print("Process exiting")
            exit(0)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=execute_subprocess_benchmark,
            args=args,
            kwargs={
                "queue": q,
                "dataset": self.dataset,
                "function_name": benchmark_function_name,
                **kwargs,
            },
        )
        p.start()
        item_count = 0
        items = []
        while (
                item_count != 4
        ):  # We do this because join does not work when putting large objects in queue.
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
        self._iterations = copy.deepcopy(items[3])

    def shallow_results_dataset(
            self,
    ):  # Note: everything (excluding cameras) points to the original dataset(!!)
        if self._results:
            copied_dataset = copy.copy(
                self.dataset
            )  # Shallow copy(!) is enough to export, but be really careful here
            copied_dataset.datasetEntries = list(
                map(lambda x: copy.copy(x), copy.copy(copied_dataset.datasetEntries))
            )
            # Since only the camera changes we can substitute it with the by the new cameras
            camera_mapping = self.results.camera_mapping
            for index, de in enumerate(copied_dataset.datasetEntries):
                de.camera = camera_mapping.get(index)
            return copied_dataset
        raise AttributeError

    def export_results_in_colmap_format(
            self, output_path=None, open_in_colmap=False
    ):
        if not output_path:
            output_path = "export_results/" + str(uuid4())
        os.makedirs(output_path, exist_ok=True)
        shallow_results_dataset = self.shallow_results_dataset()
        export_in_colmap_format(shallow_results_dataset, output_path, binary=True)
        if open_in_colmap:
            show_in_colmap(
                output_path, shallow_results_dataset.images_path, block=False
            )

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

    def reprojection_errors(self, loss_function):
        if self._results:
            dataset = self.shallow_results_dataset()
            reprojection_errors = dataset.compute_reprojection_errors_alt(
                loss_function=loss_function
            )
            reprojection_errors_list = np.array(
                [
                    item
                    for sublist in list(reprojection_errors.values())
                    for item in sublist
                ]
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
        super().__init__(dataset)
        self._results = None
        self._time = None
        self._iterations = None

    @property
    def results(self) -> BundleAdjustmentBenchmarkResults:
        if self._results:
            return self._results
        raise AttributeError

    @property
    def time(self):
        if self._time:
            return self._time
        raise AttributeError

    @property
    def iterations(self):
        if self._iterations:
            return self._iterations
        raise AttributeError

    def shallow_results_trimmed_original_dataset(self):
        # Note: everything (excluding cameras, and 3d_points) points to the original dataset(!!)
        if self._results:
            copied_dataset = copy.copy(
                self.dataset
            )  # Shallow copy(!) is enough to export, but be really careful here

            copied_dataset.datasetEntries = list(
                map(lambda x: copy.copy(x), copy.copy(copied_dataset.datasetEntries))
            )

            prior_point_length = len(copied_dataset.points3D)
            results_point_ids = [
                p.identifier for p in list(self._results.point_mapping.values())
            ]
            copied_dataset.points3D = [
                p for p in copied_dataset.points3D if p.identifier in results_point_ids
            ]
            posterior_point_length = len(self._results.point_mapping.values())

            assert prior_point_length != posterior_point_length

            """for reduced datasets only for testing"""
            copied_dataset.datasetEntries = copied_dataset.datasetEntries[
                                            0: len(self._results.camera_mapping)
                                            ]

            for index, de in enumerate(copied_dataset.datasetEntries):
                if prior_point_length != posterior_point_length:
                    de.points2D = copy.copy(
                        de.points2D
                    )  # Replace by shallow copy, points are still -> orig. dataset
                    for index in range(len(de.points2D)):  # this is slow as fk
                        try:
                            copied_dataset.points3D_mapped[
                                de.points2D[index].point3D_identifier
                            ]
                        except KeyError:
                            # Point got lost due to reduced dataset for debug test
                            mod_point = copy.deepcopy(de.points2D[index])
                            mod_point.point3D_identifier = None
                            de.points2D[
                                index
                            ] = mod_point  # Note: the list is a new object, created by copy.copy(...)
            copied_dataset.refresh_mapping()
            return copied_dataset
        raise AttributeError

    def subprocess_benchmark(
            self,
            benchmark_function_name="benchmark",
            *args,
            **kwargs,
    ):
        """
        Args:
            benchmark_function_name: function name of to-be-called benchmark function (for e.g. in-dev functions)
            *args (object): args passed to benchmark function
            **kwargs (object): kwargs passed to benchmark function
        """

        def execute_subprocess_benchmark(
                dataset, queue: multiprocessing.Queue, function_name: str, *ar, **kw
        ):
            subprocess_benchmark_class = self.__class__(copy.copy(dataset))
            benchmark_function = getattr(subprocess_benchmark_class, function_name)
            benchmark_function(*ar, **kw)
            queue.put(subprocess_benchmark_class.results)
            queue.put(subprocess_benchmark_class.time)
            queue.put(subprocess_benchmark_class.iterations)
            print("Process exiting")
            exit(0)

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=execute_subprocess_benchmark,
            args=args,
            kwargs={
                "queue": q,
                "dataset": self.dataset,
                "function_name": benchmark_function_name,
                **kwargs,
            },
        )
        p.start()
        item_count = 0
        items = []
        while (
                item_count != 3
        ):  # We do this because join does not work when putting large objects in queue.
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
        self._iterations = copy.deepcopy(items[2])

    def _shallow_results_dataset(self):
        """ Function for complete results """
        # Note: everything (excluding cameras, and 3d_points) points to the original dataset(!!)
        # Shallow copy(!) is enough to export, but be really careful here

        if not self._results:
            raise AttributeError

        assert len(self._results.point_mapping) == len(self.dataset.points3D), \
            "number of points3D must be equal"
        assert len(self._results.camera_mapping) == len(self.dataset.datasetEntries), \
            "number of cameras must be equal"

        copied_dataset = copy.copy(self.dataset)
        copied_dataset.datasetEntries = list(
            map(lambda x: copy.copy(x), copy.copy(copied_dataset.datasetEntries))
        )
        copied_dataset.points3D = list(self._results.point_mapping.values())
        copied_dataset.refresh_mapping()

        #  Since only the camera changes we can substitute it with the by the new cameras
        camera_mapping = self._results.camera_mapping
        for index, de in enumerate(copied_dataset.datasetEntries):
            de.camera = camera_mapping.get(index)
            assert de.camera is not None, \
                "something wrong happened in the camera_mapping"

        copied_dataset.refresh_mapping()
        return copied_dataset

    def _shallow_reduced_results_dataset(self, points_limit, camera_limit, only_trimmed_2d_points):
        """for reduced datasets mostly only for testing"""

        # Note: everything (excluding cameras, and 3d_points) points to the original dataset(!!)
        # Shallow copy(!) is enough to export, but be really careful here
        def _replace_point_with_p3d_id_none(de_points2d, ind):
            # Point got lost due to reduced dataset for debug test
            mod_point = copy.deepcopy(de_points2d[ind])
            mod_point.point3D_identifier = None
            # Note: the list is a new object, created by copy.copy(...)
            de_points2d[ind] = mod_point

        if not self._results:
            raise AttributeError

        assert len(self._results.camera_mapping) == camera_limit, \
            "not enough cameras found in the results"

        copied_dataset = copy.copy(self.dataset)
        copied_dataset.datasetEntries = list(
            map(lambda x: copy.copy(x), copy.copy(copied_dataset.datasetEntries))
        )

        copied_dataset.points3D = list(self._results.point_mapping.values())
        copied_dataset.refresh_mapping()

        copied_dataset.datasetEntries = copied_dataset.datasetEntries[:camera_limit]

        # Since only the camera changes we can substitute it with the by the new cameras
        camera_mapping = self._results.camera_mapping
        point_2d_ids_per_dataset_entry = self.dataset.get_reduced_dataset_2d_ids_per_camera(
            cameras_limit=camera_limit,
            points_limit=points_limit
        )

        for index, de in enumerate(copied_dataset.datasetEntries):
            de.camera = camera_mapping.get(index)
            assert de.camera is not None, f"camera index: {index} had no camera"

            # Replace list by shallow copy, points are still -> orig. dataset
            de.points2D = copy.copy(de.points2D)

            if only_trimmed_2d_points:
                ids_of_reduced_set_of_points = point_2d_ids_per_dataset_entry[index]
                list_of_ids_to_keep = ids_of_reduced_set_of_points
            else:
                list_of_ids_with_points_3d = [p.identifier for p in de.points2D
                                              if copied_dataset.points3D_mapped.get(p.point3D_identifier)]
                list_of_ids_to_keep = list_of_ids_with_points_3d

            for i in range(len(de.points2D)):
                """ we set the p3d identifier to none if its not in the list_to_keep but has a p3d_identifier """
                if de.points2D[i].identifier not in list_of_ids_to_keep and de.points2D[i].point3D_identifier:
                    _replace_point_with_p3d_id_none(de.points2D, i)

        copied_dataset.refresh_mapping()
        return copied_dataset

    def shallow_results_dataset(self, points_limit=None, camera_limit=None, only_trimmed_2d_points=True):
        if not self._results:
            raise AttributeError

        if not points_limit and not camera_limit:
            return self._shallow_results_dataset()

        assert points_limit and camera_limit, "points_limit and camera_limit have to be set at the same time"
        return self._shallow_reduced_results_dataset(
            points_limit=points_limit,
            camera_limit=camera_limit,
            only_trimmed_2d_points=only_trimmed_2d_points
        )

    def reprojection_errors(self, loss_function, points_limit, camera_limit):
        if self._results:
            dataset = self.shallow_results_dataset(points_limit=points_limit, camera_limit=camera_limit)
            reprojection_errors = dataset.compute_reprojection_errors_alt(
                loss_function=loss_function
            )
            reprojection_errors_list = np.array(
                [
                    item
                    for sublist in list(reprojection_errors.values())
                    for item in sublist
                ]
            )
            return reprojection_errors_list
        raise AttributeError

    def export_results_in_colmap_format(
            self, points_limit=None, cameras_limit=None, output_path=None, open_in_colmap=False
    ):
        # TODO: note: copied from above, can be refactored into benchmark class
        if not output_path:
            output_path = "export_results/" + str(uuid4())
        os.makedirs(output_path, exist_ok=True)
        shallow_results_dataset = self.shallow_results_dataset(points_limit=points_limit, camera_limit=cameras_limit)
        export_in_colmap_format(shallow_results_dataset, output_path, binary=True)
        if open_in_colmap:
            show_in_colmap(
                output_path, shallow_results_dataset.images_path, block=False
            )
