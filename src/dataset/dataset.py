import copy
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional
from warnings import warn

from scipy.spatial.transform import Rotation

from src.benchmark.multiprocesser import ListMultiProcessor
from src.dataset import np  # For the seed and reproducibility
from src.dataset.datasetEntry import DatasetEntry
from src.dataset.loss_functions import LossFunction
from src.dataset.point import Point2D, Point3D


@dataclass
class Dataset:
    points3D: List[Point3D]
    points3D_mapped: Dict = field(init=False)
    datasetEntries: List[DatasetEntry]
    name: Optional[str] = None

    def __post_init__(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}

    def refresh_mapping(self):
        self.points3D_mapped = {p.identifier: p for p in self.points3D}

    @staticmethod
    def _random_direction():  # TODO: helper methods
        r = np.random.rand(3)
        r /= np.linalg.norm(r)
        return r

    @staticmethod
    def _random_direction_2d():  # TODO: helper methods
        r = np.random.rand(2)
        r /= np.linalg.norm(r)
        return r

    @staticmethod
    def with_noise(dataset: "Dataset", point3d_noise=3e-2, camera_rotation_noise=5e-2, camera_translation_noise=5e-2,
                   camera_intrinsics_noise=10, point2d_noise=1):
        new_dataset = copy.deepcopy(dataset)  # TODO: this is not performant
        for p in new_dataset.points3D:
            p.translate_np(Dataset._random_direction() * point3d_noise)  # * np.random.randn(3))
        for d in new_dataset.datasetEntries:
            d.camera.camera_pose.apply_move(  # TODO: check which distributions to use for random
                Dataset._random_direction() * camera_translation_noise  # * np.random.randn(3)
            )
            d.camera.camera_pose.apply_transform_3d(
                Rotation.from_rotvec(
                    Dataset._random_direction() * camera_rotation_noise
                ).as_matrix()  # * np.random.randn(3)
            )
            d.camera.camera_intrinsics.apply_noise(np.random.rand(3, 3) * camera_intrinsics_noise)
            for p2 in d.points2D:
                p2.translate_np(Dataset._random_direction_2d() * point2d_noise)
            d.refresh_mapping()
            new_dataset.refresh_mapping()
        return new_dataset

    @staticmethod
    def with_noise_mp(dataset: "Dataset", point3d_noise=3e-2, camera_rotation_noise=5e-2, camera_translation_noise=5e-2,
                      camera_intrinsics_noise=10, point2d_noise=1):
        new_dataset = copy.deepcopy(dataset)  # TODO: this is not performant
        x = copy.deepcopy(new_dataset.points3D[0])

        def _process_p(p, point3d_noise):
            return p.translated_np(Dataset._random_direction() * point3d_noise)

        print(new_dataset.points3D[0])
        l = ListMultiProcessor(new_dataset.points3D, partial(_process_p, point3d_noise=3e-2))

        new_dataset.points3D = l.process()

        def _process_d(d: DatasetEntry):
            d.camera.camera_pose.apply_move(  # TODO: check which distributions to use for random
                Dataset._random_direction() * camera_translation_noise  # * np.random.randn(3)
            )
            d.camera.camera_pose.apply_transform_3d(
                Rotation.from_rotvec(
                    Dataset._random_direction() * camera_rotation_noise
                ).as_matrix()  # * np.random.randn(3)
            )
            d.camera.camera_intrinsics.apply_noise(np.random.rand(3, 3) * camera_intrinsics_noise)
            points2D = []
            for p2 in d.points2D:
                points2D.append(p2.translated_np(Dataset._random_direction_2d() * point2d_noise))
            return DatasetEntry(image_metadata=d.image_metadata, points2D=points2D, camera=copy.deepcopy(d.camera))

        l = ListMultiProcessor(dataset.datasetEntries, _process_d)
        new_dataset.datasetEntries = l.process()

        new_dataset.refresh_mapping()
        return new_dataset

    @property
    def images_path(self):
        if len(self.datasetEntries) > 0 and self.datasetEntries[0].image_metadata.image_path:
            return str(Path(self.datasetEntries[0].image_metadata.image_path).parent)
        return ""  # TODO: or raise error?

    def compute_reprojection_errors(self):  # TODO: Deprecated
        warn("compute_reprojection_errors is deprecated")
        reprojection_errors = {}
        for index, de in enumerate(self.datasetEntries):
            p2d, p3d = de.map2d_3d(self.points3D_mapped, zipped=False, np=True)
            reprojection_errors.update({
                index: de.camera.compute_projection_errors(p2d=p2d, p3d=p3d)
            })
        return reprojection_errors

    def compute_reprojection_errors_alt(self, loss_function: LossFunction):
        reprojection_errors = {}
        for index, de in enumerate(self.datasetEntries):
            p2d, p3d = de.map2d_3d(self.points3D_mapped, zipped=False, np=True)
            reprojection_errors.update({
                index: de.camera.compute_projection_errors_alt(p2d=p2d, p3d=p3d, loss_function=loss_function)
            })
        return reprojection_errors

    # def compute_reprojection_errors_threaded(self):
    #     reprojection_errors = {}
    #
    #     def _process_dataset_entry(index_dataset_entry, points3D_mapped):
    #         index, dataset_entry = index_dataset_entry
    #         p2d, p3d = dataset_entry.map2d_3d(points3D_mapped, zipped=False, np=True)
    #         return {
    #             index: dataset_entry.camera.compute_projection_errors(p2d=p2d, p3d=p3d)
    #         }
    #
    #     lmp = ListMultiProcessor(
    #         list(enumerate(self.datasetEntries)),
    #         partial(_process_dataset_entry, points3D_mapped=self.points3D_mapped)
    #     )
    #     reprojection_dict_list = lmp.process()
    #     for d in reprojection_dict_list:
    #         reprojection_errors.update({**d})
    #
    #     return reprojection_errors

    #  @property
    def num_3d_points(self):
        return len(self.points3D)

    #  @property
    def num_images(self):
        return len(self.datasetEntries)

    #  @property
    def avg_num_3d_points_per_image(self):  # TODO: avg or median(?)
        return np.average([de.num_3d_points for de in self.datasetEntries])

    #  @property
    def avg_num_2d_points_per_image(self):
        return np.average([de.num_2d_points for de in self.datasetEntries])

    def __deepcopy__(self, memodict):
        return Dataset(
            points3D=list(map(lambda p: Point3D(p.identifier, p.x, p.y, p.z, p.metadata), self.points3D)),
            datasetEntries=list(map(lambda de: DatasetEntry(
                image_metadata=de.image_metadata,
                points2D=list(
                    map(lambda p: Point2D(p.identifier, p.x, p.y, p.point3D_identifier, p.metadata), de.points2D)),
                camera=copy.deepcopy(de.camera)
            ), self.datasetEntries))
        )

    def make_reduced_dataset(self, camera_limit, points_limit):
        """ WARNING: THIS RETURNS A SHALLOW COPY. MOST OBJECT REFERENCE STUFF IN THE ORIGINAL DATASET """
        shallow_dataset = copy.copy(self)
        shallow_dataset.datasetEntries = [copy.copy(de) for de in self.datasetEntries[: camera_limit]]
        points_3d_by_camera_limit_times_cameras_sorted = sorted(
            list(Counter([p3d for d_entry in shallow_dataset.datasetEntries for p2d, p3d in
                          d_entry.map2d_3d(self.points3D_mapped)[: points_limit]]).items())
            , key=lambda x: x[1]
        )
        shallow_dataset.points3D = [x[0] for x in points_3d_by_camera_limit_times_cameras_sorted if
                                    x[1] >= 2]
        point3d_ids = [p.identifier for p in shallow_dataset.points3D]

        shallow_dataset.refresh_mapping()

        for de in shallow_dataset.datasetEntries:
            point_ids_we_want = [p.identifier for p in de.points_with_3d()[: points_limit]
                                 if p.point3D_identifier in point3d_ids]

            # Shallow copy of list
            de.points2D = copy.copy(de.points2D)
            for i in range(len(de.points2D)):
                if de.points2D[i].identifier not in point_ids_we_want:
                    point_copy = copy.deepcopy(de.points2D[i])
                    point_copy.point3D_identifier = None
                    # Note: the list is a new object, created by copy.copy(...)
                    de.points2D[i] = point_copy
            de.refresh_mapping()

        return shallow_dataset

    def get_reduced_dataset_2d_ids_per_camera(self, cameras_limit, points_limit, as_list=False):
        points_3d_by_camera_limit_times_cameras_sorted = sorted(
            list(Counter([p3d for d_entry in self.datasetEntries[:cameras_limit] for p2d, p3d in
                          d_entry.map2d_3d(self.points3D_mapped)[: points_limit]]).items())
            , key=lambda x: x[1]
        )
        point3d_ids = [x[0].identifier for x in points_3d_by_camera_limit_times_cameras_sorted
                       if x[1] >= 2]
        res = {
            index: [p.identifier for p in de.points_with_3d()[: points_limit]
                    if p.point3D_identifier in point3d_ids]
            for index, de in enumerate(self.datasetEntries[:cameras_limit])
        }
        if as_list:
            return list(res.values())
        return res
