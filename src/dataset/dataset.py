import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from functools import partial
from typing import List, Dict, Optional

from src.benchmark.multiprocesser import ListMultiProcessor
from src.dataset import np  # For the seed and reproducibility
from scipy.spatial.transform import Rotation

from src.dataset.datasetEntry import DatasetEntry
from src.dataset.point import Point3D, Point2D


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

    def compute_reprojection_errors(self):
        reprojection_errors = {}
        for index, de in enumerate(self.datasetEntries):
            p2d, p3d = de.map2d_3d_np(self.points3D_mapped, zipped=False)
            reprojection_errors.update({
                index: de.camera.compute_projection_errors(p2d=p2d, p3d=p3d)
            })
        return reprojection_errors

    # def compute_reprojection_errors_threaded(self):
    #     reprojection_errors = {}
    #
    #     def _process_dataset_entry(index_dataset_entry, points3D_mapped):
    #         index, dataset_entry = index_dataset_entry
    #         p2d, p3d = dataset_entry.map2d_3d_np(points3D_mapped, zipped=False)
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
