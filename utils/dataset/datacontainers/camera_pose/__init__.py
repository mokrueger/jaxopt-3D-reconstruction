from __future__ import print_function  # TODO: REMOVE

import os
import sqlite3
import random
from pathlib import Path
from typing import Callable, List, Union
from functools import partial

import numpy as np
from scipy import linalg
from scipy.spatial.transform.rotation import Rotation

from dataset.datacontainers.camera_pose.constants import CONVERSION_MATRIX
from dataset.datacontainers.camera_pose.enums_and_types import TransformationDirection, CoordinateSystem, PoseFormat
from dataset.datacontainers.camera_pose.exceptions import InvalidInputFormatErr, NonMatchingIdentifiersErr, NotEnoughCameraPosesErr
from dataset.datacontainers.camera_pose.helpers import _print_2d_matrix_formatted, parse_metadata, _opposite, create_metadata


def _raise_invalid_input_err_on_exception(func: Callable[..., "CameraPose"]):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise InvalidInputFormatErr from e

    return inner_function


def _create_transformation_translation_matrix(rotation_matrix, translation_vector):
    return np.vstack([np.hstack([rotation_matrix, translation_vector.reshape((3, 1))]), np.array([0, 0, 0, 1])])


class CameraPose:
    def __init__(self, rotation: Rotation, translation: np.ndarray, identifier: str = None,
                 coordinate_system=CoordinateSystem.UNITY,
                 direction=TransformationDirection.C2W):
        self.rotation: Rotation = rotation
        self.translation: np.ndarray = translation
        self.identifier: str = identifier
        self.coordinate_system: CoordinateSystem = coordinate_system
        self.direction: TransformationDirection = direction

    def __repr__(self):
        return f"Rotation: \n{str(self.rotation.as_matrix().round(decimals=2))} \n" \
               f"Translation {str(self.translation.round(decimals=2))}"

    def in_coordinate_system(self, target_system):
        if self.coordinate_system == target_system:
            return CameraPose(rotation=self.rotation,
                              translation=self.translation,
                              identifier=self.identifier,
                              coordinate_system=self.coordinate_system,
                              direction=self.direction)
        translation = np.dot(CONVERSION_MATRIX, self.translation)
        rotation = Rotation.from_matrix(np.dot(CONVERSION_MATRIX, np.dot(self.rotation.as_matrix(), CONVERSION_MATRIX)))
        return CameraPose(rotation=rotation,
                          translation=translation,
                          identifier=self.identifier,
                          coordinate_system=target_system,
                          direction=self.direction)

    def inverse(self) -> "CameraPose":
        inverse_rotation = self.rotation.inv()
        translation = np.dot(-inverse_rotation.as_matrix(), self.translation)
        return CameraPose(rotation=inverse_rotation,
                          translation=translation,
                          identifier=self.identifier,
                          coordinate_system=self.coordinate_system,
                          direction=_opposite(self.direction))

    def in_direction(self, target_direction):
        cp = CameraPose(rotation=self.rotation,
                        translation=self.translation,
                        identifier=self.identifier,
                        coordinate_system=self.coordinate_system,
                        direction=self.direction)
        if self.direction == target_direction:
            return cp
        return cp.inverse()

    def apply_transform_4d(self, matrix_4d):
        new_t_matrix = matrix_4d.dot(self.transformation_translation_matrix)
        self.rotation = Rotation.from_matrix(new_t_matrix[0:3, 0:3] / linalg.norm(new_t_matrix[0:3, 0:3][:, 0]))
        self.translation = new_t_matrix[0:3, 3]

    def apply_transform_3d(self, matrix_3d):
        self.rotation = Rotation.from_matrix(matrix_3d.dot(self.rotation_matrix))
        self.translation = matrix_3d.dot(self.translation)

    def apply_translation(self, translation_vector: np.ndarray):
        self.translation += translation_vector

    # TODO: Check validity
    def apply_move(self, translation_vector: np.ndarray):
        if self.direction == TransformationDirection.C2W:
            self.apply_translation(translation_vector=translation_vector)
        iv = self.inverse()
        self.apply_translation(translation_vector=translation_vector)
        self.translation = iv.inverse().translation

    @property
    def position(self):
        if self.direction == TransformationDirection.C2W:
            return self.translation
        return self.inverse().translation

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.rotation.as_matrix()

    @property
    def rotation_translation_matrix(self) -> np.ndarray:
        return np.c_[self.rotation.as_matrix(), self.translation]

    @property
    def transformation_translation_matrix(self) -> np.ndarray:
        return np.r_[self.rotation_translation_matrix, np.array([[0.0, 0.0, 0.0, 1.0]])]

    def as_wxyz_quaternion_translation_str(self):
        quaternion = list(map(str, self.rotation.as_quat()))
        translation = list(map(str, self.translation))
        return f"{' '.join([quaternion[3], *quaternion[0:3], *translation])}"

    def as_rotation_translation_str(self):
        return _print_2d_matrix_formatted(self.rotation_translation_matrix)

    def as_transformation_translation_str(self):
        return _print_2d_matrix_formatted(self.transformation_translation_matrix)

    """
    FROM FILE METHODS
    """

    @classmethod
    def from_formatted_file(cls, filepath) -> "CameraPose":
        with open(filepath) as file:
            metadata = file.readline().strip(os.linesep)
            content = file.read()
            pose_format, coordinate_system, direction = parse_metadata(metadata)
            return CameraPose.from_string(string=content, identifier=Path(filepath).stem,
                                          pose_format=pose_format,
                                          coordinate_system=coordinate_system,
                                          direction=direction)

    @classmethod
    def from_file(cls, filepath,
                  pose_format=PoseFormat.QT,
                  coordinate_system=CoordinateSystem.UNITY,
                  direction=TransformationDirection.C2W) -> "CameraPose":
        with open(filepath) as file:
            content = file.read()
            return CameraPose.from_string(string=content, identifier=Path(filepath).stem,
                                          pose_format=pose_format,
                                          coordinate_system=coordinate_system,
                                          direction=direction)

    """
    FROM STRING METHODS
    """

    @classmethod
    def from_formatted_string(cls, string, identifier=None) -> "CameraPose":
        split = string.splitlines()
        metadata = split[0]
        rest_data = os.linesep.join(split[1:])
        pose_format, coordinate_system, direction = parse_metadata(metadata)
        return CameraPose.from_string(string=rest_data, identifier=identifier,
                                      pose_format=pose_format,
                                      coordinate_system=coordinate_system,
                                      direction=direction)

    @classmethod
    def from_string(cls, string, identifier=None,
                    pose_format=PoseFormat.QT,
                    coordinate_system=CoordinateSystem.UNITY,
                    direction=TransformationDirection.C2W) -> "CameraPose":
        if pose_format == PoseFormat.QT:
            return CameraPose.from_string_wxyz_quaternion_translation(string=string, identifier=identifier,
                                                                      coordinate_system=coordinate_system,
                                                                      direction=direction)
        if pose_format == PoseFormat.RT:
            return CameraPose.from_string_rotation_translation(string=string, identifier=identifier,
                                                               coordinate_system=coordinate_system, direction=direction)
        if pose_format == PoseFormat.T:
            return CameraPose.from_string_transformation_translation(string=string, identifier=identifier,
                                                                     coordinate_system=coordinate_system,
                                                                     direction=direction)

    @classmethod
    @_raise_invalid_input_err_on_exception
    def from_string_wxyz_quaternion_translation(cls, string, identifier=None, coordinate_system=CoordinateSystem.UNITY,
                                                direction=TransformationDirection.C2W) -> "CameraPose":
        content = string.split()
        if len(content) != 7:
            raise InvalidInputFormatErr()
        content = list(map(float, content))
        #  Note: Scipy quaternion format (xyzw); Our format (wxyz)
        return cls(identifier=identifier,
                   rotation=Rotation.from_quat([*content[1:4], content[0]]),
                   translation=np.array(content[4:7]),
                   coordinate_system=coordinate_system,
                   direction=direction)

    @classmethod
    @_raise_invalid_input_err_on_exception
    def from_string_rotation_translation(cls, string, identifier=None, coordinate_system=CoordinateSystem.UNITY,
                                         direction=TransformationDirection.C2W) -> "CameraPose":
        content = list(map(str.split, string.splitlines()))
        if len(content) != 3 or len(content[0]) != 4 or len(content[1]) != 4 or len(content[2]) != 4:
            raise InvalidInputFormatErr()
        content = [list(map(float, row)) for row in content]
        return cls(identifier=identifier,
                   rotation=Rotation.from_matrix([
                       content[0][0:3],
                       content[1][0:3],
                       content[2][0:3]
                   ]),
                   translation=np.array([
                       content[0][3],
                       content[1][3],
                       content[2][3],
                   ]),
                   coordinate_system=coordinate_system,
                   direction=direction)

    @classmethod
    @_raise_invalid_input_err_on_exception
    def from_string_transformation_translation(cls, string, identifier=None, coordinate_system=CoordinateSystem.UNITY,
                                               direction=TransformationDirection.C2W) -> "CameraPose":
        content = list(map(str.split, string.splitlines()))
        if len(content) != 4 or len(content[0]) != 4 or len(content[1]) != 4 or len(content[2]) != 4:
            raise InvalidInputFormatErr()
        content = [list(map(float, row)) for row in content]
        return cls(identifier=identifier,
                   rotation=Rotation.from_matrix([
                       content[0][0:3],
                       content[1][0:3],
                       content[2][0:3]
                   ]),
                   translation=np.array([
                       content[0][3],
                       content[1][3],
                       content[2][3],
                   ]),
                   coordinate_system=coordinate_system,
                   direction=direction)

    """
    TO FILE METHODS
    """

    def to_file(self, filepath, pose_format: PoseFormat = PoseFormat.QT, include_metadata=True):
        with open(filepath, "w+") as file:
            if include_metadata:
                file.write(create_metadata(pose_format=pose_format,
                                           coordinate_system=self.coordinate_system,
                                           transformation_direction=self.direction) + os.linesep)
                if pose_format == PoseFormat.QT:
                    file.write(self.as_wxyz_quaternion_translation_str())
                if pose_format == PoseFormat.RT:
                    file.write(self.as_rotation_translation_str())
                if pose_format == PoseFormat.T:
                    file.write(self.as_transformation_translation_str())

    """
    COMPARING METHODS
    """

    @staticmethod
    def compute_position_error(camera_pose_one: "CameraPose", camera_pose_two: "CameraPose"):
        return np.linalg.norm(camera_pose_one.position - camera_pose_two.position)

    @staticmethod
    def compute_rotation_error(camera_pose_one: "CameraPose", camera_pose_two: "CameraPose"):
        cpo = camera_pose_one.in_direction(target_direction=TransformationDirection.C2W)
        cpt = camera_pose_two.in_direction(target_direction=TransformationDirection.C2W)
        rotation_diff = Rotation.from_matrix(np.linalg.inv(cpo.rotation_matrix).dot(cpt.rotation_matrix))
        return np.linalg.norm(rotation_diff.as_rotvec())


# TEST
# c1 = CameraPose.from_file("test.txt")
# c2 = CameraPose.from_file("test2.txt", pose_format=PoseFormat.RT)
# c3 = CameraPose.from_file("test3.txt", pose_format=PoseFormat.T)
#
# print(c1.rotation_translation_matrix)
# print(c1.transformation_translation_matrix)
#
# c1.to_file("test4.txt")
# c2.to_file("test5.txt", pose_format=PoseFormat.RT)
# c3.to_file("test6.txt", pose_format=PoseFormat.T)


class CameraSet:
    def __init__(self, camera_poses: List[CameraPose] = None, pose_format=PoseFormat.QT):
        unified_coordinates = [c.in_coordinate_system(target_system=CoordinateSystem.COLMAP) for c in camera_poses]
        unified_direction = list(map(lambda x: x.inverse() if x.direction == TransformationDirection.W2C else x,
                                     unified_coordinates))
        self.camera_poses = unified_direction

    def sort_by_identifier(self):
        self.camera_poses = sorted(self.camera_poses, key=lambda x: x.identifier)

    def find_by_identifier(self, identifier):
        for c in self.camera_poses:
            if c.identifier == identifier:
                return c

    # @staticmethod
    # def get_common_poses(camera_set_one: "CameraSet", camera_set_two: "CameraSet"):
    #     cpo = CameraSet(camera_poses=camera_set_one.camera_poses)
    #     cpt = CameraSet(camera_poses=camera_set_two.camera_poses)
    #     common = [c for c in cpo.camera_poses if cpt.find_by_identifier(c.identifier)]
    #     return CameraSet(camera_poses=common)

    def get_common_poses(self, camera_set_two: "CameraSet"):
        cpo = CameraSet(camera_poses=self.camera_poses)
        cpt = CameraSet(camera_poses=camera_set_two.camera_poses)
        common = [c for c in cpo.camera_poses if cpt.find_by_identifier(c.identifier)]
        return CameraSet(camera_poses=common)

    def create_sparse_model(self, sparse_folder_path, project_folder, database_path):

        # NOTE THESE TWO FUNCTIONS ARE DUPLICATES WITH COLMAP_WORKER/HELPERS.PY
        def get_image_entries_from_database(database_path):
            return sqlite3.connect(database_path).execute("SELECT * from images").fetchall()

        def get_mapping_identifier_to_id_filename(db_table):
            mapping = {}
            for row in db_table:
                mapping.update({os.path.splitext(row[1])[0]: (row[0], row[1])})
            return mapping

        db_entries = get_image_entries_from_database(database_path)
        db_image_names = [os.path.splitext(x[1])[0] for x in db_entries]
        identifier_to_filename_id_mapping = get_mapping_identifier_to_id_filename(db_table=db_entries)

        camera_pose_folder = os.path.join(project_folder, str(ImageFolders.camera_poses_folder.value))
        camera_pose_files = [os.path.join(camera_pose_folder, x) for x in os.listdir(camera_pose_folder)]

        os.makedirs(sparse_folder_path, exist_ok=True)
        Path(os.path.join(sparse_folder_path, "points3D.txt")).touch(exist_ok=True)

        camera_intrinsics_folder = os.path.join(project_folder, str(ImageFolders.camera_intrinsics_folder.value))
        camera_intrinsics_files = [os.path.join(camera_intrinsics_folder, x) for x in
                                   os.listdir(camera_intrinsics_folder)]

        # Filter ones that have not been processed by colmap
        camera_intrinsics_files = [c for c in camera_intrinsics_files if Path(c).stem in db_image_names]

        identifier_to_cam_id_mapping = {}

        params_to_cam_id_mapping = {}
        with open(os.path.join(sparse_folder_path, "cameras.txt"), "w+") as cameras_txt:
            for index, ci_file in enumerate(camera_intrinsics_files, start=1):
                with open(ci_file, "r") as open_ci_file:
                    ci_content = open_ci_file.read().split()
                    width, height, principal_x, principal_y, focal_x, focal_y = \
                        (ci_content[4], ci_content[5], ci_content[0], ci_content[1], ci_content[2], ci_content[3])
                    # ci_content format px py fx fy
                    camera_content = f"{width} {height} {focal_x} {focal_y} {principal_x} {principal_y}"

                    if cam_id := params_to_cam_id_mapping.get(camera_content):
                        identifier_to_cam_id_mapping.update({Path(ci_file).stem: cam_id})
                    else:
                        camera = f"{index} PINHOLE {camera_content}"
                        cameras_txt.write(f"{camera}\n")
                        params_to_cam_id_mapping.update({camera_content: index})
                        identifier_to_cam_id_mapping.update({Path(ci_file).stem: index})

        with open(os.path.join(sparse_folder_path, "images.txt"), "w+") as images_txt:
            images_txt_content = []
            for c in self.camera_poses:
                # Filter ones that have not been processed by colmap # TODO: Check if still required
                if c.identifier not in db_image_names:
                    continue

                ic = c.inverse() if c.direction != TransformationDirection.W2C else c
                icc = ic.in_coordinate_system(CoordinateSystem.COLMAP)
                id_filename = identifier_to_filename_id_mapping.get(icc.identifier)
                images_txt_content.append(f"{id_filename[0]} {icc.as_wxyz_quaternion_translation_str()} "
                                          f"{identifier_to_cam_id_mapping.get(icc.identifier)} {id_filename[1]}\n")
            images_txt.write("\n".join(images_txt_content) + "\n")

    @staticmethod
    def from_sparse_folder(folder_path):
        camera_poses = []
        with open(os.path.join(folder_path, "images.txt")) as images:
            images_content = images.read()
            linewise = images_content.split("\n")
            ignore_hastags = list(filter(lambda x: not x.startswith("#"), linewise))
            ignore_hastags = ignore_hastags[0:-1]  # Remove the empty string at the end
            only_camera_info = ignore_hastags[0::2]
            for line in only_camera_info:
                linesplit = line.split()
                camera_poses.append(
                    CameraPose.from_string_wxyz_quaternion_translation(string=" ".join(linesplit[1:8]),
                                                                       identifier=os.path.splitext(linesplit[9])[0],
                                                                       coordinate_system=CoordinateSystem.COLMAP,
                                                                       direction=TransformationDirection.W2C))
        return CameraSet(camera_poses=camera_poses)

    @staticmethod
    def from_folder(folder_path):
        camera_poses = []
        for file_path in [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                          os.path.isfile(os.path.join(folder_path, f))]:
            camera_poses.append(CameraPose.from_formatted_file(file_path))
        return CameraSet(camera_poses=camera_poses)

    def apply_transform_4d(self, matrix_4d):
        for c in self.camera_poses:
            c.apply_transform_4d(matrix_4d=matrix_4d)

    def apply_transform_3d(self, matrix_3d):
        for c in self.camera_poses:
            c.apply_transform_3d(matrix_3d=matrix_3d)

    def apply_translation(self, translation_vector):
        for c in self.camera_poses:
            c.apply_translation(translation_vector=translation_vector)

    def apply_move(self, translation_vector):
        for c in self.camera_poses:
            c.apply_move(translation_vector=translation_vector)

    @staticmethod
    def compute_pose_quaternion(camera_set_one: "CameraSet", camera_set_two: "CameraSet", with_scale=True,
                                match_identifiers=False):
        """
        Notes:
            This is an implementation of the quaternion based solution described
            in "Closed-form solution of absolute orientation using unit
            quaternions", B.K.P. Horn, Journal of the Optical Society of America,
            Vol. 4(4), pp 629--642, 1987.
        """

        # return a matrix representation of rotation from a unit quaternion representation
        # the quaternion is assumed to have unit norm and q = [s,qx,qy,qz]
        def _quaternion_to_matrix(q):
            return np.array(
                [[1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3],
                  2 * q[1] * q[3] + 2 * q[0] * q[2]],
                 [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                  2 * q[2] * q[3] - 2 * q[0] * q[1]],
                 [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1],
                  1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]])

        points_in_one = [c.position for c in camera_set_one.camera_poses]
        points_in_two = [c.position for c in camera_set_two.camera_poses]

        if match_identifiers:
            sorted_one = sorted(camera_set_one.camera_poses, key=lambda x: x.identifier)
            sorted_two = sorted(camera_set_two.camera_poses, key=lambda x: x.identifier)
            for o, t in list(zip(sorted_one, sorted_two)):
                if o.identifier != t.identifier:
                    raise NonMatchingIdentifiersErr()
            points_in_one = [c.position for c in sorted_one]
            points_in_two = [c.position for c in sorted_two]

        num_points = len(points_in_one)
        dim_points = len(points_in_one[0])
        # cursory check that the number of points is sufficient
        if num_points < dim_points:
            raise NotEnoughCameraPosesErr('Number of points must be greater/equal {0}.'.format(dim_points))

            # construct matrices out of the two point sets for easy manipulation
        one_mat = np.array(points_in_one).T
        two_mat = np.array(points_in_two).T

        one_mean = one_mat.mean(1)
        two_mean = two_mat.mean(1)

        m = one_mat.dot(two_mat.T) - \
            np.outer(one_mean, two_mean) * num_points

        one_mean_stack = np.outer(one_mean, np.ones(len(points_in_one)))
        two_mean_stack = np.outer(two_mean, np.ones(len(points_in_two)))
        one_mat_meaned = one_mat - one_mean_stack
        two_mat_meaned = two_mat - two_mean_stack
        one_squared_length_sum = np.sum(np.apply_along_axis(lambda x: np.square(np.linalg.norm(x)), 0, one_mat_meaned))
        two_squared_length_sum = np.sum(np.apply_along_axis(lambda x: np.square(np.linalg.norm(x)), 0, two_mat_meaned))
        scale = np.sqrt(two_squared_length_sum / one_squared_length_sum)

        delta = np.array([[m[1, 2] - m[2, 1]], [m[2, 0] - m[0, 2]], [m[0, 1] - m[1, 0]]])
        n = np.vstack((np.array([np.trace(m), delta[0][0], delta[1][0], delta[2][0]]),
                       np.hstack((delta, m + m.T - np.trace(m) * np.eye(3)))))
        (eigenValues, eigenVectors) = linalg.eig(n)
        # quaternion we want is the eigenvector corresponding to the
        # maximal eigenvalue (these are real valued as N is a real, symmetric
        # matrix)
        r = _quaternion_to_matrix(eigenVectors[:, eigenValues.argmax()])
        if with_scale:
            r *= scale
        t = two_mean - r.dot(one_mean)
        return r, t

    @staticmethod
    def compute_pose_matrix(camera_set_one: "CameraSet", camera_set_two: "CameraSet", with_scale=True,
                            match_identifiers=False):
        '''
        Absolute orientation using a matrix to represent the rotation. Solution is due to
        S. Umeyama, "Least-Squares Estimation of Transformation Parameters
        Between Two Point Patterns", IEEE Trans. Pattern Anal. Machine Intell., vol. 13(4): 376-380.

        This is a refinement of the method proposed by Arun, Huang and Blostein, ensuring that the
        rotation matrix is indeed a rotation and not a reflection.
        '''
        points_in_one = [c.position for c in camera_set_one.camera_poses]
        points_in_two = [c.position for c in camera_set_two.camera_poses]

        if match_identifiers:
            sorted_one = sorted(camera_set_one.camera_poses, key=lambda x: x.identifier)
            sorted_two = sorted(camera_set_two.camera_poses, key=lambda x: x.identifier)
            for o, t in list(zip(sorted_one, sorted_two)):
                if o.identifier != t.identifier:
                    raise NonMatchingIdentifiersErr()
            points_in_one = [c.position for c in sorted_one]
            points_in_two = [c.position for c in sorted_two]

        num_points = len(points_in_one)
        dim_points = len(points_in_one[0])
        # cursory check that the number of points is sufficient
        if num_points < dim_points:
            raise ValueError('Number of points must be greater/equal {0}.'.format(dim_points))

            # construct matrices out of the two point sets for easy manipulation
        one_mat = np.array(points_in_one).T
        two_mat = np.array(points_in_two).T

        # center both data sets on the mean
        one_mean = one_mat.mean(1)
        two_mean = two_mat.mean(1)
        one_m = one_mat - np.tile(one_mean, (num_points, 1)).T
        two_m = two_mat - np.tile(two_mean, (num_points, 1)).T

        # for scale
        one_mean_stack = np.outer(one_mean, np.ones(len(points_in_one)))
        two_mean_stack = np.outer(two_mean, np.ones(len(points_in_two)))
        one_mat_meaned = one_mat - one_mean_stack
        two_mat_meaned = two_mat - two_mean_stack
        one_squared_length_sum = np.sum(np.apply_along_axis(lambda x: np.square(np.linalg.norm(x)), 0, one_mat_meaned))
        two_squared_length_sum = np.sum(np.apply_along_axis(lambda x: np.square(np.linalg.norm(x)), 0, two_mat_meaned))
        scale = np.sqrt(two_squared_length_sum / one_squared_length_sum)

        m = one_m.dot(two_m.T)
        u, s, vt = linalg.svd(m)
        v = vt.T
        # V * diag(1,1,det(U*V)) * U' - diagonal matrix ensures that we have a rotation and not a reflection
        r = v.dot(np.diag((1, 1, linalg.det(u.dot(v))))).dot(u.T)
        if with_scale:
            r *= scale
        t = two_mean - r.dot(one_mean)
        return r, t

    @staticmethod
    def compute_robust_pose(camera_set_one: "CameraSet", camera_set_two: "CameraSet", with_scale=True,
                            match_identifiers=False):
        random.seed(123456789)

        def get_random_subset(camera_set: "CameraSet", num=6):
            return random.sample(camera_set.camera_poses, num)

        #  Note: cs_one is usually colmap poses and cs_two is ar poses
        camera_set_one_common = camera_set_one.get_common_poses(camera_set_two)
        camera_set_two_common = camera_set_two.get_common_poses(camera_set_one)
        camera_set_one_common.sort_by_identifier()
        camera_set_two_common.sort_by_identifier()

        r, t = None, None
        min_positional = float("inf")
        min_rotational = float("inf")
        for i in range(300):
            print(i)
            """Calc and apply pose for cso -> subset space"""
            cso_copy = CameraSet(camera_poses=list(camera_set_one_common.camera_poses))
            curr_subset = CameraSet(camera_poses=get_random_subset(camera_set_two_common))

            cso_common = cso_copy.get_common_poses(curr_subset)
            temp_r, temp_t = CameraSet.compute_pose_quaternion(camera_set_one=cso_common,
                                                               camera_set_two=curr_subset,
                                                               match_identifiers=True)
            transformation_translation_matrix = _create_transformation_translation_matrix(rotation_matrix=temp_r,
                                                                                          translation_vector=temp_t)
            cso_copy.apply_transform_4d(matrix_4d=transformation_translation_matrix)

            """Calc errors"""
            errors = CameraSet.compute_position_rotation_errors(cso_copy, camera_set_two_common,
                                                                already_transformed=True)
            position_errors = np.array([x["position_error"] for x in errors.values()])
            rotation_errors = np.array([x["rotation_error"] for x in errors.values()])
            median_p = np.median(position_errors)
            median_r = np.median(rotation_errors)
            if median_p < min_positional and median_r < min_rotational:
                min_positional = median_p
                min_rotational = median_r
                r = temp_r
                t = temp_t
        return r, t

    @staticmethod
    def _compute_position_rotation_errors(camera_set_one: "CameraSet", camera_set_two: "CameraSet",
                                          already_transformed=False):
        """
        Helper function to compute errors between two Sets (only the common elements)
        """
        camera_set_one_common = camera_set_one.get_common_poses(camera_set_two)
        camera_set_two_common = camera_set_two.get_common_poses(camera_set_one)

        if not already_transformed:
            r, t = CameraSet.compute_pose_quaternion(camera_set_one=camera_set_one_common,
                                                     camera_set_two=camera_set_two_common,
                                                     match_identifiers=True)
            transformation_translation_matrix = _create_transformation_translation_matrix(rotation_matrix=r,
                                                                                          translation_vector=t)
            camera_set_one_common.apply_transform_4d(matrix_4d=transformation_translation_matrix)

        mapped = list(map(lambda x: (x, camera_set_two.find_by_identifier(x.identifier)),
                          camera_set_one_common.camera_poses))
        error_dict = {}
        for one, two in mapped:
            if one and two:
                pd = CameraPose.compute_position_error(one, two)
                rd = CameraPose.compute_rotation_error(one, two)
            else:
                pd, rd = None, None
            error_dict.update({one.identifier: {
                "position_error": pd,
                "rotation_error": rd,
            }})
        return error_dict

    @staticmethod
    def compute_position_rotation_errors(camera_set_one: "CameraSet",
                                         camera_set_two: Union["CameraSet", List["CameraSet"]],
                                         already_transformed=False):
        """
        Compute errors based on single Set or a list of Sets (only the common elements)
        Note: SKIPS sequences with less than 3 cameras(!)
        """
        if type(camera_set_two) == CameraSet:
            return CameraSet._compute_position_rotation_errors(camera_set_one=camera_set_one,
                                                               camera_set_two=camera_set_two,
                                                               already_transformed=already_transformed)
        error_dict = {}
        for cst in camera_set_two:
            try:
                ed = CameraSet._compute_position_rotation_errors(camera_set_one=camera_set_one,
                                                                 camera_set_two=cst,
                                                                 already_transformed=already_transformed)
            except NotEnoughCameraPosesErr as e:
                print(e)
                ed = {}
            error_dict = {**error_dict, **ed}
        return error_dict


def create_fixed_axis_frameD(theta_x, theta_y, theta_z, dx, dy, dz):
    """
    Create a fixed axis frame from the given rotation angles and translation.
    Rotation angles are given in degrees.

    Args:
        theta_x (float): Rotation around x axis.
        theta_y (float): Rotation around y axis.
        theta_z (float): Rotation around z axis.
        dx (float): Translation along x axis.
        dy (float): Translation along y axis.
        dz (float): Translation along z axis.

    Returns:
        R,t (numpy.ndarray, numpy.array): Rigid transformation represented as
        a rotation matrix and translation.
    """
    return create_fixed_axis_frame(np.radians(theta_x), np.radians(theta_y),
                                   np.radians(theta_z), dx, dy, dz)


def create_fixed_axis_frame(theta_x, theta_y, theta_z, dx, dy, dz):
    """
    Create a fixed axis frame from the given rotation angles and translation.
    Rotation angles are given in radians.

    Args:
        theta_x (float): Rotation around x axis.
        theta_y (float): Rotation around y axis.
        theta_z (float): Rotation around z axis.
        dx (float): Translation along x axis.
        dy (float): Translation along y axis.
        dz (float): Translation along z axis.

    Returns:
        R,t (numpy.ndarray, numpy.array): Rigid transformation represented as
        a rotation matrix and translation.
    """
    s1 = np.sin(theta_z)
    s2 = np.sin(theta_y)
    s3 = np.sin(theta_x)
    c1 = np.cos(theta_z)
    c2 = np.cos(theta_y)
    c3 = np.cos(theta_x)
    return np.array([[c1 * c2, c1 * s2 * s3 - s1 * c3, c1 * s2 * c3 + s1 * s3],
                     [s1 * c2, s1 * s2 * s3 + c1 * c3, s1 * s2 * c3 - c1 * s3],
                     [-s2, c2 * s3, c2 * c3]]), \
           np.array([dx, dy, dz])


def transform_points_adding_noise(R, t, points, error_bound):
    """
    Apply the given transformation to the points and add noise to each coordinate.

    Args:
    R (numpy 3x3 array): A rotation matrix (we do not check for orthonormality
                         so it is up to the caller to ensure that this is a
                         rotation).
    t (numpy 4x4 array): A rigid transformation represented by a homogenous
        transformation.
    points (list(numpy array)): List of 3D points.
    error_bound (float): The noise added to each coordinate is in
        (-error_bound, error_bound).

    Returns:
        A list of numpy 3D arrays, each entry represents a 3D point.
    """

    return [R.dot(np.array(p)) + t +
            np.random.uniform(-error_bound, error_bound, 3) for p in points]


if __name__ == "__main__":
    # cs = CameraSet.from_folder("/home/morkru/Desktop/Baum/pose")
    # cs2 = CameraSet.from_folder("/home/morkru/Desktop/Baum/pose")
    r, t = create_fixed_axis_frameD(theta_x=0, theta_y=0, theta_z=90,
                                    dx=0.0, dy=0.0, dz=0)

    r *= 100
    transformation_translation_matrix = np.vstack([np.hstack([r, t.reshape((3, 1))]), np.array([0, 0, 0, 1])])
    # cs.apply_transform_4d(matrix4d=transformation_translation_matrix)

    print("known")
    print(r)
    print(t)

    print("calc")
    # import random
    # random.shuffle(cs2.camera_poses)
    # r_c, t_c = cs.compute_pose_quaternion(cs2, cs)
    # print(r_c)
    # print(t_c)

    # noise_range = 3.0
    # for c in cs.camera_poses:
    #    c.apply_move(translation_vector=np.random.uniform(-noise_range, noise_range, 3))
    #
    # print("calc2")
    # r_c, t_c = cs.compute_pose_matrix(cs2, cs)
    # print(r_c)
    # print(t_c)
    colmap_p = CameraSet.from_sparse_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-0fa19810-7a0f-48fd-b6ea-e298b69b5716/T-6a6466d8-47c9-482e-81ff-e5361bfa6b0e/sparse/0/00")
    ar_p = CameraSet.from_sparse_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-0fa19810-7a0f-48fd-b6ea-e298b69b5716/T-6a6466d8-47c9-482e-81ff-e5361bfa6b0e/sparse/0/ar")

    errors = CameraSet.compute_position_rotation_errors(colmap_p, ar_p, already_transformed=False)
    position_errors = np.array([x["position_error"] for x in errors.values()])
    rotation_errors = np.array([x["rotation_error"] for x in errors.values()])

    r, t = CameraSet.compute_robust_pose(camera_set_one=colmap_p, camera_set_two=ar_p)
    tt = _create_transformation_translation_matrix(rotation_matrix=r,
                                                   translation_vector=t)
    colmap_p_copy = CameraSet(colmap_p.camera_poses)
    colmap_p_copy.apply_transform_4d(matrix_4d=tt)
    errors = CameraSet.compute_position_rotation_errors(colmap_p_copy, ar_p, already_transformed=True)
    position_errors_robust = np.array([x["position_error"] for x in errors.values()])
    rotation_errors_robust = np.array([x["rotation_error"] for x in errors.values()])

    # eval stuff
    import matplotlib.pyplot as plt

    plt.xlim((0, 0.07))
    plt.ylim((0, 30))
    plt.hist(position_errors, bins=100)
    plt.show()

    plt.xlim((0, 0.07))
    plt.ylim((0, 30))
    plt.hist(position_errors_robust, bins=100)
    plt.show()

    plt.hist(rotation_errors_robust * (180 / np.pi), bins=100)
    plt.show()


    """ COMPARISON """
    colmap_p2 = CameraSet.from_sparse_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-f0975b0b-b121-495f-936c-bd7918e7e6bc/T-f4e5092f-ee43-46e3-b7f3-82d2383d7e13/sparse/0/00")
    ar_p2 = CameraSet.from_sparse_folder( # Same as above
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-0fa19810-7a0f-48fd-b6ea-e298b69b5716/T-6a6466d8-47c9-482e-81ff-e5361bfa6b0e/sparse/0/ar")

    errors2 = CameraSet.compute_position_rotation_errors(colmap_p2, ar_p2, already_transformed=False)
    position_errors2 = np.array([x["position_error"] for x in errors2.values()])
    rotation_errors2 = np.array([x["rotation_error"] for x in errors2.values()])

    r2, t2 = CameraSet.compute_robust_pose(camera_set_one=colmap_p2, camera_set_two=ar_p2)
    tt2 = _create_transformation_translation_matrix(rotation_matrix=r2,
                                                   translation_vector=t2)
    colmap_p_copy2 = CameraSet(colmap_p2.camera_poses)
    colmap_p_copy2.apply_transform_4d(matrix_4d=tt2)
    errors2 = CameraSet.compute_position_rotation_errors(colmap_p_copy2, ar_p2, already_transformed=True)
    position_errors_robust2 = np.array([x["position_error"] for x in errors2.values()])
    rotation_errors_robust2 = np.array([x["rotation_error"] for x in errors2.values()])

    # eval stuff
    import matplotlib.pyplot as plt

    plt.xlim((0, 0.07))
    plt.ylim((0, 30))
    plt.hist(position_errors2, bins=100)
    plt.show()

    plt.xlim((0, 0.07))
    plt.ylim((0, 30))
    plt.hist(position_errors_robust2, bins=100)
    plt.show()

    plt.hist(rotation_errors_robust2 * (180 / np.pi), bins=100)
    plt.show()


    """"STUFF"""
    ar_poses = CameraSet.from_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-f0975b0b-b121-495f-936c-bd7918e7e6bc/camera_poses")
    colmap_poses = CameraSet.from_sparse_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-572bcd19-f4bf-4f91-95d7-2e9cd8e12608/T-cbe99bf5-8fc6-4c6f-993b-1513aa227864/sparse/1")

    ar_poses.sort_by_identifier()
    colmap_poses.sort_by_identifier()


    #  1-11, 12-13, 14-17, 18-23, 24-31, 32-40 (oder 24-40)
    def stupid(x, min, max):
        return min <= int(x.identifier[11:]) <= max


    #  ar_poses.camera_poses = list(filter(stupid, ar_poses.camera_poses))
    ar_sequence_one = CameraSet(camera_poses=list(filter(partial(stupid, min=1, max=11), ar_poses.camera_poses)))
    ar_sequence_two = CameraSet(camera_poses=list(filter(partial(stupid, min=12, max=13), ar_poses.camera_poses)))
    ar_sequence_three = CameraSet(camera_poses=list(filter(partial(stupid, min=14, max=17), ar_poses.camera_poses)))
    ar_sequence_four = CameraSet(camera_poses=list(filter(partial(stupid, min=18, max=23), ar_poses.camera_poses)))
    ar_sequence_five = CameraSet(camera_poses=list(filter(partial(stupid, min=24, max=31), ar_poses.camera_poses)))
    ar_sequence_six = CameraSet(camera_poses=list(filter(partial(stupid, min=32, max=40), ar_poses.camera_poses)))
    # colmap_poses.camera_poses = list(filter(stupid, colmap_poses.camera_poses))

    """"""
    errors = CameraSet.compute_position_rotation_errors(colmap_poses, [ar_sequence_one,
                                                                       ar_sequence_two,
                                                                       ar_sequence_three,
                                                                       ar_sequence_four,
                                                                       ar_sequence_five,
                                                                       ar_sequence_six])
    position_errors = np.array([x["position_error"] for x in errors.values()])
    rotation_errors = np.array([x["rotation_error"] for x in errors.values()])
    """"""

    # from colmap_service.colmap_worker.helpers import create_sparse_model
    #  rr, tt = CameraSet.compute_pose_quaternion(colmap_poses, ar_poses, with_scale=True, match_identifiers=True)
    colmap_poses_again = colmap_poses = CameraSet.from_sparse_folder(
        "/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-572bcd19-f4bf-4f91-95d7-2e9cd8e12608/T-cbe99bf5-8fc6-4c6f-993b-1513aa227864/sparse/1")
    colmap_poses_again.camera_poses = list(
        filter(lambda x: "32" not in x.identifier and "33" not in x.identifier and "40" not in x.identifier,
               colmap_poses_again.camera_poses))
    #  transformation_translation_matrix = np.vstack([np.hstack([rr, tt.reshape((3, 1))]), np.array([0, 0, 0, 1])])
    #  colmap_poses_again.apply_transform_4d(matrix_4d=transformation_translation_matrix)

    colmap_poses_again.create_sparse_model(
        sparse_folder_path="/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-572bcd19-f4bf-4f91-95d7-2e9cd8e12608/T-cbe99bf5-8fc6-4c6f-993b-1513aa227864/sparse/edit",
        project_folder='/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-572bcd19-f4bf-4f91-95d7-2e9cd8e12608/',
        database_path='/home/morkru/Desktop/Github/ba-ss21-krueger-moritz-photogrammetry-backend/source_code/files/P-572bcd19-f4bf-4f91-95d7-2e9cd8e12608/T-a664f173-e305-424a-b00a-95edac904111/database.db')
