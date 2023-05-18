from __future__ import print_function  # TODO: REMOVE

import os
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg
from scipy.spatial.transform.rotation import Rotation

from dataset.datacontainers.camera_pose.constants import CONVERSION_MATRIX
from dataset.datacontainers.camera_pose.enums_and_types import TransformationDirection, CoordinateSystem, PoseFormat
from dataset.datacontainers.camera_pose.exceptions import InvalidInputFormatErr
from dataset.datacontainers.camera_pose.helpers import _print_2d_matrix_formatted, parse_metadata, _opposite, \
    create_metadata


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
        r1, r2, r3 = new_t_matrix[0:3, 1], new_t_matrix[0:3, 1], new_t_matrix[0:3, 2]
        r1, r2, r3 = r1 / linalg.norm(r1), r2 / linalg.norm(r2), r3 / linalg.norm(r3)
        #  self.rotation = Rotation.from_matrix(new_t_matrix[0:3, 0:3] / linalg.norm(new_t_matrix[0:3, 0:3][:, 0]))
        self.rotation = Rotation.from_matrix(np.vstack([r1 / linalg.norm(r1),
                                                        r2 / linalg.norm(r2),
                                                        r3 / linalg.norm(r3)]))
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
        rotation_diff_matrix = np.linalg.inv(cpo.rotation_matrix).dot(cpt.rotation_matrix)

        #  r1 = rotation_diff_matrix[0:3, 0]  TODO: Normalization probably not necessary
        #  r2 = rotation_diff_matrix[0:3, 1]
        #  r3 = rotation_diff_matrix[0:3, 2]
        #  r1, r2, r3 = r1 / linalg.norm(r1), r2 / linalg.norm(r2), r3 / linalg.norm(r3)
        #  rotation_diff_matrix = np.vstack([r1 / linalg.norm(r1), r2 / linalg.norm(r2), r3 / linalg.norm(r3)])

        return np.linalg.norm(rotation_diff_matrix - np.identity(3))
        #  rotation_diff = Rotation.from_matrix(np.linalg.inv(cpo.rotation_matrix).dot(cpt.rotation_matrix))
        #  return np.linalg.norm(rotation_diff.as_rotvec())  # TODO: this was changed
