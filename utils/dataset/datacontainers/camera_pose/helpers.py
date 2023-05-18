import os
from enum import Enum

from dataset.datacontainers.camera_pose import TransformationDirection, InvalidInputFormatErr, PoseFormat, CoordinateSystem


def _2d_map(func, iterable):
    return list(map(lambda x: list(map(func, x)), iterable))


def _print_2d_matrix_formatted(matrix):
    rows_mapped_to_str = _2d_map(str, matrix)
    return f"{os.linesep.join(list(map(' '.join, rows_mapped_to_str)))}"


def _opposite(direction):
    if direction == TransformationDirection.C2W:
        return TransformationDirection.W2C
    return TransformationDirection.C2W


def parse_metadata(metadata: str):
    def find_in_split(E, s):
        for string in s:
            if e := E.get_enum_by_name(string):
                return e
        return None

    split = metadata.split()
    if len(split) != 3:
        raise InvalidInputFormatErr("Invalid Metadata")
    pose_format = find_in_split(PoseFormat, split)
    coordinate_system = find_in_split(CoordinateSystem, split)
    transformation_direction = find_in_split(TransformationDirection, split)
    if not (pose_format and coordinate_system and transformation_direction):
        raise InvalidInputFormatErr("Invalid Metadata Format")
    return pose_format, coordinate_system, transformation_direction


def create_metadata(pose_format: Enum, coordinate_system: Enum, transformation_direction: Enum):
    return " ".join([pose_format.name, coordinate_system.name, transformation_direction.name])
