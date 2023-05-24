import numpy as np
import pytest

from common.camera_poses import CameraPose, CoordinateSystem, CONVERSION_MATRIX


def setup_camera_pose(func):
    def inner():
        c = CameraPose.from_formatted_file("test_pose.txt")
        func(c)

    return inner


def _inverse_tests(c: CameraPose):
    assert c.inverse().inverse().rotation.as_matrix() == pytest.approx(c.rotation.as_matrix())
    assert np.dot(c.inverse().rotation.as_matrix(), c.rotation.as_matrix()) == pytest.approx(np.identity(3))
    assert c.inverse().rotation.as_matrix() == pytest.approx(c.rotation.as_matrix().transpose())
    assert c.inverse().translation == pytest.approx(np.dot(-c.rotation.as_matrix().transpose(), c.translation))


@setup_camera_pose
def test_inverse(c: CameraPose):
    _inverse_tests(c)


@setup_camera_pose
def test_inverse_coordinate_system(c: CameraPose):
    c = c.in_coordinate_system(CoordinateSystem.COLMAP)
    _inverse_tests(c)


@setup_camera_pose
def test_in_coordinate_system(c: CameraPose):
    assert c.in_coordinate_system(CoordinateSystem.COLMAP).rotation_matrix == \
           pytest.approx(np.dot(CONVERSION_MATRIX, np.dot(c.rotation_matrix, CONVERSION_MATRIX)))
