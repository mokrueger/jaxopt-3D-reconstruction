import os
import random
import sqlite3
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
from scipy import linalg

from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import (CoordinateSystem,
                                                     PoseFormat,
                                                     TransformationDirection)
from src.dataset.camera_pose.exceptions import (InvalidInputFormatErr,
                                                NonMatchingIdentifiersErr,
                                                NotEnoughCameraPosesErr)


def _raise_invalid_input_err_on_exception(func: Callable[..., "CameraPose"]):
    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise InvalidInputFormatErr from e

    return inner_function


def _create_transformation_translation_matrix(rotation_matrix, translation_vector):
    return np.vstack([np.hstack([rotation_matrix, translation_vector.reshape((3, 1))]), np.array([0, 0, 0, 1])])


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

        camera_pose_folder = os.path.join(project_folder,
                                          "camera_poses")  # str(ImageFolders.camera_poses_folder.value))
        camera_pose_files = [os.path.join(camera_pose_folder, x) for x in os.listdir(camera_pose_folder)]

        os.makedirs(sparse_folder_path, exist_ok=True)
        Path(os.path.join(sparse_folder_path, "points3D.txt")).touch(exist_ok=True)

        camera_intrinsics_folder = os.path.join(project_folder,  # str(ImageFolders.camera_intrinsics_folder.value))
                                                "camera_intrinsics")
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
    t (numpy 4x4 array): A rigid tranfsormation represented by a homogenous
        transformation.
    points (list(numpy array)): List of 3D points.
    error_bound (float): The noise added to each coordinate is in
        (-error_bound, error_bound).

    Returns:
        A list of numpy 3D arrays, each entry represents a 3D point.
    """

    return [R.dot(np.array(p)) + t +
            np.random.uniform(-error_bound, error_bound, 3) for p in points]
