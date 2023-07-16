import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from triangulation_relaxations.se3 import Se3
from triangulation_relaxations.so3 import rotvec_to_r

from src.dataset.camera import Camera
from src.dataset.camera_pose.camera_pose import CameraPose
from src.dataset.camera_pose.enums_and_types import (
    CoordinateSystem,
    TransformationDirection,
)
from src.dataset.loaders.colmap_dataset_loader.loader import params_to_intrinsics
from src.reconstruction.bundle_adjustment.utils import get_reprojection_residuals_cpu


def _parse_output_params(param_list, dataset):
    # VERY BIG NOTE: If we don't use np.array(...) or float(...) we reference memory stored on the GPU
    # This will be "revived" if we get the values in the queue.get(), filling up the complete GPU memory again
    cameras = {}
    for index, params in enumerate(param_list):
        if any(np.isnan(params)):  # TODO: adjust this later
            raise Exception(
                "NANANANANANANANANANANANANANANANANANANANANANANANA BATMAN (nan detected)"
            )
        old_camera = dataset.datasetEntries[index].camera
        new_camera_pose = CameraPose(
            rotation=Rotation.from_rotvec(np.array(params[0:3])),
            translation=np.array(params[3:6]),
            coordinate_system=CoordinateSystem.COLMAP,
            identifier=old_camera.camera_pose.identifier,
            direction=TransformationDirection.W2C,
        )
        new_intrinsics = params_to_intrinsics(
            fx=float(params[6]),
            fy=float(params[7]),
            cx=float(params[8]) if len(params) == 11 else 0.0,  # TODO: Fix this
            cy=float(params[9]) if len(params) == 11 else 0.0,
            s=float(params[10]) if len(params) == 11 else 0.0,
        )
        cameras.update(
            {
                index: Camera(
                    camera_pose=new_camera_pose,
                    camera_intrinsics=new_intrinsics,
                    width=old_camera.width,
                    height=old_camera.height,
                )
            }
        )
    return cameras


def _parse_output_params_bundle(
    param_list,
    dataset,
    cx_cy_skew,
    num_3d_points,
    num_cams,
    benchmark_index_to_point_identifier_mapping,
):
    # VERY BIG NOTE: If we don't use np.array(...) or float(...) we reference memory stored on the GPU
    # This will be "revived" if we get the values in the queue.get(), filling up the complete GPU memory again

    # Note: opt_params = jnp.concatenate([cam_params, intr_params, point_params])
    num_cam_params = 6
    num_intr_params = 2
    poses = param_list[: num_cams * num_cam_params].reshape((num_cams, num_cam_params))
    intrinsics = param_list[
        num_cams * num_cam_params : num_cams * num_cam_params
        + num_cams * num_intr_params
    ].reshape((num_cams, num_intr_params))

    """parse cameras"""
    cameras = {}
    for index in range(num_cams):
        pose = poses[index]
        intr = intrinsics[index]
        intr_2 = cx_cy_skew[index]
        if any(np.isnan(np.concatenate([pose, intr]))):  # TODO: adjust this later
            raise Exception(
                "NANANANANANANANANANANANANANANANANANANANANANANANA BATMAN (nan detected)"
            )

        old_camera = dataset.datasetEntries[index].camera
        new_camera_pose = CameraPose(
            rotation=Rotation.from_rotvec(np.array(pose[0:3])),
            translation=np.array(pose[3:6]),
            coordinate_system=CoordinateSystem.COLMAP,
            identifier=old_camera.camera_pose.identifier,
            direction=TransformationDirection.W2C,
        )
        new_intrinsics = params_to_intrinsics(
            fx=float(intr[0]),
            fy=float(intr[1]),
            cx=float(intr_2[0]),
            cy=float(intr_2[1]),
            s=float(intr_2[2]),
        )
        cameras.update(
            {
                index: Camera(
                    camera_pose=new_camera_pose,
                    camera_intrinsics=new_intrinsics,
                    width=old_camera.width,
                    height=old_camera.height,
                )
            }
        )
    """ parse points """
    new_points = param_list[
        num_cams * num_cam_params + num_cams * num_intr_params :
    ].reshape((num_3d_points, 3))
    point_mapping = {}
    for index, point in enumerate(new_points):
        identifier = benchmark_index_to_point_identifier_mapping.get(index)
        copied_point = copy.deepcopy(dataset.points3D_mapped.get(identifier))

        # get new data

        # Note we need this mapping and cannot use point[0], point[1], ...; because it revives GPU memory
        copied_point.x, copied_point.y, copied_point.z = list(
            map(float, np.array(point))
        )
        copied_point.metadata.update({"note": "returned from bundle adjustment"})
        point_mapping.update({identifier: copied_point})
    return cameras, point_mapping


def plot_costs(
    ax,
    pose0,
    pose1,
    points,
    observations,
    intrinsics,
    eps=0.1,
    n=1000,
    label0="",
    label1="",
):
    """Plot cost function when interpolating between pose0 and pose1"""
    taus = np.linspace(-eps, 1 + eps, n)
    index_0, index_1 = np.searchsorted(taus, [0, 1])
    taus = np.insert(taus, [index_0, index_1], [0, 1])
    index_1 += 1  # compensate for the insertion of 0

    p0 = Se3(pose0[:3, :3], pose0[:3, 3])
    p1 = Se3(rotvec_to_r(pose1[:3]), pose1[3:])

    objective_values = []
    for tau in taus:
        p_int = Se3(
            (p0.q ** (1 - tau) * p1.q**tau).R,
            p0.t * (1 - tau) + p1.t * tau,
        )

        objective_values.append(
            get_reprojection_residuals_cpu(
                p_int.T, points, observations, intrinsics
            ).sum()
        )

    ax.plot(taus, objective_values)
    ax.plot(0, objective_values[index_0], "o", color="red", label=label0)
    ax.plot(1, objective_values[index_1], "o", color="blue", label=label1)


def create_plot(camera_pose0, camera_pose1, points3D, points2D, intrinsics):
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 1)
    plot_costs(
        ax,
        camera_pose0,
        camera_pose1,
        points3D,
        points2D,
        intrinsics,
        label0="initial pose",
        label1="optimized pose",
        n=100,
    )
    # ax.axhline(results_gt.cost, color='k', linestyle='--')
    ax.set_xlabel("distance (normalized)")
    ax.set_ylabel("cost function")
    ax.legend()
    fig.savefig("test_jaxopt.png")
    fig.show()
