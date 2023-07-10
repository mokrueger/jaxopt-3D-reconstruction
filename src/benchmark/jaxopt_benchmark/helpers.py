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
            cx=float(params[8]),
            cy=float(params[9]),
            s=float(params[10]),
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
