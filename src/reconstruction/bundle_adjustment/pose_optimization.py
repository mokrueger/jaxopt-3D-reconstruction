import jax.numpy as jnp
import numpy as np

from jax import device_put, jit
from jaxopt import LevenbergMarquardt

# from triangulation_relaxations import Se3, so3


# def pose_to_x(pose):
#     return np.concatenate([so3.r_to_rotvec(pose.R), pose.t])


def get_reprojection_residuals_cpu(pose, points, observations, intrinsics):
    # _pose = Se3(so3.rotvec_to_r(pose[:3]), pose[3:])

    KE = np.einsum("ij,jk->i k", intrinsics, pose)
    x = np.einsum("ij,hj->hi", KE, points)  # reprojected_points
    x = x[..., :2] / x[..., 2:3]  # 2:3 to prevent axis from being removed

    return ((observations - x) ** 2).sum(axis=1)


@jit
def rot_mat_from_vec(rodrigues_vec):
    theta = jnp.linalg.norm(rodrigues_vec)
    r = rodrigues_vec / theta
    I = jnp.eye(3, dtype=float)
    r_rT = jnp.outer(r, r)
    r_cross = jnp.cross(jnp.eye(3), r)
    return jnp.cos(theta) * I + (1 - jnp.cos(theta)) * r_rT + jnp.sin(theta) * r_cross


@jit
def get_reprojection_residuals(pose, points, observations, intrinsics):
    _pose = jnp.block(
        [[rot_mat_from_vec(pose[:3]), pose[3:, jnp.newaxis]], [jnp.zeros(3).T, 1]]
    )  # build pose matrix (SE3) from pose vector

    # reproject
    KE = jnp.einsum("ij,jk->i k", intrinsics, _pose)
    x = jnp.einsum("ij,hj->hi", KE, points)  # reprojected_points
    x = x[..., :2] / x[..., 2:3]  # 2:3 to prevent axis from being removed

    return ((observations - x) ** 2).sum(axis=1)


class JaxPoseOptimizer:
    def __init__(self):
        # set params

        # create optimizer
        self.optimizer, self.optimizer_func = self.create_lm_optimizer()

    def create_lm_optimizer(self):
        lm = LevenbergMarquardt(
            residual_fun=get_reprojection_residuals,
            tol=1e-15,
            gtol=1e-15,
            jit=True,
            solver="inv",
        )

        return lm, jit(lm.run)

    def run_pose_opt(self, pose0, points_gpu, observations_gpu, intrinsics_gpu):
        params, state = self.optimizer_func(
            JaxPoseOptimizer.pose_mat_to_vec(pose0),
            points=points_gpu,
            observations=observations_gpu,
            intrinsics=intrinsics_gpu,
        )

        params = params.block_until_ready()

        return params, state

    def compile_pose_opt(self, point_shape, observations_shape, intrinsics_shape):
        _pose0 = jnp.zeros(6)
        _points_gpu = jnp.zeros(point_shape)
        _observations_gpu = jnp.zeros(observations_shape)
        _intrinsics_gpu = jnp.zeros(intrinsics_shape)
        self.optimizer_func(
            _pose0,
            points=_points_gpu,
            observations=_observations_gpu,
            intrinsics=_intrinsics_gpu,
        ).params.block_until_ready()

    @staticmethod
    def pose_mat_to_vec(pose):
        R = pose[:3, :3]
        n = np.zeros((*R.shape[:-2], 3))
        angle = np.arccos(
            np.clip((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2, -1, 1)
        )
        n[..., 0] = R[..., 2, 1] - R[..., 1, 2]
        n[..., 1] = R[..., 0, 2] - R[..., 2, 0]
        n[..., 2] = R[..., 1, 0] - R[..., 0, 1]
        norms = np.linalg.norm(n, axis=-1)
        nonzero_indices = norms != 0
        n[nonzero_indices] *= (angle[nonzero_indices] / norms[nonzero_indices])[
            ..., None
        ]

        return jnp.concatenate([n, pose[:3, 3]])

    @staticmethod
    def to_gpu(data):
        if isinstance(data, (list, tuple)):
            return [device_put(i) for i in data]
        return device_put(data)
