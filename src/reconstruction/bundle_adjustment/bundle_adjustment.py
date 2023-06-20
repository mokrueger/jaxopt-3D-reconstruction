import jax.numpy as jnp
import numpy as np
import optax
from jax import device_put, jit, vmap
from jax.experimental import sparse
from jax.tree_util import register_pytree_node_class
from jaxopt import OptaxSolver
from triangulation_relaxations import so3
from jaxopt import LevenbergMarquardt


@jit
def rot_mat_from_vec(rodrigues_vec):
    theta = jnp.linalg.norm(rodrigues_vec)
    r = rodrigues_vec / theta
    I = jnp.eye(3, dtype=float)
    r_rT = jnp.outer(r, r)
    r_cross = jnp.cross(jnp.eye(3), r)
    return jnp.cos(theta) * I + (1 - jnp.cos(theta)) * r_rT + jnp.sin(theta) * r_cross


@jit
@vmap
def parse_intrinsics(intr_vec):
    return jnp.array(
        [
            [intr_vec[0], intr_vec[4], intr_vec[2]],
            [0, intr_vec[1], intr_vec[3]],
            [0, 0, 1],
        ]
    )


@jit
@vmap
def parse_cam_poses(cam_vec):
    return jnp.concatenate(
        [rot_mat_from_vec(cam_vec[:3]), cam_vec[3:6, jnp.newaxis]], axis=1
    )


@register_pytree_node_class
class BundleAdjustment:
    def __init__(self, cam_num, batch_size: int = 8):
        self.batch_size = batch_size
        self.cam_num = cam_num

    def tree_flatten(self):
        children = ()
        aux_data = {"batch_size": self.batch_size, "cam_num": self.cam_num}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jit
    def get_residuals(self, opt_params, p3d_ind, p2d_list, cam_ind):
        # (n, 3, 4), (n, 3, 3), (n, m, 4), (n, m, 2)
        # poses, intrinsics, points, observations = opt_params
        cam_end_index = self.cam_num * 6
        intr_end_index = cam_end_index + self.cam_num * 5

        cam_params = opt_params[:cam_end_index].reshape((-1, 6))
        intr_params = opt_params[cam_end_index:intr_end_index].reshape((-1, 5))
        points = opt_params[intr_end_index:].reshape((-1, 4))

        poses = parse_cam_poses(cam_params)
        intrinsics = parse_intrinsics(intr_params)

        p3d_selected = points.take(p3d_ind, axis=0)
        intr_selected = intrinsics.take(cam_ind, axis=0)
        poses_selected = poses.take(cam_ind, axis=0)

        KE = jnp.einsum("bij,bjk->bik", intr_selected, poses_selected)
        x = jnp.einsum("bij,bj->bi", KE, p3d_selected)  # reprojected_points
        x = x[..., :2] / x[..., 2:3]  # 2:3 to prevent axis from being removed

        return ((p2d_list - x) ** 2).sum(axis=1)


class JaxBundleAdjustment:
    def __init__(self, cam_num, learning_rate=0.01):
        # set params
        self.cam_num = cam_num
        self.ba = BundleAdjustment(self.cam_num)
        self.learning_rate = learning_rate

        # create optimizer
        self.optimizer, self.solver = self.create_optimizer()

    def create_optimizer(self):
        # opt = OptaxSolver(
        #     self.ba.get_residuals,
        #     opt=optax.adamw(self.learning_rate),
        #     tol=1e-15,
        #     jit=True,
        #     maxiter=1000,
        # )

        opt = LevenbergMarquardt(
            residual_fun=self.ba.get_residuals,
            tol=1e-15,
            gtol=1e-15,
            jit=True,
            solver="inv",
        )

        return opt, jit(opt.run)

    def optimize(self, poses0, intrinsics0, p3d_list, p3d_ind, p2d_list, cam_ind):
        cam_params = jnp.array(
            [JaxBundleAdjustment.pose_mat_to_vec(p0) for p0 in poses0]
        ).flatten()
        intr_params = jnp.array(intrinsics0).flatten()
        point_params = jnp.array(p3d_list).flatten()

        opt_params = jnp.concatenate([cam_params, intr_params, point_params])

        params, state = self.solver(opt_params, p3d_ind, p2d_list, cam_ind)
        params = params.block_until_ready()

        return params, state

    def compile(self, points_num, batch_size=8):
        # 6 for pose, 5 for intrinsics
        opt_params = jnp.zeros((batch_size, 11))
        _points_gpu = jnp.zeros((batch_size, points_num, 4))
        _observations_gpu = jnp.zeros((batch_size, points_num, 2))
        _mask_gpu = jnp.zeros((batch_size, points_num), dtype=bool)

        self.solver(
            opt_params, _points_gpu, _observations_gpu, _mask_gpu
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
            return np.array([device_put(i) for i in data])
        return device_put(data)
