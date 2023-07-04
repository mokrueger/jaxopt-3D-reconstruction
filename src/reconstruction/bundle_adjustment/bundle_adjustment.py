from enum import Enum

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import optax
from jax import device_put, disable_jit, jit, lax, make_jaxpr, pmap, vmap
from jax.experimental import sparse
from jax.profiler import save_device_memory_profile, trace
from jax.tree_util import register_pytree_node_class
from jaxopt import LevenbergMarquardt
from memory_profiler import profile

# jax.config.update("jax_enable_x64", False)


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


@jit
def reproject_point(KE, point_2d, p3d_index, points_3d):
    point_2d_projected = KE[:, :3] @ points_3d[p3d_index] + KE[:, 3]
    point_2d_projected = point_2d_projected[:2] / point_2d_projected[2:3]
    return ((point_2d_projected - point_2d) ** 2).sum()


@jit
def reproject_point_scan(KE, points_2d, p3d_indices, points_3d):
    return lax.scan(
        lambda _, x: (_, reproject_point(KE, *x, points_3d)),
        0,
        (points_2d, p3d_indices),
    )[1]


@jit
def reproject_points(KE, points_2d, p3d_indices, points_3d):
    points_3d_selected = points_3d.take(p3d_indices, axis=0)
    points_2d_projected = jnp.einsum("ij,bj->bi", KE[:, :3], points_3d_selected)
    points_2d_projected = points_2d_projected + KE[:, 3]
    points_2d_projected = points_2d_projected[..., :2] / points_2d_projected[..., 2:3]
    return ((points_2d_projected - points_2d) ** 2).sum(axis=1)


def rp_scan(KE, points_2d_all, p3d_indices_all, points_3d):
    return lax.scan(
        lambda _, x: (_, reproject_points(*x, points_3d).sum()),
        None,
        (KE, points_2d_all, p3d_indices_all),
    )[1]


reproject_point_vmap = jit(vmap(reproject_point, in_axes=(None, 0, 0, None)))
rp_vmap = jit(vmap(reproject_points, in_axes=(0, 0, 0, None)))
rp_vmap_full = jit(vmap(reproject_point_vmap, in_axes=(0, 0, 0, None)))
rp_vmap_scan = jit(vmap(reproject_point_scan, in_axes=(0, 0, 0, None)))


class OptimizationMode(Enum):
    VMAPPED = 1  # mid
    VMAPPED_FULL = 2  # fastest so far
    VMAPPED_LAX_SCAN = 3  # unbelievably slow
    LAX_SCAN = 4  # slow, low mem?


@register_pytree_node_class
class BundleAdjustment:
    def __init__(
        self,
        cam_num,
        batch_size: int = 8,
        opt_mode: OptimizationMode = OptimizationMode.VMAPPED,
    ):
        self.batch_size = batch_size
        self.cam_num = cam_num
        self.opt_mode = opt_mode

    def tree_flatten(self):
        children = ()
        aux_data = {
            "batch_size": self.batch_size,
            "cam_num": self.cam_num,
            "opt_mode": self.opt_mode,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jit
    def get_residuals(self, opt_params, points_2d_all, p3d_indices_all):
        cam_end_index = self.cam_num * 6
        intr_end_index = cam_end_index + self.cam_num * 5

        # parse opt params
        poses = parse_cam_poses(opt_params[:cam_end_index].reshape((-1, 6)))
        intrinsics = parse_intrinsics(
            opt_params[cam_end_index:intr_end_index].reshape((-1, 5))
        )
        points_3d = opt_params[intr_end_index:].reshape((-1, 3))

        KE = jnp.einsum("bij,bjk->bik", intrinsics, poses)

        print(self.cam_num, points_3d.shape, points_2d_all.shape, p3d_indices_all.shape)

        if self.opt_mode == OptimizationMode.VMAPPED:
            error = rp_vmap(KE, points_2d_all, p3d_indices_all, points_3d)
            return error.sum(axis=1)
        if self.opt_mode == OptimizationMode.VMAPPED_FULL:
            error = rp_vmap_full(KE, points_2d_all, p3d_indices_all, points_3d)
            return error.sum(axis=1)
        if self.opt_mode == OptimizationMode.VMAPPED_LAX_SCAN:
            error = rp_vmap_scan(KE, points_2d_all, p3d_indices_all, points_3d)
            return error.sum(axis=1)
        elif self.opt_mode == OptimizationMode.LAX_SCAN:
            return rp_scan(KE, points_2d_all, p3d_indices_all, points_3d)


# @jax.custom_vjp
# def clip_gradient(lo, hi, x):
#     return x  # identity function


# def clip_gradient_fwd(lo, hi, x):
#     return x, (lo, hi)  # save bounds as residuals


# def clip_gradient_bwd(res, g):
#     lo, hi = res
#     return (
#         None,
#         None,
#         jnp.clip(g, lo, hi),
#     )  # use None to indicate zero cotangents for lo and hi


# clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)


class JaxBundleAdjustment:
    def __init__(self, cam_num, learning_rate=0.01):
        # set params
        self.cam_num = cam_num
        self.ba = BundleAdjustment(self.cam_num, opt_mode=OptimizationMode.VMAPPED_FULL)
        self.learning_rate = learning_rate

        # create optimizer
        self.optimizer, self.solver = self.create_optimizer()

    def create_optimizer(self):
        opt = LevenbergMarquardt(
            residual_fun=sparse.sparsify(self.ba.get_residuals),
            tol=1e-15,
            # gtol=1e-15,
            jit=True,
        )

        return opt, jit(opt.run)

    def optimize(
        self,
        poses0,
        intrinsics0,
        points_2d_all,
        points_3d_all,
        p3d_indices_all,
    ):
        cam_params = jnp.array(
            [JaxBundleAdjustment.pose_mat_to_vec(p0) for p0 in poses0]
        ).flatten()
        intr_params = jnp.array(intrinsics0).flatten()
        point_params = jnp.array(points_3d_all).flatten()

        opt_params = jnp.concatenate([cam_params, intr_params, point_params])

        # self.ba.get_residuals(opt_params, points_2d_all, p3d_indices_all)

        # print(
        #     make_jaxpr(self.ba.get_residuals)(
        #         opt_params, points_2d_all, p3d_indices_all
        #     )
        # )

        params, state = self.solver(opt_params, points_2d_all, p3d_indices_all)
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
            return jnp.array([device_put(i) for i in data])
        return device_put(data)
