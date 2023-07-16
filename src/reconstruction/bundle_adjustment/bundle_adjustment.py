import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxopt import LevenbergMarquardt

from src.reconstruction.bundle_adjustment.utils import (
    parse_cam_pose_vmap,
    parse_intrinsics_vmap,
    pose_mat_to_vec,
)

from .loss import l2_loss

jax.config.update("jax_enable_x64", True)


@jax.jit
def reproject_point(KE, point_2d, p3d_index, points_3d, mask):
    point_2d_projected = KE[:, :3] @ points_3d[p3d_index] + KE[:, 3]
    point_2d_projected = point_2d_projected[:2] / point_2d_projected[2:3]
    return (l2_loss(point_2d_projected, point_2d) * mask).sum()


reproject_points = jax.jit(
    jax.vmap(
        jax.vmap(reproject_point, in_axes=(None, 0, 0, None, 0)),
        in_axes=(0, 0, 0, None, 0),
    )
)


@register_pytree_node_class
class BundleAdjustment:
    def __init__(self, cam_num, avg_cam_width_sqr):
        self.cam_num = cam_num
        self.avg_cam_width_sqr = avg_cam_width_sqr
        self.cam_end_index = self.cam_num * 6
        self.intr_end_index = self.cam_end_index + self.cam_num * 2

    def tree_flatten(self):
        children = ()
        aux_data = {
            "cam_num": self.cam_num,
            "avg_cam_width_sqr": self.avg_cam_width_sqr,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.jit
    def get_residuals(
        self,
        opt_params,
        points_2d_all,
        p3d_indices_all,
        cx_cy_skew,
        masks_all,
    ):
        poses = parse_cam_pose_vmap(opt_params[: self.cam_end_index].reshape((-1, 6)))
        points_3d = opt_params[self.intr_end_index :].reshape((-1, 3))
        intrinsics = parse_intrinsics_vmap(
            opt_params[self.cam_end_index : self.intr_end_index].reshape((-1, 2)),
            cx_cy_skew,
        )

        KE = jnp.einsum("bij,bjk->bik", intrinsics, poses)

        error = reproject_points(
            KE, points_2d_all, p3d_indices_all, points_3d, masks_all
        )

        return error.sum(axis=1) / self.avg_cam_width_sqr


class JaxBundleAdjustment:
    def __init__(self, cam_num, avg_cam_width):
        self.cam_num = cam_num
        self.ba = BundleAdjustment(self.cam_num, avg_cam_width)
        self.optimizer, self.solver = self.create_lm_optimizer()

    def create_lm_optimizer(self):
        opt = LevenbergMarquardt(
            residual_fun=self.ba.get_residuals,
            tol=1e-6,
            jit=True,
            maxiter=100,
        )

        return opt, jax.jit(opt.run)

    def prepare_params(self, poses0, intrinsics0, points0):
        fx_fy = intrinsics0[..., :2]
        cx_cy_skew = intrinsics0[..., 2:]

        cam_params = jnp.array([pose_mat_to_vec(p0) for p0 in poses0]).flatten()
        fx_fy_params = jnp.array(fx_fy).flatten()
        point_params = jnp.array(points0).flatten()

        opt_params = jnp.concatenate([cam_params, fx_fy_params, point_params])
        return opt_params, cx_cy_skew

    def optimize(
        self,
        opt_params,
        points_2d_all,
        p3d_indices_all,
        cx_cy_skew,
        masks_all,
    ):
        params, state = self.solver(
            opt_params,
            points_2d_all,
            p3d_indices_all,
            cx_cy_skew,
            masks_all,
        )
        params = params.block_until_ready()
        return params, state

    def compile(self, points_num, indices_num):
        self.optimize(
            jnp.zeros(self.cam_num * 8 + points_num * 3),
            jnp.zeros((self.cam_num, indices_num, 2)),
            jnp.zeros((self.cam_num, indices_num), dtype=int),
            jnp.zeros((self.cam_num, 3)),
            jnp.zeros((self.cam_num, indices_num)),
        )
