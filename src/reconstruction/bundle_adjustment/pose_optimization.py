import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LevenbergMarquardt

from .loss import LossFunction, cauchy_loss, l2_loss
from .utils import parse_cam_pose, pose_mat_to_vec

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class PoseOptimization:
    def __init__(
        self,
        avg_cam_width_sqr,
        loss_fn: LossFunction = LossFunction.L2,
    ):
        self.avg_cam_width_sqr = avg_cam_width_sqr
        self.loss_fn = loss_fn

    def tree_flatten(self):
        children = ()
        aux_data = {
            "avg_cam_width_sqr": self.avg_cam_width_sqr,
            "loss_fn": self.loss_fn,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @jax.jit
    def get_residuals(self, opt_params, points, observations, mask):
        pose = parse_cam_pose(opt_params)

        intrinsics = jnp.diag(jnp.array([opt_params[6], opt_params[7], 1]))

        # reproject
        KE = jnp.einsum("ij,jk->ik", intrinsics, pose)
        p2d_projected = jnp.einsum("ij,bj->bi", KE[:, :3], points)
        p2d_projected = p2d_projected + KE[:, 3]
        p2d_projected = p2d_projected[..., :2] / p2d_projected[..., 2:3]

        if self.loss_fn == LossFunction.CAUCHY:
            res = cauchy_loss(observations, p2d_projected)
        elif self.loss_fn == LossFunction.L2:
            res = l2_loss(observations, p2d_projected)
        else:
            res = l2_loss(observations, p2d_projected)
        return res.sum(axis=1) * mask / self.avg_cam_width_sqr


class JaxPoseOptimizer:
    def __init__(self, avg_cam_width, loss_fn: LossFunction = LossFunction.L2):
        self.po = PoseOptimization(avg_cam_width**2, loss_fn=loss_fn)
        # create optimizer
        self.optimizer, self.solver = self.create_lm_optimizer()

    def create_lm_optimizer(self):
        lm = LevenbergMarquardt(
            residual_fun=self.po.get_residuals,
            tol=1e-8,
            gtol=1e-8,
            jit=True,
            solver="cholesky",
            maxiter=100,
        )

        return lm, jax.jit(jax.vmap(lm.run, in_axes=(0, 0, 0, 0)))

    def optimize(self, poses0, intrinsics0, points_gpu, observations_gpu, mask):
        opt_params = jnp.array(
            [
                jnp.concatenate([pose_mat_to_vec(p0), jnp.array(i0)])
                for p0, i0 in zip(poses0, intrinsics0)
            ]
        )

        params, state = self.solver(opt_params, points_gpu, observations_gpu, mask)
        params = params.block_until_ready()

        return params, state

    def compile(self, points_num, batch_size=8):
        # 6 for pose, 5 for intrinsics
        opt_params = jnp.zeros((batch_size, 8))
        _points_gpu = jnp.zeros((batch_size, points_num, 3))
        _observations_gpu = jnp.zeros((batch_size, points_num, 2))
        _mask_gpu = jnp.zeros((batch_size, points_num), dtype=float)

        self.solver(
            opt_params, _points_gpu, _observations_gpu, _mask_gpu
        ).params.block_until_ready()
