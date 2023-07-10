import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt

from .loss import JaxLossFunction, cauchy_loss, l2_loss
from .utils import parse_cam_pose, pose_mat_to_vec

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class PoseOptimization:
    def __init__(
        self,
        avg_cam_width_sqr,
        loss_fn: JaxLossFunction = JaxLossFunction.L2,
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
    def get_residuals(self, opt_params, points, observations):
        pose = parse_cam_pose(opt_params)

        intrinsics = jnp.diag(jnp.array([opt_params[6], opt_params[7], 1]))

        # reproject
        KE = jnp.einsum("ij,jk->ik", intrinsics, pose)
        p2d_projected = jnp.einsum("ij,bj->bi", KE[:, :3], points)
        p2d_projected = p2d_projected + KE[:, 3]
        p2d_projected = p2d_projected[..., :2] / p2d_projected[..., 2:3]

        if self.loss_fn == JaxLossFunction.CAUCHY:
            res = cauchy_loss(observations, p2d_projected)
        elif self.loss_fn == JaxLossFunction.L2:
            res = l2_loss(observations, p2d_projected)
        else:
            res = l2_loss(observations, p2d_projected)
        return res.sum(axis=1) / self.avg_cam_width_sqr


class JaxPoseOptimizer:
    def __init__(self, avg_cam_width, loss_fn: JaxLossFunction = JaxLossFunction.L2):
        self.po = PoseOptimization(avg_cam_width**2, loss_fn=loss_fn)
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

        return lm, jax.jit(lm.run)

    def optimize(self, poses0, intrinsics0, points_gpu, observations_gpu):
        opt_params = jnp.concatenate([pose_mat_to_vec(poses0), jnp.array(intrinsics0)])

        params, state = self.solver(opt_params, points_gpu, observations_gpu)
        params = params.block_until_ready()

        return params, state

    def compile(self, points_num):
        # 6 for pose, 2 for intrinsics
        params, _ = self.solver(
            jnp.zeros(8), jnp.zeros((points_num, 3)), jnp.zeros((points_num, 2))
        )
        params.block_until_ready()
