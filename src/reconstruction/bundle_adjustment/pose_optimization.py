import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt

from .loss import JaxLossFunction
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
    def get_residuals(self, params, points, observations, cx_cy_skew, mask):
        # fmt: off
        pose = parse_cam_pose(params)
        intrinsics = jnp.array( 
            [
                [params[6], cx_cy_skew[2], cx_cy_skew[0]],
                [        0,     params[7], cx_cy_skew[1]],
                [        0,             0,             1],
            ]
        )
        # fmt: on

        KE = jnp.einsum("ij,jk->ik", intrinsics, pose)
        p2d_projected = jnp.einsum("ij,bj->bi", KE[:, :3], points)
        p2d_projected = p2d_projected + KE[:, 3]
        p2d_projected = p2d_projected[..., :2] / p2d_projected[..., 2:3]

        res = self.loss_fn(observations, p2d_projected)
        return res.sum(axis=1) * mask / self.avg_cam_width_sqr


class JaxPoseOptimizer:
    def __init__(self, avg_cam_width, loss_fn: JaxLossFunction = JaxLossFunction.L2):
        self.po = PoseOptimization(avg_cam_width**2, loss_fn=loss_fn)
        self.optimizer, self.solver = self.create_lm_optimizer()

    def create_lm_optimizer(self):
        lm = LevenbergMarquardt(
            residual_fun=self.po.get_residuals,
            tol=1e-6,
            jit=True,
            solver="cholesky",
            maxiter=100,
        )

        return lm, jax.jit(jax.vmap(lm.run, in_axes=(0, 0, 0, 0, 0)))

    def prepare_params(self, poses0, intrinsics0):
        fx_fy = intrinsics0[..., :2]
        cx_cy_skew = intrinsics0[..., 2:]

        opt_params = jnp.array(
            [
                jnp.concatenate([pose_mat_to_vec(p0), jnp.array(i0)])
                for p0, i0 in zip(poses0, fx_fy)
            ]
        )

        return opt_params, cx_cy_skew

    def optimize(self, opt_params, points, observations, cx_cy_skew, mask):
        params, state = self.solver(opt_params, points, observations, cx_cy_skew, mask)
        params = params.block_until_ready()
        return params, state

    def compile(self, points_num, batch_size=8):
        # 6 for pose, 5 for intrinsics
        self.optimize(
            jnp.zeros((batch_size, 8), dtype=float),
            jnp.zeros((batch_size, points_num, 3)),
            jnp.zeros((batch_size, points_num, 2)),
            jnp.zeros((batch_size, 3), dtype=float),
            jnp.zeros((batch_size, points_num), dtype=float),
        )
