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
    def get_residuals(self, params, points, observations, cx_cy_skew):
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

    def prepare_params(self, poses0, fx, fy):
        return jnp.array([*pose_mat_to_vec(poses0), fx, fy])

    def optimize(self, opt_params, cx_cy_skew, points_gpu, observations_gpu):
        params, state = self.solver(
            opt_params, points_gpu, observations_gpu, cx_cy_skew
        )

        params = params.block_until_ready()

        return params, state

    def compile(self, points_num):
        params, _ = self.optimize(
            jnp.zeros(8),  # 6 pose, 2 intrinsics
            jnp.zeros(3),  # intrinsics (fx, fy, cx, cy, skew)
            jnp.zeros((points_num, 3)),  # 3d points
            jnp.zeros((points_num, 2)),  # observations (2d points)
        )
        params.block_until_ready()
