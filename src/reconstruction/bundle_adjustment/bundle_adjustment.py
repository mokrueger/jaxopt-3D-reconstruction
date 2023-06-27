import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import optax
from jax import device_put, jit, make_jaxpr, vmap
from jax.experimental import sparse
from jax.lax import associative_scan, fori_loop, scan
from jax.profiler import save_device_memory_profile
from jax.tree_util import Partial, register_pytree_node_class
from jaxopt import GradientDescent, LevenbergMarquardt, NonlinearCG, OptaxSolver
from triangulation_relaxations import so3


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
@vmap
def reproject_point(KE, point):
    return KE @ point


def reproject_points_scan(KE, points_2d, p3d_indices, points_3d):
    points_3d_selected = points_3d.take(p3d_indices, axis=0)
    point_2d_projected = jnp.einsum("ij,bj->bi", KE, points_3d_selected)
    point_2d_projected = point_2d_projected[..., :2] / point_2d_projected[..., 2:3]

    return ((point_2d_projected - points_2d) ** 2).sum(axis=1)


def reproject_points(KE, points_2d, p3d_indices, points_3d):
    points_3d_selected = points_3d.take(p3d_indices, axis=0)
    point_2d_projected = jnp.einsum("ij,bj->bi", KE, points_3d_selected)
    point_2d_projected = point_2d_projected[..., :2] / point_2d_projected[..., 2:3]

    return ((point_2d_projected - points_2d) ** 2).sum(axis=1)
    # print(
    #     KE.shape,
    #     points_2d.shape,
    #     p3d_indices.shape,
    #     points_3d_selected.shape,
    #     point_2d_projected.shape,
    #     error.shape,
    # )


reproject_points_vmap = jit(vmap(reproject_points, in_axes=[0, 0, 0, None]))


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
    def get_residuals(self, opt_params, points_2d_all, p3d_indices_all):
        cam_end_index = self.cam_num * 6
        intr_end_index = cam_end_index + self.cam_num * 5

        # parse opt params
        poses = parse_cam_poses(opt_params[:cam_end_index].reshape((-1, 6)))
        intrinsics = parse_intrinsics(
            opt_params[cam_end_index:intr_end_index].reshape((-1, 5))
        )
        points_3d = opt_params[intr_end_index:].reshape((-1, 3))
        points_4d = jnp.column_stack([points_3d, jnp.ones(len(points_3d))])

        # select corresponding poses and points
        KE = jnp.einsum("bij,bjk->bik", intrinsics, poses)
        print(points_4d.shape, points_2d_all.shape, p3d_indices_all.shape)
        # error = reproject_points_vmap(KE, points_2d_all, p3d_indices_all, points_3d)

        # scan(reproject_points_scan, 0, length=self.cam_num)
        def f_error(i, val):
            _KE = KE[i]
            _p3d_indices = p3d_indices_all[i]
            _points_2d = points_2d_all[i]
            p3d_selected = points_4d.take(_p3d_indices, axis=0)
            p2d_projected = jnp.einsum("ij,bj->bi", _KE, p3d_selected)
            p2d_projected = p2d_projected[..., :2] / p2d_projected[..., 2:3]

            return val + ((p2d_projected - _points_2d) ** 2).sum()

        # return fori_loop(0, self.cam_num, f_error, 0)
        return jnp.array([fori_loop(0, self.cam_num, f_error, 0)])
        return error.sum(axis=1)


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

        def f_solver(matvec, b, ridge=None, init=None, **kwargs):
            x, _ = jscipy.sparse.linalg.cg(matvec, b, x0=init)
            return x

        opt = LevenbergMarquardt(
            residual_fun=sparse.sparsify(self.ba.get_residuals),
            tol=1e-15,
            gtol=1e-15,
            jit=True,
            solver=f_solver,
        )

        # opt = GradientDescent(
        #     fun=self.ba.get_residuals,
        #     tol=1e-6,
        #     jit=True,
        # )

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

        # print(
        #     make_jaxpr(self.ba.get_residuals)(
        #         opt_params, points_2d_all, p3d_indices_all
        #     )
        # )

        params, state = self.solver(opt_params, points_2d_all, p3d_indices_all)
        params = params.block_until_ready()
        return params, state

        # try:
        # except:
        #     save_device_memory_profile("memory.prof")

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
