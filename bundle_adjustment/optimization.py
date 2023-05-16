import jax.numpy as jnp
from jax import vmap, jit
from jaxopt import LevenbergMarquardt

from triangulation_relaxations import so3


@jit
def project_single_point_gpu(tree_arg):
    E, x, K = tree_arg
    return K @ E @ x


project_point_vmap = jit(
    vmap(
        jit(vmap(project_single_point_gpu, in_axes=((None, 0, None),))),
        in_axes=((0, 0, 0),),
    )
)
# project_point_vmap = jit(vmap(project_single_point_gpu, in_axes=((None, 0, None),)))


@jit
def reproject_gpu(points: jnp.array, pose: jnp.array, K: jnp.array):
    _pose = jnp.linalg.inv(pose)
    # print((_pose.shape, points.shape, K.shape))
    x = project_point_vmap((_pose, points, K))
    # print(x.shape)
    x = x[..., :2] / x[..., 2:3]
    return x


@jit
def rotvec_to_r_gpu(rodrigues_vec):
    theta = jnp.linalg.norm(rodrigues_vec)
    r = rodrigues_vec / theta
    I = jnp.eye(3, dtype=float)
    r_rT = jnp.outer(r, r)
    r_cross = jnp.cross(jnp.eye(3), r)
    return jnp.cos(theta) * I + (1 - jnp.cos(theta)) * r_rT + jnp.sin(theta) * r_cross


@jit
@vmap
def x_to_pose_gpu(x):
    R = rotvec_to_r_gpu(x[:3])
    return jnp.block([[R, x[3:, jnp.newaxis]], [jnp.zeros(3).T, 1]])


def pose_to_x_gpu(pose):
    return jnp.concatenate([so3.r_to_rotvec(pose.R), pose.t])


@jit
def get_reprojection_residuals_gpu(pose, points, observations, intrinsics):
    reprojected_points = reproject_gpu(points, x_to_pose_gpu(pose), intrinsics)
    # print(((observations - reprojected_points) ** 2).sum(axis=[0, 2]).shape)
    return ((observations - reprojected_points) ** 2).sum(axis=[0, 2])


lm = LevenbergMarquardt(
    residual_fun=get_reprojection_residuals_gpu,
    tol=1e-15,
    gtol=1e-15,
    jit=True,
    solver="inv",
)

jitted_lm = jit(lm.run)


def compile_lm_gpu(_pose0, points_gpu, observations_gpu, intrinsics_gpu):
    _points_gpu = jnp.full_like(points_gpu, 0)
    _observations_gpu = jnp.full_like(observations_gpu, 0)
    _intrinsics_gpu = jnp.full_like(intrinsics_gpu, 0)
    jitted_lm(
        _pose0,
        points=_points_gpu,
        observations=_observations_gpu,
        intrinsics=_intrinsics_gpu,
    ).params.block_until_ready()


def run_lm_gpu(_pose0, points_gpu, observations_gpu, intrinsics_gpu):
    return jitted_lm(
        _pose0,
        points=points_gpu,
        observations=observations_gpu,
        intrinsics=intrinsics_gpu,
    ).params
