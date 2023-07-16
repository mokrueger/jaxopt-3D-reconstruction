import jax
import jax.numpy as jnp
import numpy as np


def get_reprojection_residuals_cpu(pose, points, observations, intrinsics, mask):
    KE = np.einsum("ij,jk->ik", intrinsics, pose[:3, :])
    x = np.einsum("ij,hj->hi", KE, points)  # reprojected_points
    x = x[..., :2] / x[..., 2:3]  # 2:3 to prevent axis from being removed

    res = ((observations - x) ** 2).sum(axis=1) / 250000
    return np.where(mask, res, np.zeros_like(res))


def to_gpu(data):
    if isinstance(data, (list, tuple)):
        return jnp.array([jax.device_put(i) for i in data])
    return jax.device_put(data)


@jax.jit
def rot_mat_from_vec(rodrigues_vec):
    theta = jnp.linalg.norm(rodrigues_vec)
    r = rodrigues_vec / theta
    I = jnp.eye(3, dtype=float)
    r_rT = jnp.outer(r, r)
    r_cross = jnp.cross(jnp.eye(3), r)
    return jnp.cos(theta) * I + (1 - jnp.cos(theta)) * r_rT + jnp.sin(theta) * r_cross


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
    n[nonzero_indices] *= (angle[nonzero_indices] / norms[nonzero_indices])[..., None]

    return jnp.concatenate([n, pose[:3, 3]])


@jax.jit
def parse_intrinsics(params, cx_cy_skew):
    # fmt: off
    return jnp.array( 
        [
            [params[0], cx_cy_skew[2], cx_cy_skew[0]],
            [        0,     params[1], cx_cy_skew[1]],
            [        0,             0,             1],
        ]
    )
    # fmt: on


@jax.jit
def parse_cam_pose(cam_vec):
    return jnp.concatenate(
        [rot_mat_from_vec(cam_vec[:3]), cam_vec[3:6, jnp.newaxis]], axis=1
    )


parse_intrinsics_vmap = jax.jit(jax.vmap(parse_intrinsics))
parse_cam_pose_vmap = jax.jit(jax.vmap(parse_cam_pose))
