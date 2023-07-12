from enum import Enum

import jax
import jax.numpy as jnp


@jax.jit
def l2_loss(y, x):
    return (y - x) ** 2


@jax.jit
def cauchy_loss(y, x, scale=1):
    return jnp.log(1 + l2_loss(y, x) ** scale)


class JaxLossFunction(Enum):
    L2 = l2_loss
    CAUCHY = cauchy_loss
