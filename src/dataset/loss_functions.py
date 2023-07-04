from enum import Enum
from typing import Any, Callable

import numpy as np


def cauchy_loss(scale=1):
    return lambda x: np.log(1 + x ** scale)


def trivial_loss():
    return lambda x: x


class LossFunction(Enum):
    CAUCHY_LOSS: Callable[[Any], Any] = cauchy_loss()
    TRIVIAL_LOSS: Callable[[Any], Any] = trivial_loss()
