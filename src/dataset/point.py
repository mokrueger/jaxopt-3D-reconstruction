import copy
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass
class Point2D:
    identifier: int
    x: float
    y: float
    point3D_identifier: Union[int, None]
    metadata: Dict

    @property
    def xy(self):
        return np.array([self.x, self.y])

    def translate_np(self, xy):
        self.x += xy[0]
        self.y += xy[1]

    def translated_np(self, xy):
        new = copy.deepcopy(self)
        new.translate_np(xy)
        return new


@dataclass
class Point3D:
    identifier: int
    x: float
    y: float
    z: float
    metadata: Dict

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    def translate(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z

    def translated(self, x, y, z):
        new = copy.deepcopy(self)
        new.translate(x, y, z)
        return new

    def translate_np(self, xyz):
        self.x += xyz[0]
        self.y += xyz[1]
        self.z += xyz[2]

    def translated_np(self, xyz):
        new = copy.deepcopy(self)
        new.translate_np(xyz)
        return new
