from enum import Enum
from typing import Literal

#  Literals must be the enum.names
TransformationDirectionLiteral = Literal["C2W", "W2C"]
PoseFormatLiteral = Literal["QT", "RT", "T"]
CoordinateSystemLiteral = Literal["LH", "RH"]


class Mapper(Enum):
    @classmethod
    def get_enum_by_name(cls, name):
        return cls.__members__.get(name)


class TransformationDirection(Mapper):
    C2W = "C2W"
    W2C = "W2C"


class CoordinateSystem(Mapper):
    LH = "LH"
    RH = "RH"
    UNITY = LH
    COLMAP = RH


class PoseFormat(Mapper):
    QT = "QT"
    RT = "RT"
    T = "T"
