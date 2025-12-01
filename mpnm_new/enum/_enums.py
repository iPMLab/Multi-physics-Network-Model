from enum import IntEnum


class Boundary_Condition_Types(IntEnum):
    dirichlet = 0
    neumann = 1
    outflow = 2
    robin = 3


class Pore_Types(IntEnum):
    void = 0
    solid = 1


class Throat_Types(IntEnum):
    void = 0
    solid = 1
    interface = 2
