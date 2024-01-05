from enum import Enum, auto

class Optimizer(Enum):
    SGD = auto()
    SGD_MOMENTUM = auto()
    ADAGRAD = auto()
    RMSPROP = auto()
    ADAM = auto()
    