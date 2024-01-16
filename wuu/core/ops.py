import numpy as np
from copy import deepcopy
from . import Node


class Operator(Node):
    """
    These Nodes have the calculation function
    No instance
    """
    pass


class Add(Operator):
    """ Adds several tensors

    """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def compute_value(self):
        for source in self._sources:
            if self.value is None:
                self.value = np.copy(source.value)
            else:
                self.value += source.value

    def compute_jacobi(self, source) -> np.ndarray:
        return np.eye(len(self.value))


class Mul(Operator):
    """ Multiplies two tensors

    """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def compute_value(self):
        if len(self._sources) != 2:
            raise ValueError("The num of factor isn't 2")
        return np.dot(self._sources[0], self._sources[1])

    def compute_part_jacobi(self, to):
        if self is to.get_sources()[0]:
            return np.dot(to.jacobi, to.compute_jacobi(self))
        else:
            return np.dot(to.compute_jacobi(self), to.jacobi)

    def compute_jacobi(self, source) -> np.ndarray:
        if len(self._sources) != 2:
            raise ValueError("The num of factor isn't 2")
        # get another factor
        idx = 1 if source is self._sources[0] else 0
        return np.transpose(np.copy(self._sources[idx]))


