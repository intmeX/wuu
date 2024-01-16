import numpy as np
from abc import ABC, abstractmethod
from . import Graph, default_graph


class Node(ABC):
    """ The Node of computational graph

    """

    def __init__(
            self,
            *sources: 'Node',
            graph: Graph = default_graph,
            trainable: bool = True,
    ):
        self._sources = set(sources)
        self._tos = set()
        for source in sources:
            source.add_to(self)
        self.trainable = trainable
        self.graph = graph
        graph.add_node(self)
        self.value = None
        self.jacobi = None

    def add_to(self, node: 'Node'):
        self._tos.add(node)

    def get_tos(self):
        return self._tos

    def add_source(self, node: 'Node'):
        self._sources.add(node)

    def get_source(self):
        return self._sources

    @abstractmethod
    def compute_value(self):
        """
        when sources' value has already been computed,
        use this method to compute the value
        """
        raise NotImplementedError

    def forward(self):
        for source in self._sources:
            if source.value is None:
                source.forward()
        self.compute_value()

    @abstractmethod
    def compute_jacobi(self, source) -> np.ndarray:
        """
        when the jacobi has already been computed,
        use this method to compute self
        contribution for one source's jacobi matrix
        """
        raise NotImplementedError

    def compute_part_jacobi(self, to) -> np.ndarray:
        return to.jacobi * to.compute_jacobi(self)

    def backward(self):
        if self.value is None:
            raise ValueError("No value has been computed before backward")
        if not self._tos:
            # the jacobi of result node
            self.jacobi = np.eye(len(self.value), dtype=np.float32)
            return self.jacobi
        self.jacobi = np.zeros_like(self.value)
        for to in self._tos:
            if to.jacobi is None:
                to.backward()
            # the order of multiply will impact on mul op,
            # so use an extra method
            self.jacobi += self.compute_part_jacobi(to)
        return self.jacobi

    def reset(self):
        self.value = None
        self.jacobi = None
