from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, gradient):
        pass

    @abstractmethod
    def decode(self, gradients):
        pass
