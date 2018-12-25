from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    @abstractmethod
    def record(self):
        pass

    @abstractmethod
    def apply(self):
        pass
