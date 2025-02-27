from abc import ABC, abstractmethod # abstract classes

class PE(ABC):
    @abstractmethod
    def get_pitch(self, x, fs):
        raise NotImplementedError('Pitch Extractor not implemented')