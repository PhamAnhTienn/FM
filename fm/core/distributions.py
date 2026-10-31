from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import torch
import torch.nn as nn

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int):
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
        """
        pass
    
    
class LabeledSampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int):
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b d
            - labels: b
        """
        pass
    
    
class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # To track the state of curr device

    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)