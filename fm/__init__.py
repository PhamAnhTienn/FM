"""
Flow Matching Generative Models (MLP and DiT on MNIST)
"""

from . import core
from . import envs
from . import models
from . import trainers

__version__ = "0.1.0"

__all__ = [
    "core",
    "envs",
    "models",
    "trainers",
]
