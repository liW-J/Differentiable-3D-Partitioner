"""
Differentiable 3D Partitioner
Provides multiple interfaces for 3D placement partitioning
"""

from .core.partitioner import Partitioner3D
from .interfaces.def_interface import DEFInterface
from .interfaces.tensor_interface import TensorInterface

__version__ = "0.1.0"
__all__ = ["Partitioner3D", "DEFInterface", "TensorInterface"]
