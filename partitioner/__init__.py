'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2026-02-05 16:56:01
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2026-02-06 16:48:16
FilePath: /Differentiable-3D-Partitioner/partitioner/__init__.py
Description: Provides multiple interfaces for 3D placement partitioning
'''

from .core.partitioner import Partitioner
from .interfaces.def_interface import DEFInterface
from .interfaces.tensor_interface import TensorInterface

__version__ = "0.1.0"
__all__ = ["Partitioner", "DEFInterface", "TensorInterface"]
