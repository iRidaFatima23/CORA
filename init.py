"""
CORA: Cognitive Optimization and Refinement Architecture
"""

__version__ = "1.0.0"
__author__ = "CORA Research Team"
__email__ = "your-email@example.com"

from .core.cora_engine import CORA
from .core.ckal import CKAL
from .core.sol import SOL
from .core.macl import MACL

__all__ = [
    "CORA",
    "CKAL", 
    "SOL",
    "MACL",
]
