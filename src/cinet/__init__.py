# read version from installed package
from importlib.metadata import version
__version__ = version("cinet")

from .interfaces import deepCINET, ECINET