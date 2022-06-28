# read version from installed package
from importlib.metadata import version
__version__ = version("cinet")

from .hello_world import hello_name
from .cinet import cinet, getCINETSampleInput