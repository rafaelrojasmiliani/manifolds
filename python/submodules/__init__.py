"""
Imporst used submodules
"""
from os.path import dirname, join, abspath
import sys

try:
    import code_generation  # noqa
except ImportError:
    DIR = join(dirname(str(abspath(__file__))), 'code_generator/src/')
    sys.path.append(DIR)
    import code_generation  # noqa
