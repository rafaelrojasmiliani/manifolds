
import unittest
import sys
from sympy import lambdify
import numpy as np
import tracemalloc

tracemalloc.start()
try:
    from sym_manifolds.generator import ChartGenerator
    from sym_manifolds.sphere import Sphere
except ImportError:
    from os.path import dirname, join, abspath
    DIR = join(dirname(str(abspath(__file__))), '../python/')
    sys.path.append(DIR)
    from sym_manifolds.sphere import Sphere
    from sym_manifolds.generator import ChartGenerator


class MyTest(unittest.TestCase):
    """ Test """

    def __init__(self, *args, **kwargs):
        """ Test """
        super().__init__(*args, **kwargs)

    def test(self):
        pass

        # ChartGenerator(Sphere, "test", "./generation_tests/include/",
        #                "./generation_tests/src/",
        #                "./generation_tests/tests/").write()


if __name__ == '__main__':
    unittest.main()
