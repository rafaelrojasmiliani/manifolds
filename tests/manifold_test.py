"""
Test of linear manifolds. Load and test operations
"""

import unittest
import tracemalloc
import manifolds

tracemalloc.start()


class MyTest(unittest.TestCase):
    """ Test """

    def __init__(self, *args, **kwargs):
        """ Test """
        super().__init__(*args, **kwargs)

        v = manifolds.linear_manifolds.R2()
        m = v.crepr()

    def test(self):
        pass

        # ChartGenerator(Sphere, "test", "./generation_tests/include/",
        #                "./generation_tests/src/",
        #                "./generation_tests/tests/").write()


if __name__ == '__main__':
    unittest.main()
