"""
Defines an atlas on the unit sphere
"""
import sympy as sp
from . import Rn
from . import Atlas
import numpy as np


class Sphere(Atlas):
    manifold = Rn(3)
    coordinates = Rn(2)
    dimension = 2
    tanget_repr_dim = 3
    cols = manifold.cols_
    rows = manifold.rows_
    coordinates_bounds = [[0, 3.1], [-3.1, 3.1]]
    cpp_repr_type = 'Eigen::Vector3d'

    def __init__(self):
        pass

    @staticmethod
    def random_repr_projection_impl(representation: sp.Matrix) -> sp.Matrix:
        return representation / representation.norm()

    @staticmethod
    def get_random():
        point = np.random.rand(3)
        point /= np.linalg.norm(point)

        return sp.Matrix(point)

    @staticmethod
    def get_random_coordinats():
        inclination = np.random.uniform(0, 3.1)
        azimuth = np.random.uniform(-3.1, 3.1)

        return sp.Matrix([inclination, azimuth])

    @staticmethod
    def chart_impl(point1, point2, representation):

        midpoint = 0.5*(point1 + point2)

        x_normal_vector = midpoint/midpoint.norm()
        y_normal_vector = point2 - point2.dot(x_normal_vector)*x_normal_vector
        y_normal_vector = y_normal_vector/y_normal_vector.norm()

        z_normal_vector = x_normal_vector.cross(y_normal_vector)

        x_coordinate = representation.dot(x_normal_vector)
        y_coordinate = representation.dot(y_normal_vector)
        z_coordinate = representation.dot(z_normal_vector)

        return sp.Matrix([sp.acos(z_coordinate),
                          sp.atan2(y_coordinate, x_coordinate)])

    @staticmethod
    def param_impl(point1, point2, coordinates):

        inclination = coordinates[0]
        azimuth = coordinates[1]

        midpoint = 0.5*(point1 + point2)
        x_normal_vector = midpoint/midpoint.norm()
        y_normal_vector = point2 - point2.dot(x_normal_vector)*x_normal_vector
        y_normal_vector = y_normal_vector/y_normal_vector.norm()
        z_normal_vector = x_normal_vector.cross(y_normal_vector)

        result = sp.cos(azimuth) * sp.sin(inclination) * x_normal_vector \
            + sp.sin(azimuth) * sp.sin(inclination) * y_normal_vector \
            + sp.cos(inclination) * z_normal_vector

        return result
