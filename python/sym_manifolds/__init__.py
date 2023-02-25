"""
Definition of symbolic atlases
"""

import sympy as sp


class SymManifold:
    def __init__(self, _rows: int = 1, _cols: int = 1):
        self.rows_ = _rows
        self.cols_ = _cols
        self.repr_dim_ = _rows*_cols

        if _rows < 1 or _cols < 0:
            raise ValueError('cols or rows cannot be negative')

    def point(self, _name: str):

        if self.rows_ == 1 and self.cols_ == 1:
            return sp.symbols(
                fr'{_name}',
                real=True)

        if self.rows_ == 1:
            return sp.Matrix(self.rows_, self.cols_,
                             sp.symbols(
                                 fr'{_name}((0:{self.cols_}))',
                                 real=True))
        if self.cols_ == 1:
            return sp.Matrix(self.rows_, self.cols_,
                             sp.symbols(
                                 fr'{_name}((0:{self.rows_}))',
                                 real=True))

        return sp.Matrix(self.rows_, self.cols_,
                         sp.symbols(
                             fr'{_name}((0:{self.rows_})\,(0:{self.cols_}))',
                             real=True))


class Rn(SymManifold):
    def __init__(self, _n):
        super().__init__(_n)


class Atlas:
    """ Class able to statically generate sympy expressions which represent a
    chart on a manifold.
    This Sympy expressions must have the following characteristics:
    1. The manifold element is of class sympy.Matrix
    2. The manifold element in named element
    3. The Rn element is a row vector of class sympy.Matrix
    4. The Rn element is called coordinates
    5. Each expression contains the following named sympy variables
        - element(i, j)
        - coordinate(i)
        - point1(i, j)
        - point2(i, j)
    """

    def __init__(self):
        pass

    @classmethod
    def chart(cls):
        cls._sanity_check()
        point1 = cls.get_sym_point1()
        point2 = cls.get_sym_point2()
        element = cls.get_sym_element()
        result = cls.chart_impl(point1, point2, element)
        assert isinstance(
            result, sp.matrices.dense.MutableDenseMatrix), \
            "The result must be a dense matrix"
        assert result.cols == 1, "The result must be a col vector"
        assert result.rows == cls.dimension, \
            "The result must have as many coordinates as" + \
            " the dimension of the manifold"
        return result

    @classmethod
    def chart_diff(cls):
        cls._sanity_check()
        element = cls.manifold.point("element")
        if (not hasattr(cls, 'chart_diff_impl')):
            assert cls.cols == 1 or cls.rows == 1
            chart_expression = cls.chart()
            result = sp.zeros(cls.dimension,
                              cls.tanget_repr_dim)
            for i in range(cls.dimension):
                for j in range(cls.tanget_repr_dim):
                    result[i, j] = chart_expression[i].diff(element[j])

            return result

    @classmethod
    def parametrization(cls):
        cls._sanity_check()
        point1 = cls.get_sym_point1()
        point2 = cls.get_sym_point2()
        coordinates = cls.get_sym_coordinates()
        result = cls.param_impl(point1, point2, coordinates)
        return result

    @classmethod
    def parametrization_diff(cls):
        cls._sanity_check()
        if (not hasattr(cls, 'chart_diff_impl')):
            assert cls.cols == 1 or cls.rows == 1
            param = cls.parametrization()
            coordinates = cls.get_sym_coordinates()
            result = sp.zeros(cls.tanget_repr_dim,
                              cls.dimension)
            for i in range(cls.tanget_repr_dim):
                for j in range(cls.dimension):
                    result[i, j] = param[i].diff(coordinates[j])

            return result

    @classmethod
    def change_of_coordinates(cls):
        cls._sanity_check()
        point1A = cls.manifold.point("point1A")
        point2A = cls.manifold.point("point2A")
        coordinates = cls.coordinates.point("coordinates")
        param = cls.param_impl(point1A, point2A, coordinates)

        point1B = cls.manifold.point("point1B")
        point2B = cls.manifold.point("point2B")
        element = cls.manifold.point("element")
        coordinatesB = cls.chart_impl(point1B, point2B, element)

        coordinatesB = coordinatesB.subs(zip(list(element), list(param)))

        return coordinatesB

    @classmethod
    def change_of_coordinates_diff_k(cls, k: int) -> sp.Array:

        coordinates = cls.coordinates.point("coordinates")
        coc = cls.change_of_coordinates()
        if k == 0:
            return coc
        result = sp.derive_by_array(coc, list(coordinates))
        for i in range(k-1):
            result = sp.derive_by_array(result, list(coordinates))
        return result

    @classmethod
    def change_of_coordinates_diff(cls) -> sp.Matrix:
        cls._sanity_check()

        coc = cls.change_of_coordinates()
        coordinates = cls.get_sym_coordinates()
        result = sp.zeros(cls.dimension,
                          cls.dimension)
        for i in range(cls.dimension):
            for j in range(cls.dimension):
                result[i, j] = coc[i].diff(coordinates[j])

        return result

    @classmethod
    def get_sym_coordinates(cls):
        cls._sanity_check()
        return cls.coordinates.point(cls.coordinates_name())

    @classmethod
    def get_sym_element(cls):
        return cls.manifold.point(cls.element_name())

    @classmethod
    def get_sym_point1(cls):
        return cls.manifold.point(cls.point1_name())

    @classmethod
    def get_sym_point2(cls):
        return cls.manifold.point(cls.point2_name())

    @classmethod
    def chart_differential_type(cls):
        return f'Eigen::Matrix<double, {cls.dimension}, {cls.tanget_repr_dim}>'

    @classmethod
    def param_differential_type(cls):
        return f'Eigen::Matrix<double, {cls.tanget_repr_dim}, {cls.dimension}>'

    @classmethod
    def change_of_coordinates_diff_type(cls):
        return f'Eigen::Matrix<double, {cls.dimension}, {cls.dimension}>'

    @classmethod
    def random_projection(cls):
        cls._sanity_check()
        return cls.random_repr_projection_impl(cls.get_sym_element())

    @classmethod
    def point1_name(cls):
        return "point1"

    @classmethod
    def point2_name(cls):
        return "point2"

    @classmethod
    def element_name(cls):
        return "element"

    @classmethod
    def coordinates_name(cls):
        return "coordinates"

    @classmethod
    def _sanity_check(cls):
        assert hasattr(
            cls, 'dimension'), "Must have a dimension static variable"
        assert hasattr(
            cls, 'manifold'), "Must have a manifold static variable"
        assert hasattr(
            cls, 'coordinates'), "Must have an coordinates static varible"
        assert hasattr(
            cls, 'chart_impl'), "Must implement a chart"
        assert hasattr(
            cls, 'param_impl'), "Must implement a paramtrization"
        assert hasattr(
            cls, 'tanget_repr_dim'), "Must implement a dimension of" \
            + " the tanget representation"
        assert hasattr(
            cls, 'cpp_repr_type'), "Must implement a representation type"

    @classmethod
    def get_representation_type(cls):
        return cls.cpp_repr_type

    @classmethod
    def get_tangent_representation_type(cls):
        return "Eigen::Matrix<double, " + \
            f"{cls.get_tangent_representation_dimension()}, 1>"

    @classmethod
    def get_dimension(cls):
        return cls.dimension

    @classmethod
    def get_tangent_representation_dimension(cls):
        return cls.tanget_repr_dim

    @classmethod
    def get_coordinates_type(cls):
        return "Eigen::Matrix<double, " + \
            f"{cls.get_dimension()}, 1>"
