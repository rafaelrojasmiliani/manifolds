"""
First Try
"""

import sympy as sp
from os.path import dirname, join, abspath
from os import path
# from submodules import code_generator
try:
    from code_generation import cpp_generator, code_generator
except ImportError:
    import sys
    DIR = join(dirname(str(abspath(__file__))),
               '../submodules/code_generator/src')
    sys.path.append(DIR)
    from code_generation import cpp_generator, code_generator


def sp_matrix_to_code(_cpp: code_generator.CodeFile, _sp_matrix: sp.Matrix):
    if _sp_matrix.cols == 1 or _sp_matrix.rows == 1:
        for i, element in enumerate(list(_sp_matrix)):
            _cpp(f'result({i}) = ' + sp.printing.cxxcode(element)+';')

    else:
        for i in range(_sp_matrix.rows):
            for j in range(_sp_matrix.cols):
                _cpp(f'result({i}, {j}) = ' +
                     sp.printing.cxxcode(_sp_matrix[i, j])+';')


def sp_array_to_code(_cpp: code_generator.CodeFile, _sp_array: sp.Array):

    rank = _sp_array.rank()
    shape = _sp_array.shape()

    if rank == 3:
        for i1 in range(shape[0]):
            for i2 in range(shape[2]):
                for i3 in range(shape[3]):
                    _cpp(f'result({i1}, {i2}, {i3}) = ' +
                         sp.printing.cxxcode(_sp_array[i1, i2, i3])+';')

# codegen = sp.printing.cxxcode


class ChartGenerator:
    """ Base class for charts, parametrizations and change of coordinate
        generation.
        This class contains the main functionalities to generate a .h, a .cpp
        defining a chart from a manifold into Rn with its differntial, its
        inverce an an arbitrary change of coordinates.

        Any chart/parametrization is uniquelly identified by a pair of two
        points on the manifol.

        Prerequisites to define a manifold

        1- Represenation: An Eigen Type
        2- Dimension of the manifold  (std::size_t)
        3- Dimension of the tangent representation (std::size_t).
            Remark: Note that formally the dimension of the tanget space
            of a manifold at any point has the same dimesion that the
            manifold it self. However, we can use another vector space
            to represent the tanget space of a manifold as a subspace
            of another vector space. For example, to represent the sphere
            using points in R3, we can use also a tangent space of dimension 3
        """

    def __init__(self, _cls, _file_name='', _include_dir='./',
                 _source_dir='./', _test_dir=''):
        """ The common variables for chart generation
            _name: String, name to give manifold element
            _representation: C++ Eigen type, which can be  a matrix
            _dimension: intger
            _tanget_repr_dimension: integer
        """

        assert path.isdir(_include_dir), "_include_dir must be a directory"
        assert path.isdir(_source_dir), "_include_dir must be a directory"

        assert hasattr(
            _cls, 'cpp_repr_type'), 'Class must define a c++ type\
                     for its representation'

        self.cls_ = _cls
        # 1. Define files names and file generators
        self.class_name_ = _cls.__name__
        if _file_name:
            self.header_file_name_ = _file_name + '.h'
        else:
            self.header_file_name_ = _cls.__name__.lower() + '.h'

        if _file_name:
            self.source_file_name_ = _file_name + '.cpp'
        else:
            self.source_file_name_ = _cls.__name__.lower() + '.cpp'

        self.test_generator = None
        if _test_dir:
            assert path.isdir(_test_dir), "_test_dir must be a directory"
            self.test_generator = code_generator.CppFile(
                _test_dir+_file_name+'_test.cpp')

        self.source_generator = code_generator.CppFile(
            _source_dir+self.source_file_name_)

        self.header_generator = code_generator.CppFile(
            _include_dir+self.header_file_name_)

        # Build the chart
        chart = cpp_generator.CppFunction(
            name='chart', ret_type='void',
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.chart()))

        chart.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1')
        chart.add_argument(
            'const '+_cls.cpp_repr_type+' &point2')
        chart.add_argument(
            'const '+_cls.cpp_repr_type+' &element')
        chart.add_argument(
            _cls.cpp_coordinates_type+' &result'
        )

        # Build the differential of the chart
        chart_diff = cpp_generator.CppFunction(
            name='chart_diff', ret_type='void',
            implementation_handle=lambda
            _, cpp: sp_matrix_to_code(cpp, _cls.chart_diff()))

        chart_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1')
        chart_diff.add_argument(
            'const '+_cls.cpp_repr_type+' &point2')
        chart_diff.add_argument(
            'const '+_cls.cpp_repr_type+' &element')
        chart_diff.add_argument(
            _cls.chart_differential_type()+' &result'
        )
        # Build the parametrization
        param = cpp_generator.CppFunction(
            name='param', ret_type='void',
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.parametrization()))

        param.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1')
        param.add_argument(
            'const '+_cls.cpp_repr_type+' &point2')
        param.add_argument(
            _cls.cpp_repr_type+' &result')
        param.add_argument(
            'const ' + _cls.cpp_coordinates_type+' &coordinates'
        )

        # Build the differential parametrization
        param_diff = cpp_generator.CppFunction(
            name='param_diff', ret_type='void',
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.parametrization_diff()))

        param_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1')
        param_diff.add_argument(
            'const '+_cls.cpp_repr_type+' &point2')
        param_diff.add_argument(
            'const ' + _cls.cpp_coordinates_type+' &coordinates')
        param_diff.add_argument(
            _cls.param_differential_type() + ' &result')

        def temp(_cpp, _fun):
            _cpp('auto element = '+_cls.cpp_repr_type+'::Random();')
            _cpp(_cls.cpp_repr_type+' result;')
            sp_matrix_to_code(_cpp, _fun)
            _cpp('return result;')

        random_projection = cpp_generator.CppFunction(
            name='random_projection', ret_type=_cls.cpp_repr_type,
            implementation_handle=lambda _, cpp:
                temp(cpp, _cls.random_projection()))

        # Build the change of coordinates
        change_of_coordinates = cpp_generator.CppFunction(
            name='change_of_coordinates', ret_type='void',
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.change_of_coordinates()))
        change_of_coordinates.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1A')
        change_of_coordinates.add_argument(
            'const ' + _cls.cpp_repr_type+' &point2A')
        change_of_coordinates.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1B')
        change_of_coordinates.add_argument(
            'const ' + _cls.cpp_repr_type+' &point2B')
        change_of_coordinates.add_argument(
            'const ' + _cls.cpp_coordinates_type+' &coordinates')
        change_of_coordinates.add_argument(
            _cls.cpp_coordinates_type+' &result')

        # Build the change of coordinates differential
        change_of_coordinates_diff = cpp_generator.CppFunction(
            name='change_of_coordinates_diff', ret_type='void',
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.change_of_coordinates_diff()))
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1A')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point2A')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point1B')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.cpp_repr_type+' &point2B')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.cpp_coordinates_type+' &coordinates')
        change_of_coordinates_diff.add_argument(
            _cls.change_of_coordinates_diff_type()+' &result')

        self.functions_array_ = [chart, chart_diff,
                                 param, param_diff,
                                 random_projection,
                                 change_of_coordinates,
                                 change_of_coordinates_diff]

        self.cls_ = _cls

    def write(self):
        self.source_generator(f'#include<{self.header_file_name_}>')
        self.header_generator('#include<Eigen/Core>')

        for fun in self.functions_array_:
            self.write_source(fun)
            self.write_header(fun)
        # Write Test

        if not self.test_generator:
            return

        self.test_generator(f"""
#include <gtest/gtest.h>
#include <{self.header_file_name_}>
TEST(Manifolds, FaithfullManifolds) {{
    {self.cls_.cpp_repr_type} point1 = random_projection();
    {self.cls_.cpp_repr_type} point2 = random_projection();

    {self.cls_.cpp_repr_type} element = random_projection();
    {self.cls_.cpp_repr_type} element2;

    {self.cls_.chart_differential_type()} d_chart;
    {self.cls_.param_differential_type()} d_param;

    {self.cls_.cpp_coordinates_type} coordinates;

    chart(point1, point2, element, coordinates);
    param(point1, point2, element2, coordinates);

    element.isApprox(element2, 1.0e-8);

    chart_diff(point1, point1, element, d_chart);
    param_diff(point1, point1, coordinates, d_param);

    Eigen::Matrix<double,
        {self.cls_.dimension},
        {self.cls_.dimension}>::Identity().isApprox(d_chart*d_param, 1.0e-8);


    {self.cls_.cpp_repr_type} point1A = random_projection();
    {self.cls_.cpp_repr_type} point2A = random_projection();

    {self.cls_.cpp_coordinates_type} coordinates2;
    {self.cls_.cpp_coordinates_type} coordinates3;
    change_of_coordinates(point1, point2, point1A, point2A,
                            coordinates, coordinates2);
    change_of_coordinates(point1A, point2A, point1, point2,
                            coordinates2, coordinates3);

    coordinates3.isApprox(coordinates, 1.0e-8);

    {self.cls_.change_of_coordinates_diff_type()} nomAdiff;
    {self.cls_.change_of_coordinates_diff_type()} Anomdiff;
    change_of_coordinates_diff(point1, point2, point1A,point2A,
                                coordinates, nomAdiff);
    change_of_coordinates_diff(point1A, point2A, point1, point2,
                                coordinates2, Anomdiff);

    Eigen::Matrix<double,
        {self.cls_.dimension},
        {self.cls_.dimension}>::Identity().isApprox(Anomdiff*nomAdiff, 1.0e-8);
}}

int main(int argc, char **argv) {{

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}}
""")

    def write_source(self, _function: cpp_generator.CppFunction):
        _function.render_to_string_implementation(self.source_generator)

    def write_header(self, _function: cpp_generator.CppFunction):
        _function.render_to_string_declaration(self.header_generator)

        #        # 3. Define the methods
        #        # 3.1 Value
        #        value_method = cpp_generator.CppFunction(
        #            name='value_on_repr', ret_type='bool',
        #            implementation_handle=lambda _, cpp: self.chart_value_impl(cpp))
        #        value_method.add_argument(
        #            f'const {_representation}& '+self.manifold_element_symbol_)
        #        value_method.add_argument(
        #            f'Eigen::Matrix<double, {self.dimension_}, 1>& result')
        #        self.chart_class_.add_method(value_method)
        #
        #        # 3.2 Differential
        #        diff_method = cpp_generator.CppFunction(
        #            name='diff_from_repr', ret_type='bool',
        #            implementation_handle=lambda _, cpp: self.chart_diff_impl(cpp))
        #
        #        diff_method.add_argument(
        #            f'const {_representation}& '+self.manifold_element_symbol_)
        #        diff_method.add_argument(
        #            f'const Eigen::Matrix<double, {self.tangent_repr_dimension_},' +
        #            ' 1>& result')
        #        self.chart_class_.add_method(diff_method)
        #
        #        # 4. Constructor
        #        constructor = cpp_generator.CppFunction(
        #            name='Chart',
        #            implementation_handle=lambda _, cpp: self.constructor(cpp))
        #        constructor.add_argument(f'const {self.class_name_}& _p1')
        #        constructor.add_argument(f'const {self.class_name_}& _p2')
        #        self.chart_class_.add_method(constructor)
        #
        #    def get_sympy_element_repr(self) -> sp.Matrix:
        #        """ Returns the sympy representation of an element of the manifold"""
        #        return self.sympy_element_repr_
        #
        #    def get_sympy_point_1_repr(self) -> sp.Matrix:
        #        """ Returns the sympy representation of the point 1"""
        #        return self.point_1_repr_
        #
        #    def get_sympy_point_2_repr(self) -> sp.Matrix:
        #        """ Returns the sympy representation of the point 2"""
        #        return self.point_2_repr_
        #
        #    def get_sympy_coordinates_repr(self) -> sp.Matrix:
        #        """ Returns the sympy representation of the coordinates"""
        #        return self.coordinate_symbol_
        #
        #    def write(self):
        #        internal_vars = self.internal_variables()
        #
        #        for var in internal_vars:
        #            self.chart_class_.add_variable(var)
        #
        #        self.chart_class_.render_to_string_declaration(self.header_generator)
        #        self.chart_class_.render_to_string_implementation(
        #            self.source_generator)
        #
        #    def internal_variables(self):
        #        return [
        #            cpp_generator.CppVariable(name='x_', type='Eigen::Vector3d'),
        #            cpp_generator.CppVariable(name='y_', type='Eigen::Vector3d'),
        #            cpp_generator.CppVariable(name='z_', type='Eigen::Vector3d'),
        #        ]
        #
        #    def constructor(self, cpp):
        #        """ Stuff to do in the constructor"""
        #        pass
        #
        def chart_value_impl(self, _cpp):

            res = self.chart()

            for i, com in enumerate(res):
                _cpp(f'_result({i}) = ' + sp.printing.cxxcode(com)+';')

        #    def chart_diff_impl(self, _cpp):
        #
        #        the_chart = self.chart()
        #        the_domain = self.tangent_representation()
        #
        #        for i, com in enumerate(the_chart):
        #            for j, var in enumerate(the_domain):
        #                _cpp(f'_result({i},{j}) = ' +
        #                     sp.printing.cxxcode(com.diff(var))+';')
        #
        #    def tangent_representation(self):
        #
        #        return colvector('_p', self.tangent_repr_dimension_)
        #
        #    def manifold_representation(self):
        #        return colvector('_p', 3)
        #
        #    def chart_differentiation(self):
        #        pass
        #
        #    def parametrization_differentiation(self):
        #        pass
        #
        #
        # class Sphere(ChartGenerator):
        #    def __init__(self):
        #        super().__init__('S2', 'Eigen::Vectord3', 2, 3)
        #        pass
        #
        #    def chart(self):
        #        element = self.get_sympy_element_repr()
        #        point1 = self.get_sympy_point_1_repr()
        #        point2 = self.get_sympy_point_2_repr()
        #
        #        midpoint = 0.5*(point1 + point2)
        #
        #        x_normal_vector = midpoint/midpoint.norm()
        #        y_normal_vector = point2 - point2.dot(x_normal_vector)
        #        y_normal_vector = y_normal_vector/y_normal_vector.norm()
        #
        #        z_normal_vector = x_normal_vector.cross(y_normal_vector)
        #
        #        x_coordinate = element.dot(x_normal_vector)
        #        y_coordinate = element.dot(y_normal_vector)
        #        z_coordinate = element.dot(z_normal_vector)
        #
        #        return [sp.acos(z_coordinate), sp.atan2(y_coordinate, x_coordinate)]
        #
        #    def parametrization(self):
        #
        #        coordinates = self.get_sympy_coordinates_repr()
        #
        #        inclination = coordinates[0]
        #        azimuth = coordinates[1]
        #
        #        return [inclination, azimuth]
        #
        #
        # class SO3(ChartGenerator):
        #    def __init__(self):
        #        super().__init__('SO3', 'Eigen::Matrix<double, 3, 3>', 3, 3)
        #        pass
        #
        #    def chart(self):
        #        # https://math.stackexchange.com/questions/4389545/how-to-get-the-matrix-logarithm-of-the-rotation
        #        element = self.get_sympy_element_repr()
        #        point1 = self.get_sympy_point_1_repr()
        #        point2 = self.get_sympy_point_2_repr()
        #
        #        midmatrix = point1
        #
        #        mid_to_element = midmatrix.transpose*element
        #
        #        theta = sp.acos((mid_to_element.trace()-1)/2)
        #
        #        log_of_matrix = theta/(2*sp.sin(theta)) * \
        #            (mid_to_element-mid_to_element.transpose())
        #
        #        return [-log_of_matrix[1, 2],
        #                log_of_matrix[0, 2],
        #                -log_of_matrix[0, 1]]
        #
        #    def parametrization(self):
        #        # Use rodrigues formula
        #        # https://www.emis.de/journals/BJGA/v18n2/B18-2-an.pdf
        #
        #        coordinates = self.get_sympy_coordinates_repr()
        #
        #        theta = coordinates.norm()
        #
        #        matrix_x = sp.Matrix([[0, -coordinates[2], coordinates[1]],
        #                              [coordinates[2], 0, -coordinates[0]],
        #                              [-coordinates[1], coordinates[0], 0]])
        #
        #        return sp.eye(3, 3) + \
        #            sp.sin(theta)/theta * matrix_x + \
        #            (1-sp.cos(theta))/theta**2 * matrix_x**2
        #
        #    def chart_differentiation(self):
        #        pass
        #
        #    def parametrization_differentiation(self):
        #        pass
        #
        #
        # c = ChartGenerator('S2', 'Eigen::Vector3d', 2, 3)
        # c.write()
