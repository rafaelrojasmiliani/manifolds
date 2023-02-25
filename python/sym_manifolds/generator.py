"""
First Try
"""

import sympy as sp
from os.path import dirname, join, abspath
from os import path
# from submodules import code_generator
try:
    from code_generation.cpp.cpp_class import CppClass
    from code_generation.cpp.cpp_variable import CppVariable
    from code_generation.cpp.cpp_using_type import CppUsingType
    from code_generation.core.code_generator import CppFile
except ModuleNotFoundError:
    import sys
    DIR = join(dirname(str(abspath(__file__))),
               '../submodules/code_generator/src')
    sys.path.append(DIR)
    from code_generation.cpp.cpp_class import CppClass
    from code_generation.cpp.cpp_variable import CppVariable
    from code_generation.cpp.cpp_using_type import CppUsingType
    from code_generation.core.code_generator import CppFile


def sp_matrix_to_code(_cpp: CppFile, _sp_matrix: sp.Matrix):
    if _sp_matrix.cols == 1 or _sp_matrix.rows == 1:
        for i, element in enumerate(list(_sp_matrix)):
            _cpp(f'result({i}) = ' + sp.printing.cxxcode(element)+';')

    else:
        for i in range(_sp_matrix.rows):
            for j in range(_sp_matrix.cols):
                _cpp(f'result({i}, {j}) = ' +
                     sp.printing.cxxcode(_sp_matrix[i, j])+';')


def sp_array_to_code(_cpp: CppFile, _sp_array: sp.Array):

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
        self.class_name_ = _cls.__name__+'Atlas'
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
            self.test_generator = CppFile(
                _test_dir+_file_name+'_test.cpp')

        self.source_generator = CppFile(
            _source_dir+self.source_file_name_)

        self.header_generator = CppFile(
            _include_dir+self.header_file_name_)

        # Build the chart
        chart = CppClass.CppMethod(
            name='chart', ret_type='void', is_static=True,
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.chart()))

        chart.add_argument(
            'const ' + _cls.get_representation_type()+' &point1')
        chart.add_argument(
            'const '+_cls.get_representation_type()+' &point2')
        chart.add_argument(
            'const '+_cls.get_representation_type()+' &element')
        chart.add_argument(
            _cls.get_coordinates_type()+' &result'
        )

        # Build the differential of the chart
        chart_diff = CppClass.CppMethod(
            name='chart_diff', ret_type='void', is_static=True,
            implementation_handle=lambda
            _, cpp: sp_matrix_to_code(cpp, _cls.chart_diff()))

        chart_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point1')
        chart_diff.add_argument(
            'const '+_cls.get_representation_type()+' &point2')
        chart_diff.add_argument(
            'const '+_cls.get_representation_type()+' &element')
        chart_diff.add_argument(
            _cls.chart_differential_type()+' &result'
        )
        # Build the parametrization
        param = CppClass.CppMethod(
            name='param', ret_type='void', is_static=True,
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.parametrization()))

        param.add_argument(
            'const ' + _cls.get_representation_type()+' &point1')
        param.add_argument(
            'const '+_cls.get_representation_type()+' &point2')
        param.add_argument(
            'const ' + _cls.get_coordinates_type()+' &coordinates'
        )
        param.add_argument(
            _cls.get_representation_type()+' &result')

        # Build the differential parametrization
        param_diff = CppClass.CppMethod(
            name='param_diff', ret_type='void', is_static=True,
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.parametrization_diff()))

        param_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point1')
        param_diff.add_argument(
            'const '+_cls.get_representation_type()+' &point2')
        param_diff.add_argument(
            'const ' + _cls.get_coordinates_type()+' &coordinates')
        param_diff.add_argument(
            _cls.param_differential_type() + ' &result')

        def temp(_cpp, _fun):
            _cpp('auto element = '+_cls.get_representation_type()+'::Random();')
            _cpp(_cls.get_representation_type()+' result;')
            sp_matrix_to_code(_cpp, _fun)
            _cpp('return result;')

        random_projection = CppClass.CppMethod(
            name='random_projection', ret_type=_cls.get_representation_type(),
            is_static=True,
            implementation_handle=lambda _, cpp:
            temp(cpp, _cls.random_projection()))

        # Build the change of coordinates
        change_of_coordinates = CppClass.CppMethod(
            name='change_of_coordinates', ret_type='void', is_static=True,
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.change_of_coordinates()))
        change_of_coordinates.add_argument(
            'const ' + _cls.get_representation_type()+' &point1A')
        change_of_coordinates.add_argument(
            'const ' + _cls.get_representation_type()+' &point2A')
        change_of_coordinates.add_argument(
            'const ' + _cls.get_representation_type()+' &point1B')
        change_of_coordinates.add_argument(
            'const ' + _cls.get_representation_type()+' &point2B')
        change_of_coordinates.add_argument(
            'const ' + _cls.get_coordinates_type()+' &coordinates')
        change_of_coordinates.add_argument(
            _cls.get_coordinates_type()+' &result')

        # Build the change of coordinates differential
        change_of_coordinates_diff = CppClass.CppMethod(
            name='change_of_coordinates_diff', ret_type='void', is_static=True,
            implementation_handle=lambda _, cpp:
                sp_matrix_to_code(cpp, _cls.change_of_coordinates_diff()))
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point1A')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point2A')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point1B')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.get_representation_type()+' &point2B')
        change_of_coordinates_diff.add_argument(
            'const ' + _cls.get_coordinates_type()+' &coordinates')
        change_of_coordinates_diff.add_argument(
            _cls.change_of_coordinates_diff_type()+' &result')

        self.cpp_class_ = CppClass(
            name=self.class_name_)
        self.functions_array_ = [chart, chart_diff,
                                 param, param_diff,
                                 random_projection,
                                 change_of_coordinates,
                                 change_of_coordinates_diff]

        for fun in self.functions_array_:
            self.cpp_class_.add_method(fun)

        self.cpp_class_.add_variable(
            CppVariable(name='dimension', is_static=True, is_const=True,
                        type='std::size_t',
                        initialization_value=f"{_cls.dimension}"),
            is_private=False)

        self.cpp_class_.add_variable(
            CppVariable(name='tanget_repr_dimension', is_static=True,
                        is_const=True,
                        type='std::size_t',
                        initialization_value=f'{_cls.tanget_repr_dim}'),
            is_private=False)

        self.cpp_class_.add_using_type(CppUsingType(
            name='Representation', type=self.cls_.get_representation_type()))

        self.cpp_class_.add_using_type(CppUsingType(
            name='Tangent', type=self.cls_.get_tangent_representation_type()))

        self.cls_ = _cls

    def write(self):
        self.source_generator(f'#include<{self.header_file_name_}>')
        self.header_generator('#include<Eigen/Core>')

        self.cpp_class_.render_to_string_implementation(self.source_generator)
        self.cpp_class_.render_to_string_declaration(self.header_generator)
        # Write Test
        self.source_generator.close()
        self.header_generator.close()

        if not self.test_generator:
            return

        self.test_generator(f"""// Generated automatically
#include <gtest/gtest.h>
#include <{self.header_file_name_}>
TEST(Manifolds, FaithfullManifolds) {{
    {self.cls_.get_representation_type()} point1 =
                    {self.class_name_}::random_projection();
    {self.cls_.get_representation_type()} point2 =
                    {self.class_name_}::random_projection();

    {self.cls_.get_representation_type()} element =
                                {self.class_name_}::random_projection();
    {self.cls_.get_representation_type()} element2;

    {self.cls_.chart_differential_type()} d_chart;
    {self.cls_.param_differential_type()} d_param;

    {self.cls_.get_coordinates_type()} coordinates;

    {self.class_name_}::chart(point1, point2, element, coordinates);
    {self.class_name_}::param(point1, point2, coordinates, element2);

    element.isApprox(element2, 1.0e-8);

    {self.class_name_}::chart_diff(point1, point1, element, d_chart);
    {self.class_name_}::param_diff(point1, point1, coordinates, d_param);

    Eigen::Matrix<double,
        {self.cls_.dimension},
        {self.cls_.dimension}>::Identity().isApprox(d_chart*d_param, 1.0e-8);


    {self.cls_.get_representation_type()} point1A =
                        {self.class_name_}::random_projection();
    {self.cls_.get_representation_type()} point2A =
                        {self.class_name_}::random_projection();

    {self.cls_.get_coordinates_type()} coordinates2;
    {self.cls_.get_coordinates_type()} coordinates3;
    {self.class_name_}::change_of_coordinates(point1, point2, point1A, point2A,
                            coordinates, coordinates2);
    {self.class_name_}::change_of_coordinates(point1A, point2A, point1, point2,
                            coordinates2, coordinates3);

    coordinates3.isApprox(coordinates, 1.0e-8);

    {self.cls_.change_of_coordinates_diff_type()} nomAdiff;
    {self.cls_.change_of_coordinates_diff_type()} Anomdiff;
    {self.class_name_}::change_of_coordinates_diff(point1, point2, point1A,
                                point2A,
                                coordinates, nomAdiff);
    {self.class_name_}::change_of_coordinates_diff(point1A, point2A, point1,
                                point2,
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
        self.test_generator.close()
