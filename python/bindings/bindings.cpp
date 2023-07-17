/*#include "pyManifold.hpp" */
/*#include "pyMap.hpp" */
/*#include <Manifolds/LinearManifolds/GLPolynomial.hpp> */
/*#include <Manifolds/LinearManifolds/LinearManifolds.hpp> */
/*#include <Manifolds/LinearManifolds/LinearMaps.hpp> */
/*#include <Manifolds/LinearManifolds/Reals.hpp> */
/*#include <Manifolds/Manifold.hpp> */
/*#include <Manifolds/Maps/Map.hpp> */
/*#include <Manifolds/Maps/MapComposition.hpp> */
/*#include <pybind11/eigen.h> */
/*#include <pybind11/numpy.h> */
/*#include <pybind11/pybind11.h> */
/*#include <pybind11/pytypes.h> */

/*namespace py = pybind11; */

/*using namespace manifolds; */

/*template <int... Is> */
/*void register_linear_manifold(py::module &m, */
/*                              std::integer_sequence<int, Is...>) { */
/*  (py::class_<LinearManifold<Is>, ManifoldBase>( */
/*       m, ("R" + std::to_string(Is)).c_str()) */
/*       .def(py::init<>()) */
/*       .def("crepr", &LinearManifold<Is>::crepr) */
/*       .def("__add__", */
/*            [](const LinearManifold<Is> &self, */
/*               const LinearManifold<Is> &other) { return self + other; }) */
/*       .def("__mul__", [](const LinearManifold<Is> &self, */
/*                          const double &other) { return self * other; }), */
/*   ...); */
/*} */

/*template <int NumberOfPoints, int Intervals, int... Is> */
/*void register_pwglvpolynomial(py::module &m, */
/*                              std::integer_sequence<int, Is...>) { */
/*  (py::class_<PWGLVPolynomial<NumberOfPoints, Intervals, Is>, */
/*              LinearManifold<Is * Intervals * NumberOfPoints>, MapBase>( */
/*       m, ("PWGLVPolynomial_" + std::to_string(NumberOfPoints) + "_" + */
/*           std::to_string(Intervals) + "_" + std::to_string(Is)) */
/*              .c_str()) */
/*       .def( */
/*           py::init<const typename PWGLVPolynomial<8, 10, Is>::interval_t
 * &>()), */
/*   ...); */
/*} */
/*template <int NumberOfPoints, int... IsIntervals, int... Is> */
/*void register_pwglvpolynomial_2(py::module &m, */
/*                                std::integer_sequence<int, IsIntervals...>, */
/*                                std::integer_sequence<int, Is...>) { */
/*  (register_pwglvpolynomial<NumberOfPoints, IsIntervals>( */
/*       m, std::integer_sequence<int, Is...>()), */
/*   ...); */
/*} */

/*template <int... IsNumberOfPoints, int... IsIntervals, int... Is> */
/*void register_pwglvpolynomial_3(py::module &m, */
/*                                std::integer_sequence<int,
 * IsNumberOfPoints...>, */
/*                                std::integer_sequence<int, IsIntervals...>, */
/*                                std::integer_sequence<int, Is...>) { */
/*  (register_pwglvpolynomial_2<IsNumberOfPoints>( */
/*       m, std::integer_sequence<int, IsIntervals...>(), */
/*       std::integer_sequence<int, Is...>()), */
/*   ...); */
/*} */

/*PYBIND11_MODULE(pymanifolds, manifolds_module) { */

/*  py::class_<ManifoldBase, PyManifoldBase>(manifolds_module, "Manifold") */
/*      .def(py::init<>()) */
/*      .def("clone", &ManifoldBase::clone) */
/*      .def("move_clone", &ManifoldBase::move_clone) */
/*      .def("get_dim", &ManifoldBase::get_dim) */
/*      .def("get_tanget_repr_dim", &ManifoldBase::get_tanget_repr_dim); */

/*  /1* */
/*  py::class_<MapBase, PyMapBase>(manifolds_module, "Map") */
/*      .def(py::init<>()) */
/*      .def("clone", &MapBase::clone) */
/*      .def("move_clone", &MapBase::move_clone) */
/*      .def("get_dom_dim", &MapBase::get_dom_dim) */
/*      .def("get_codom_dim", &MapBase::get_codom_dim); */

/*  py::module linear_manifolds = */
/*      manifolds_module.def_submodule("linear_manifolds"); */

/*  register_linear_manifold(linear_manifolds, */
/*                           std::make_integer_sequence<int, 20>()); */

/*  register_pwglvpolynomial_3(linear_manifolds, */
/*                             std::make_integer_sequence<int, 10>(), */
/*                             std::make_integer_sequence<int, 30>(), */
/*                             std::make_integer_sequence<int, 20>()); */
/*                             *1/ */
/*} */
