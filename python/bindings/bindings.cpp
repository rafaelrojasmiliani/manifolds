#include "pyManifold.hpp"
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <Manifolds/Maps/MapComposition.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

using namespace manifolds;

template <int... Is>
void register_linear_manifold(py::module &m,
                              std::integer_sequence<int, Is...>) {
  (py::class_<LinearManifold<Is>, ManifoldBase>(
       m, ("R" + std::to_string(Is)).c_str())
       .def(py::init<>())
       .def("crepr", &LinearManifold<Is>::crepr)
       .def("__add__",
            [](const LinearManifold<Is> &self,
               const LinearManifold<Is> &other) { return self + other; })
       .def("__mul__", [](const LinearManifold<Is> &self,
                          const double &other) { return self * other; }),
   ...);
}
PYBIND11_MODULE(pymanifolds, manifolds_module) {

  py::class_<ManifoldBase, PyManifoldBase>(manifolds_module, "ManifoldBase")
      .def(py::init<>())
      .def("clone", &ManifoldBase::clone)
      .def("move_clone", &ManifoldBase::move_clone)
      .def("get_dim", &ManifoldBase::get_dim)
      .def("get_tanget_repr_dim", &ManifoldBase::get_tanget_repr_dim);

  py::module linear_manifolds =
      manifolds_module.def_submodule("linear_manifolds");

  register_linear_manifold(linear_manifolds,
                           std::make_integer_sequence<int, 20>());
}
