#include "pyManifold.hpp"
#include "pyMap.hpp"
#include <Manifolds/Interval.hpp>
#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/GLPolynomialOperators.hpp>
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

// this avoid to register several times the map from reals to a vector space
// when registering several partitions and gauss-lobatto points
template <typename M>
void register_map_once(
    std::variant<std::reference_wrapper<py::detail::generic_type>,
                 std::reference_wrapper<py::module>>
        m,
    const std::string &name) {
  static bool firstCall = true;
  if (firstCall) {
    std::visit(
        [&](auto &&x) {
          py::class_<M, MapTrampoline<M>, MapBase>(x.get(), name.c_str())
              .def(py::init<>());
        },
        m);
  }
  firstCall = false;
}
// this avoid to register several times the map from reals to a vector space
// when registering several partitions and gauss-lobatto points
template <typename M>
void register_manifold_once(
    std::variant<std::reference_wrapper<py::detail::generic_type>,
                 std::reference_wrapper<py::module>>
        m,
    const std::string &name) {
  static bool firstCall = true;
  if (firstCall) {
    std::visit(
        [&](auto &&x) {
          py::class_<M, ManifoldBase>(x.get(), name.c_str()).def(py::init<>());
        },
        m);
  }
  firstCall = false;
}

template <typename M, typename Mod = py::module>
auto register_user_map(Mod &m, const std::string &name) {

  register_map_once<typename M::map_t>(m, name + "Map");

  py::class_<typename M::base_t, MapTrampoline<typename M::base_t>,
             typename M::map_t>(m, name + "Clonable")
      .def(py::init<>())
      .def("clone", &M::base_t::clone)
      .def("move_clone", &M::base_t::move_clone);

  py::class_<M, typename M::base_t> result(m, name.c_str());

  result.def("__call__", [](const M &_this, const typename M::domain_t &_in) {
    return _this(_in);
  });

  return result;
}

template <typename M, typename Mod = py::module>
auto register_user_linear_map(Mod &m, const std::string &name) {
  register_manifold_once<typename M::map_t::manifold_t>(m, name + "Manifold");
  register_map_once<typename M::map_t::map_t>(m, name + "Map");

  py::class_<typename M::map_t::base_t,
             MapTrampoline<typename M::map_t::base_t>, typename M::map_t::map_t,
             typename M::map_t::manifold_t>(m, (name + "ClonableA").c_str())
      .def(py::init<>())
      .def("clone", &M::map_t::base_t::clone)
      .def("move_clone", &M::map_t::base_t::move_clone);

  py::class_<typename M::map_t, typename M::map_t::base_t>(
      m, (name + "LinearMap").c_str())
      .def(py::init<>());

  py::class_<typename M::base_t, MapTrampoline<typename M::base_t>>(
      m, (name + "Clonable").c_str())
      .def(py::init<>())
      .def("clone", &M::base_t::clone)
      .def("move_clone", &M::base_t::move_clone);

  py::class_<M, typename M::base_t> result(m, name.c_str());

  result.def("__call__", [](const M &_this, const typename M::domain_t &_in) {
    return _this(_in);
  });
  return result;
}

template <typename M>
auto register_polynomial(py::module &m, const std::string &name) {
  // Read here how to define nested classes
  // https://pybind11.readthedocs.io/en/stable/classes.html?highlight=def_readwrite#instance-and-static-fields
  // Define the polynomial
  register_manifold_once<typename M::manifold_t>(m, name + "Manifold");
  register_map_once<typename M::map_t>(m, (name + "Map").c_str());
  py::class_<typename M::base_t, MapTrampoline<typename M::base_t>,
             typename M::map_t, typename M::manifold_t>(
      m, (name + "Clonable").c_str())
      .def(py::init<>())
      .def("clone", &M::base_t::clone)
      .def("move_clone", &M::base_t::move_clone);

  // Add Polynomial
  py::class_<M, typename M::base_t> result(m, name.c_str());
  result.def(py::init<double, double>())
      .def("get_domain", &M::get_domain)
      .def("__call__",
           [](const M &_this, double t) { return _this(t).crepr(); })
      .def("deriv",
           [](const M &_this) {
             const auto &dom = _this.get_domain();
             const IntervalPartition<10> ip(dom);
             return typename M::template Derivative<1>(ip)(_this);
           })
      .def_static("constant", &M::constant)
      .def_static("random", &M::constant)
      .def_static("zero", &M::zero);

  register_user_linear_map<typename M::template Derivative<1>>(result,
                                                               "Derivative")
      .def(py::init<const typename M::interval_partition_t &>());

  // Define the continuous version of the polynomial
  using C = typename M::template Continuous<2>;

  register_manifold_once<typename C::manifold_t>(result,
                                                 "C" + name + "Manifold");
  register_map_once<typename C::map_t>(result, ("C" + name + "Map").c_str());
  py::class_<typename C::base_t, MapTrampoline<typename C::base_t>,
             typename C::map_t, typename C::manifold_t>(
      result, ("C" + name + "Clonable").c_str())
      .def(py::init<>())
      .def("clone", &C::base_t::clone)
      .def("move_clone", &C::base_t::move_clone);

  py::class_<C, typename C::base_t> continuous(result, ("C" + name).c_str());
  continuous.def(py::init<double, double>()).def_static("random", &C::random);

  // Define the inclusion of continuous polynomials

  register_user_linear_map<typename C::Inclusion>(continuous, "Inclusion")
      .def(py::init<const typename M::interval_partition_t &>());
  /// Register functions

  py::class_<typename M::functions> functions(result, "functions");
  functions.def(py::init<>());

  return result;
}

template <int... Is>
void register_linear_manifold(py::module &m,
                              std::integer_sequence<int, Is...>) {
  (py::class_<DenseLinearManifold<Is>, ManifoldBase>(
       m, ("R" + std::to_string(Is)).c_str())
       .def(py::init<>())
       .def("crepr", &DenseLinearManifold<Is>::crepr),
   ...);
}

template <int NumberOfPoints, int Intervals, int... Is>
void register_pwglvpolynomial(py::module &m,
                              std::integer_sequence<int, Is...>) {
  (py::class_<PWGLVPolynomial<NumberOfPoints, Intervals, Is>,
              DenseLinearManifold<Is * Intervals * NumberOfPoints>, MapBase>(
       m, ("PWGLVPolynomial_" + std::to_string(NumberOfPoints) + "_" +
           std::to_string(Intervals) + "_" + std::to_string(Is))
              .c_str())
       .def(
           py::init<const typename PWGLVPolynomial<8, 10, Is>::interval_t &>()),
   ...);
}

template <int NumberOfPoints, int... IsIntervals, int... Is>
void register_pwglvpolynomial_2(py::module &m,
                                std::integer_sequence<int, IsIntervals...>,
                                std::integer_sequence<int, Is...>) {
  (register_pwglvpolynomial<NumberOfPoints, IsIntervals>(
       m, std::integer_sequence<int, Is...>()),
   ...);
}

template <int... IsNumberOfPoints, int... IsIntervals, int... Is>
void register_pwglvpolynomial_3(py::module &m,
                                std::integer_sequence<int, IsNumberOfPoints...>,
                                std::integer_sequence<int, IsIntervals...>,
                                std::integer_sequence<int, Is...>) {
  (register_pwglvpolynomial_2<IsNumberOfPoints>(
       m, std::integer_sequence<int, IsIntervals...>(),
       std::integer_sequence<int, Is...>()),
   ...);
}

PYBIND11_MODULE(pymanifolds, manifolds_module) {

  py::class_<ManifoldBase, PyManifoldBase>(manifolds_module, "Manifold")
      .def(py::init<>())
      .def("clone", &ManifoldBase::clone)
      .def("move_clone", &ManifoldBase::move_clone)
      .def("get_dim", &ManifoldBase::get_dim)
      .def("get_tanget_repr_dim", &ManifoldBase::get_tanget_repr_dim);

  py::class_<MapBase, PyMapBase>(manifolds_module, "Map")
      .def(py::init<>())
      .def("clone", &MapBase::clone)
      .def("move_clone", &MapBase::move_clone)
      .def("get_dom_dim", &MapBase::get_dom_dim)
      .def("get_codom_dim", &MapBase::get_codom_dim)
      .def("value",
           [](const MapBase &_this, const ManifoldBase &_that) {
             return _this.value(_that);
           })
      .def("diff", [](const MapBase &_this, const ManifoldBase &_that) {
        return _this.diff(_that);
      });

  py::class_<MapBaseComposition, MapBase>(manifolds_module, "MapComposition")
      .def(py::init<const MapBase &>());

  py::module linear_manifolds =
      manifolds_module.def_submodule("linear_manifolds");

  register_linear_manifold(linear_manifolds,
                           std::make_integer_sequence<int, 7>());

  py::class_<Interval, ManifoldBase>(manifolds_module, "Interval")
      .def(py::init<double, double>())
      .def("length", &Interval::length)
      .def("first", &Interval::length)
      .def("second", &Interval::second)
      .def("contains", &Interval::contains)
      .def("as_tuple", &Interval::as_tuple)
      .def("get_random", &Interval::get_random)
      .def("sized_linspace", &Interval::sized_linspace)
      .def("spaced_linspace", &Interval::spaced_linspace);

  py::class_<IntervalPartition<10>, ManifoldBase>(manifolds_module,
                                                  "IntervalPartition")
      .def(py::init<double, double>());
  using Pol = PWGLVPolynomial<10, 10, 7>;
  register_polynomial<Pol>(linear_manifolds, "GLPolynomial");
  // register_polynomial<PWGLVPolynomial<10, 10, 3>>(linear_manifolds,
  //                                                 "GLPolynomial_3");

  // auto p = register_polynomial<PWGLVPolynomial<10, 10, 1>>(linear_manifolds,
  //                                                          "GLPolynomial_1");

  // register_user_linear_map<typename PWGLVPolynomial<10, 10, 1>::Integral>(
  //     p, "Integral");
}
