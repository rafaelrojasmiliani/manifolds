#pragma once
#include <Manifolds/Maps/Map.hpp>
#include <Manifolds/Maps/MapBase.hpp>
#include <Manifolds/Maps/MapBaseComposition.hpp>
#include <eigen3/Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace manifolds {

class PyMapBase : public MapBase {
private:
  using MapBase::MapBase;
  virtual MapBase *clone_impl() const override {
    PYBIND11_OVERRIDE_PURE(MapBase *, MapBase, clone_impl);
  }
  virtual MapBase *move_clone_impl() override {
    PYBIND11_OVERRIDE_PURE(MapBase *, MapBase, move_clone_impl);
  }

  virtual ManifoldBase *domain_buffer_impl() const override {
    PYBIND11_OVERRIDE_PURE(ManifoldBase *, MapBase, domain_buffer_impl);
  }
  virtual ManifoldBase *codomain_buffer_impl() const override {
    PYBIND11_OVERRIDE_PURE(ManifoldBase *, MapBase, codomain_buffer_impl);
  }

public:
  virtual std::size_t get_dom_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, MapBase, get_dom_dim);
  }
  virtual std::size_t get_codom_dim() const override {
    std::cout << "\n----  here -wowowoowo\n\n;";
    PYBIND11_OVERRIDE_PURE(std::size_t, MapBase, get_codom_dim);
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, MapBase, get_dom_tangent_repr_dim);
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, MapBase, get_codom_tangent_repr_dim);
  }
  virtual detail::mixed_matrix_t linearization_buffer() const override {
    PYBIND11_OVERRIDE_PURE(detail::mixed_matrix_t, MapBase,
                           linearization_buffer);
  }
  virtual detail::MatrixTypeId differential_type() const override {
    PYBIND11_OVERRIDE_PURE(detail::MatrixTypeId, MapBase, differential_type);
  }
  virtual bool value_impl(const ManifoldBase *, ManifoldBase *) const override {

    PYBIND11_OVERRIDE_PURE(bool, MapBase, value_impl);
  }

  virtual bool diff_impl(const ManifoldBase *, ManifoldBase *,
                         detail::mixed_matrix_ref_t) const override {

    PYBIND11_OVERRIDE_PURE(bool, MapBase, diff_impl);
  }
  virtual MapBase *pipe_impl(const MapBase &that) const override {
    PYBIND11_OVERRIDE_PURE(MapBase *, MapBase, pipe_impl, that);
  }
  virtual MapBase *pipe_move_impl(MapBase &&that) const override {

    PYBIND11_OVERRIDE_PURE(MapBase *, MapBase, pipe_impl, that);
  }
};

template <typename M> class MapTrampoline : public M {
public:
  virtual bool
  value_on_repr(const typename M::domain_facade_t &_in,
                typename M::codomain_facade_t &_result) const override {
    PYBIND11_OVERRIDE_PURE(bool, MapTrampoline, value_on_repr, _in, _result);
  }

  /// Function to copmute the result of the map differential using just the
  /// representation types
  virtual bool
  diff_from_repr(const typename M::domain_facade_t &_in,
                 typename M::codomain_facade_t &_out,
                 typename M::differential_ref_t _mat) const override {

    PYBIND11_OVERRIDE_PURE(bool, MapTrampoline, diff_from_repr, _in, _out,
                           _mat);
  }

  virtual M *clone_impl() const override {
    PYBIND11_OVERRIDE_PURE(M *, MapTrampoline, clone_impl);
  }
  virtual M *move_clone_impl() override {
    PYBIND11_OVERRIDE_PURE(M *, MapTrampoline, clone_impl);
  }
};

} // namespace manifolds
