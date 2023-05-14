#pragma once
#include <Manifolds/Maps/MapBase.hpp>
#include <eigen3/Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace manifolds {

class PyMapBase : public MapBase {
private:
  virtual MapBase *clone_impl() const override {
    PYBIND11_OVERRIDE_PURE(MapBase *, PyMapBase, clone_impl);
  }
  virtual MapBase *move_clone_impl() override {
    PYBIND11_OVERRIDE_PURE(MapBase *, PyMapBase, move_clone_impl);
  }

  virtual ManifoldBase *domain_buffer_impl() const override {
    PYBIND11_OVERRIDE_PURE(ManifoldBase *, PyMapBase, domain_buffer_impl);
  }
  virtual ManifoldBase *codomain_buffer_impl() const override {
    PYBIND11_OVERRIDE_PURE(ManifoldBase *, PyMapBase, codomain_buffer_impl);
  }

public:
  virtual std::size_t get_dom_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, PyMapBase, get_dom_dim);
  }
  virtual std::size_t get_codom_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, PyMapBase, get_codom_dim);
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, PyMapBase, get_dom_tangent_repr_dim);
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    PYBIND11_OVERRIDE_PURE(std::size_t, PyMapBase, get_codom_tangent_repr_dim);
  }
  virtual DifferentialReprType linearization_buffer() const override {
    PYBIND11_OVERRIDE_PURE(DifferentialReprType, PyMapBase,
                           linearization_buffer);
  }
  virtual bool is_differential_sparse() const override {
    PYBIND11_OVERRIDE_PURE(bool, PyMapBase, is_differential_sparse);
  }
};

} // namespace manifolds
