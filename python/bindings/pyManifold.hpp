
#pragma once
#include <Manifolds/ManifoldBase.hpp>
#include <eigen3/Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace manifolds {

class PyManifoldBase : public ManifoldBase {
private:
void assign(const std::unique_ptr<ManifoldBase> &) override {
  // PYBIND11_OVERRIDE_PURE(void, PyManifoldBase, assign, _other);
}
virtual ManifoldBase *clone_impl() const override {
  PYBIND11_OVERRIDE_PURE(ManifoldBase *, PyManifoldBase, clone_impl);
}
virtual ManifoldBase *move_clone_impl() override {
  PYBIND11_OVERRIDE_PURE(ManifoldBase *, PyManifoldBase, move_clone_impl);
}

public:
virtual std::size_t get_dim() const override {
  PYBIND11_OVERRIDE_PURE(std::size_t, PyManifoldBase, get_dim);
}
virtual std::size_t get_tanget_repr_dim() const override {
  PYBIND11_OVERRIDE_PURE(std::size_t, PyManifoldBase, get_tanget_repr_dim);
}
virtual bool has_value() const override {
  PYBIND11_OVERRIDE_PURE(bool, PyManifoldBase, has_value);
}

  virtual bool is_equal(const std::unique_ptr<ManifoldBase> &) const  override
{
  PYBIND11_OVERRIDE_PURE(bool, PyManifoldBase, is_equal);
}
};

} // namespace manifolds
