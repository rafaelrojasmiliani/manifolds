#pragma once

#include <Manifolds/LinearManifolds.hpp>
#include <Manifolds/Map.hpp>
namespace manifolds {

template <typename Manifold>
class Chart : public Map<Manifold, MatrixManifold<Manifold::dim, 1>> {

public:
  Chart() = default;
  Chart(const Chart &) = default;
  Chart(Chart &&) = default;

  //  MatrixManifold<Manifold::dim, 1>
  //  operator()(const Manifold &_in) const override {
  //    return MatrixManifold<Manifold::dim, 1>(value_impl(_in));
  //  }
};

template <typename Manifold>
class Parametrization : public Map<MatrixManifold<Manifold::dim, 1>, Manifold> {

public:
  Parametrization() = default;
  Parametrization(const Parametrization &) = default;
  Parametrization(Parametrization &&) = default;
  //  Manifold
  //  operator()(const MatrixManifold<Manifold::dim, 1> &_in) const override {
  //    return value_impl(_in.repr());
  //  }
  //
  //  virtual Manifold value_impl(const Eigen::Matrix<double, 2, 1> &_in) const
  //  = 0;
};
} // namespace manifolds
