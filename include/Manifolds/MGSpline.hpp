#pragma once
#include <Manifolds/Charts.hpp>
#include <Manifolds/Map.hpp>
#include <eigen3/Eigen/Core>

#include <gsplines/Basis/Basis.hpp>
#include <gsplines/Collocation/GaussLobattoLagrange.hpp>
#include <memory>
#include <vector>

namespace manifolds {

template <typename M> class IntervalData {
public:
  IntervalData(double t0, double t1, const gsplines::basis::Basis &_base,
               const Eigen::VectorXd &coeff, const Parametrization<M> &_param);

private:
  const double t0_;
  const double t1_;
  double scal(const double &_t) {
    if (_t < t0_)
      return t0_;
    if (_t > t1_)
      return t1_;
    return (_t - t0_) / (t1_ - t0_) + t0_;
  }
};

template <typename M> class MGSpline {

protected:
  Eigen::VectorXd coefficients_;
  Eigen::VectorXd domain_interval_lengths_;

  std::unique_ptr<gsplines::basis::Basis> basis_;

  std::vector<typename M::Parametrization> parametrizations_;
};
} // namespace manifolds
