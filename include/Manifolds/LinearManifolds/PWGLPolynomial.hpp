#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
// #include <Manifolds/LinearManifolds/PWGLPolynomialDetail.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace manifolds {
/*
template <std::size_t NumPoints, std::size_t DomainDim>
class PWGLPolynomial
    : public ManifoldInheritanceHelper<PWGLPolynomial<NumPoints, DomainDim>,
                                       LinearManifold<NumPoints * DomainDim>>,
      public MapInheritanceHelper<PWGLPolynomial<NumPoints, DomainDim>,
                                  Map<Reals, LinearManifold<DomainDim>>> {

public:
  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  static constexpr std::array<double, NumPoints> g_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  // static std::optional<std::array<double, NumPoints>> barycentric_weights_ =
  // {}; static std::optional<Eigen::Matrix<double, NumPoints, NumPoints>>
  //     derivative_matrix_ = {};

  bool value_on_repr(const double &,
                     Eigen::Matrix<double, DomainDim, 1> &) const override {
    int j;
    const auto &c = this->crepr();
    double p = c[j = this->crepr().size() - 1];
    while (j > 0)
      p = p * _in + c[--j];
    _result = p;
    return true;
  }
  virtual bool
  diff_from_repr(const double &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    _mat(0, 0) = _in;
    return true;
  }

  template <std::size_t N>
  static constexpr std::tuple<double, double, double>
  q_and_evaluation(double _t);

  static constexpr std::array<double, NumPoints>
  legendre_gauss_lobatto_points();

private:
  std::tuple<double, double> domain_;
};
*/
} // namespace manifolds
