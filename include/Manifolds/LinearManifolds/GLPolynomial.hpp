#pragma once
#include <Manifolds/LinearManifolds/GLPolynomialDetail.hpp>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>

#include <Manifolds/Interval.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace manifolds {

template <std::size_t NumPoints>
class GLPolynomial
    : public LinearManifoldInheritanceHelper<GLPolynomial<NumPoints>,
                                             LinearManifold<NumPoints>>,
      public MapInheritanceHelper<GLPolynomial<NumPoints>,
                                  Map<CanonicInterval, Reals, false>> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

public:
  using base_t = LinearManifoldInheritanceHelper<GLPolynomial<NumPoints>,
                                                 LinearManifold<NumPoints>>;

  using base_t::base_t;

  using base_t::operator=;

  GLPolynomial(const GLPolynomial &) = default;
  GLPolynomial(GLPolynomial &&) = default;

  const GLPolynomial &operator=(const GLPolynomial &that) {
    base_t::operator=(that);
    return *this;
  }

  /// ----------------------------------------
  /// --------  Constants -------------------
  /// ----------------------------------------
  /// Gauss-lobatto points
  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  /// Gauss-lobatto weights
  static constexpr std::array<double, NumPoints> g_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  /// Barycentry weights
  static constexpr std::array<double, NumPoints> barycentric_weights =
      manifolds::collocation::detail::barycentric_weights<NumPoints>(
          manifolds::collocation::detail::compute_glp<NumPoints>());
  /// Derivative Matrix
  static Eigen::Matrix<double, NumPoints, NumPoints> derivative_matrix() {
    static Eigen::Matrix<double, NumPoints, NumPoints> result =
        Eigen::Map<Eigen::Matrix<double, NumPoints, NumPoints>>(
            collocation::detail::derivative_matrix<NumPoints>(gl_points)
                .data());
    return result;
  }
  /// Derivative Matrix of deg ,
  template <std::size_t M>
  static Eigen::Matrix<double, NumPoints, NumPoints>
  derivative_matrix_order_m() {
    static Eigen::Matrix<double, NumPoints, NumPoints> result =
        Eigen::Map<Eigen::Matrix<double, NumPoints, NumPoints>>(
            collocation::detail::derivative_matrix_order_m<NumPoints, M>(
                gl_points)
                .data());
    return result;
  }

  // static std::optional<std::array<double, NumPoints>>
  // barycentric_weights_ =
  // {}; static std::optional<Eigen::Matrix<double, NumPoints, NumPoints>>
  //     derivative_matrix_ = {};

  bool value_on_repr(const double &in, double &result) const override {

    result = evaluation(this->crepr(), in);
    return true;
  }

  /**
   * Evaluation of gauss-lobatto polynomial.
   *  David A. Kopriva
   *  Implementing Spectral
   *  Methods for Partial
   *  Differential Equations
   *  Algorithm 34: LagrangeInterpolatingPolynomials */
  static double evaluation(const Eigen::Vector<double, NumPoints> &vec,
                           double in) {
    static std::array<double, NumPoints> evaluation_buffer;
    double result;

    for (std::size_t i = 0; i < NumPoints; i++)
      if (std::fabs(gl_points[i] - in) < 1.0e-9) {
        return vec(i);
      }

    memset(evaluation_buffer.data(), 0.0, NumPoints);
    double t_book, s_book;

    s_book = 0.0;
    for (std::size_t j = 0; j < NumPoints; j++) {
      t_book = barycentric_weights[j] / (in - gl_points[j]);
      evaluation_buffer[j] = t_book;
      s_book += t_book;
    }
    for (std::size_t j = 0; j < NumPoints; j++)
      evaluation_buffer[j] /= s_book;

    result = 0.0;
    for (std::size_t j = 0; j < NumPoints; j++)
      result += vec(j) * evaluation_buffer[j];

    return result;
  }

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {

    _mat(0, 0) = evaluation_derivative(this->crepr(), _in);
    return true;
  }

  static double
  evaluation_derivative(const Eigen::Vector<double, NumPoints> &vec,
                        double _in) {

    /*  David A. Kopriva
     *  Implementing Spectral
     *  Methods for Partial
     *  Differential Equations
     *  Algorithm 36: mthOrderPolynomialDerivativeMatrix*/
    bool atNode = false;
    double numerator = 0.0;
    double denominator = 0.0;
    double p_book = 0.0;
    std::size_t i_book = 0;
    for (std::size_t j = 0; j < NumPoints; j++)
      if (std::fabs(gl_points[j] - _in) < 1.0e-9) {
        atNode = true;
        denominator = -barycentric_weights[j];
        i_book = j;
        p_book = vec(j);
      }

    if (atNode) {
      for (std::size_t j = 0; j < NumPoints; j++)
        if (i_book != j)
          numerator = numerator + barycentric_weights[j] * (p_book - vec(j)) /
                                      (_in - gl_points[j]);
    } else {
      denominator = 0.0;
      p_book = evaluation(vec, _in);
      for (std::size_t j = 0; j < NumPoints; j++) {
        double t = barycentric_weights[j] / (_in - gl_points[j]);
        numerator = numerator + t * (p_book - vec(j)) / (_in - gl_points[j]);
        denominator += t;
      }
    }
    return numerator / denominator;
  }

  template <std::size_t Deg> class Derivative;

  class Integral;

  static GLPolynomial<NumPoints> identity() {
    return GLPolynomial<NumPoints>(
        Eigen::Map<const Eigen::Matrix<double, NumPoints, 1>>(
            gl_points.data()));
  }

  static GLPolynomial<NumPoints> constant(double val) {
    return GLPolynomial<NumPoints>(
        Eigen::Map<Eigen::Matrix<double, NumPoints, 1>>(
            std::vector<double>(NumPoints, val).data()));
  }
};

template <std::size_t NumPoints, std::size_t CoDomainDim>
class GLVPolynomial
    : public LinearManifoldInheritanceHelper<
          GLPolynomial<NumPoints>, LinearManifold<NumPoints * CoDomainDim>>,
      public MapInheritanceHelper<
          GLPolynomial<NumPoints>,
          Map<CanonicInterval, LinearManifold<CoDomainDim>, false>> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

public:
  using base_t =
      LinearManifoldInheritanceHelper<GLPolynomial<NumPoints>,
                                      LinearManifold<NumPoints * CoDomainDim>>;

  using base_t::base_t;

  using base_t::operator=;

  using base_t::dimension;

  GLVPolynomial(const GLVPolynomial &) = default;
  GLVPolynomial(GLVPolynomial &&) = default;

  const GLVPolynomial &operator=(const GLVPolynomial &that) {
    base_t::operator=(that);
    return *this;
  }

  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  static constexpr std::array<double, NumPoints> g_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  static constexpr std::array<double, NumPoints> barycentric_weights =
      manifolds::collocation::detail::barycentric_weights<NumPoints>(
          manifolds::collocation::detail::compute_glp<NumPoints>());

  // static std::optional<std::array<double, NumPoints>> barycentric_weights_ =
  // {}; static std::optional<Eigen::Matrix<double, NumPoints, NumPoints>>
  //     derivative_matrix_ = {};

  bool value_on_repr(const double &in,
                     Eigen::Vector<double, CoDomainDim> &out) const override {

    for (std::size_t i = 0; i < CoDomainDim; i++) {
      out(i) = GLPolynomial<NumPoints>::evaluation(
          this->crerp().segment<NumPoints>(i * NumPoints), in);
    }
    return true;
  }

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {
    for (std::size_t i = 0; i < CoDomainDim; i++) {
      _mat(i, 0) =
          GLPolynomial<NumPoints>::evaluation_derivative(this->crerp(), _in);
    }
    return true;
  }

  template <std::size_t Deg> class Derivative;

  class Integral;

  static GLPolynomial<NumPoints> identity() {
    return GLPolynomial<NumPoints>(
        Eigen::Map<const Eigen::Matrix<double, NumPoints, 1>>(
            gl_points.data()));
  }

  static GLPolynomial<NumPoints> constant(double val) {
    return GLPolynomial<NumPoints>(
        Eigen::Map<Eigen::Matrix<double, NumPoints, 1>>(
            std::vector<double>(NumPoints, val).data()));
  }
};

/** Piece-wise Gauss Lobatto polynomial
 * NumPoints: number of gauss-lobato points
 * Intervals: number of intervals
 * DomainDim: Dimpension of the values of the pol
 * */
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial
    : public LinearManifoldInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          LinearManifold<NumPoints * CoDomainDim * Intervals>>,
      public MapInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          Map<ZeroToNInterval<Intervals + 1>, LinearManifold<CoDomainDim>,
              false>> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

  const std::pair<double, double> domain_interval_;

public:
  using interval_t = std::pair<double, double>;
  using base_t =
      LinearManifoldInheritanceHelper<GLPolynomial<NumPoints>,
                                      LinearManifold<NumPoints * CoDomainDim>>;

  using base_t::dimension;
  using base_t::operator=;

  struct point_value_t {
    long i0;
    long i1;
    Eigen::Map<Eigen::Vector<double, CoDomainDim>> value;
  };

  PWGLVPolynomial(const interval_t &domain);
  PWGLVPolynomial(const interval_t &_domain,
                  const Eigen::Vector<double, NumPoints> &val)
      : base_t(val), domain_interval_(_domain) {}

  PWGLVPolynomial(const PWGLVPolynomial &) = default;
  PWGLVPolynomial(PWGLVPolynomial &&) = default;

  const PWGLVPolynomial &operator=(const PWGLVPolynomial &that) {
    domain_interval_ = that.domain_interval_;
    base_t::operator=(that);
    return *this;
  }

  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  static constexpr std::array<double, NumPoints> g_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  static constexpr std::array<double, NumPoints> barycentric_weights =
      manifolds::collocation::detail::barycentric_weights<NumPoints>(
          manifolds::collocation::detail::compute_glp<NumPoints>());

  static std::pair<long, long> range_for_pint_index(std::size_t _point_index) {
    std::pair<long, long> result;
    if (_point_index > (Intervals + 1) * NumPoints)
      throw std::invalid_argument("");

    result.first = CoDomainDim * _point_index;
    result.second = CoDomainDim * (_point_index + 1);

    return result;
  }

  static double point_in_domain(const interval_t &_interval,
                                std::size_t _index) {
    if (_index > dimension)
      throw std::invalid_argument("");
  }
  static double element_index_to_point(const interval_t &_interval,
                                       std::size_t _index) {
    if (_index > dimension)
      throw std::invalid_argument("");
  }
  /// ---------------------------------------------------------
  /// --------------- Evaluation of the polynomial ------------
  /// ---------------------------------------------------------

  bool value_on_repr(const double &in,
                     Eigen::Vector<double, CoDomainDim> &out) const override {

    for (std::size_t interval = 0; interval < Intervals; interval++) {
      if (interval <= in <= interval + 1) {
        for (std::size_t i = 0; i < CoDomainDim; i++) {
          out(i) = GLPolynomial<NumPoints>::evaluation(
              this->crerp().segment<NumPoints>(
                  NumPoints * CoDomainDim * interval + i * NumPoints),
              in);
        }
        return true;
      }
    }
    return false;
  }

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {
    for (std::size_t interval = 0; interval < Intervals; interval++) {
      if (interval <= _in <= interval + 1) {
        for (std::size_t i = 0; i < CoDomainDim; i++) {
          _mat(i, 0) = GLPolynomial<NumPoints>::evaluation_derivative(
              this->crerp().segment<NumPoints>(
                  NumPoints * CoDomainDim * interval + i * NumPoints),
              _in);
        }
        return true;
      }
    }
    return false;
  }

  /// ---------------------------------------------------------
  /// --------------- Operators -------------------------------
  /// ---------------------------------------------------------
  template <std::size_t Deg> class Derivative;

  class Integral;

  class Composition;

  /// ---------------------------------------------------------
  /// --------------- Default Elements ------------
  /// ---------------------------------------------------------
  static GLPolynomial<NumPoints> identity() {
    return GLPolynomial<NumPoints>(
        Eigen::Map<const Eigen::Matrix<double, NumPoints, 1>>(
            gl_points.data()));
  }

  static GLPolynomial<NumPoints> constant(double val) {
    return GLPolynomial<NumPoints>(
        Eigen::Map<Eigen::Matrix<double, NumPoints, 1>>(
            std::vector<double>(NumPoints, val).data()));
  }
};

} // namespace manifolds
