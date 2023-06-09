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

#include <Eigen/Dense>

namespace manifolds {

/** Piece-wise Gauss Lobatto polynomial
 * NumPoints: number of gauss-lobato points
 * Intervals: number of intervals
 * DomainDim: Dimpension of the values of the pol
 * */
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial
    : public LinearManifoldInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          DenseLinearManifold<NumPoints * CoDomainDim * Intervals>>,
      public MapInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          Map<Reals, DenseLinearManifold<CoDomainDim>>, MatrixTypeId::Dense> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

  IntervalPartition<Intervals> domain_partition_;

public:
  // ----------------------------
  // ------ Types ---------------
  // ----------------------------
  using interval_t = std::pair<double, double>;
  using base_t = LinearManifoldInheritanceHelper<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
      DenseLinearManifold<NumPoints * CoDomainDim * Intervals>>;

  using base_t::dimension;
  using base_t::operator=;

  struct point_value_t {
    long i0;
    long i1;
    Eigen::Map<Eigen::Vector<double, CoDomainDim>> value;
  };

  class iterator;

  // ----------------------------
  // ------ Constants ---------------
  // ----------------------------
  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  static constexpr std::array<double, NumPoints> g_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  static constexpr std::array<double, NumPoints> barycentric_weights =
      manifolds::collocation::detail::barycentric_weights<NumPoints>(
          manifolds::collocation::detail::compute_glp<NumPoints>());

  // ----------------------------
  // ------ Live Cycle ---------------
  // ----------------------------
  PWGLVPolynomial() : base_t(), domain_partition_() {}

  PWGLVPolynomial(const Interval &_domain)
      : base_t(), domain_partition_(_domain) {}

  PWGLVPolynomial(const Interval &_domain,
                  const Eigen::Vector<double, dimension> &val)
      : base_t(val), domain_partition_(_domain) {}

  PWGLVPolynomial(const PWGLVPolynomial &) = default;
  PWGLVPolynomial(PWGLVPolynomial &&) = default;

  const PWGLVPolynomial &operator=(const PWGLVPolynomial &that) {
    domain_partition_ = that.domain_partition_;
    base_t::operator=(that);
    return *this;
  }

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

    auto [interval, s] =
        this->domain_partition_.subinterval_index_and_canonic_value(in);
    for (std::size_t i = 0; i < CoDomainDim; i++) {

      Eigen::Ref<const Eigen::Vector<double, CoDomainDim * NumPoints>> aux_vec =
          this->crepr().template segment<NumPoints * CoDomainDim>(
              NumPoints * CoDomainDim * interval);

      out(i) =
          evaluation(aux_vec(Eigen::seqN(0, NumPoints, CoDomainDim), 0), s);
    }
    return true;
  }

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {

    auto [interval, s] =
        this->domain_partition_.subinterval_index_and_canonic_value(_in);
    for (std::size_t i = 0; i < CoDomainDim; i++) {

      Eigen::Ref<const Eigen::Vector<double, CoDomainDim * NumPoints>> aux_vec =
          this->crepr().template segment<NumPoints * CoDomainDim>(
              NumPoints * CoDomainDim * interval);

      _mat(i, 0) = evaluation_derivative(
                       aux_vec(Eigen::seqN(0, NumPoints, CoDomainDim), 0), s) *
                   2.0 / domain_partition_.subinterval_length(interval);
    }
    return true;
  }

  /// ---------------------------------------------------------
  /// --------------- Operators -------------------------------
  /// ---------------------------------------------------------
  template <std::size_t Deg> class Derivative;

  class Integral;

  template <typename CoDomain> class Composition;

  /// ---------------------------------------------------------
  /// --------------- Polynomial evaluation -------------------
  /// ---------------------------------------------------------

  /*  David A. Kopriva
   *  Implementing Spectral
   *  Methods for Partial
   *  Differential Equations
   *  Algorithm 36: mthOrderPolynomialDerivativeMatrix*/
  static double
  evaluation_derivative(Eigen::Ref<const Eigen::Vector<double, NumPoints>> vec,
                        double _in) {

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

    memset(evaluation_buffer.data(), 0.0, NumPoints * sizeof(double));
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
  /// ---------------------------------------------------------
  /// --------------- Getters ---------------------------------
  /// ---------------------------------------------------------

  const Interval &get_domain() const { return domain_partition_.interval(); }

  Eigen::MatrixXd &points() {
    return this->repr().reshaped(NumPoints * Intervals, CoDomainDim);
  }

  Eigen::Ref<Eigen::Vector<double, CoDomainDim>>
  operator[](const std::size_t i) {
    return this->repr().segment(i * CoDomainDim, CoDomainDim);
  }

  Eigen::Ref<const Eigen::Vector<double, CoDomainDim>>
  operator[](const std::size_t i) const {
    return this->crepr().segment(i * CoDomainDim, CoDomainDim);
  }

  Eigen::Ref<Eigen::Vector<double, CoDomainDim>>
  codomain_point(const std::size_t i) {
    return this->repr().segment(i * CoDomainDim, CoDomainDim);
  }

  Eigen::Ref<const Eigen::Vector<double, CoDomainDim>>
  codomain_point(const std::size_t i) const {
    return this->crepr().segment(i * CoDomainDim, CoDomainDim);
  }

  double domain_point(std::size_t idx) const {
    const double &glp = this->gl_points[idx % NumPoints];
    std::size_t interval = idx / NumPoints;

    return domain_partition_.interval().first() +
           domain_partition_.crepr().array().segment(0, interval).sum() +
           (glp + 1) / 2.0 * domain_partition_.subinterval_length(interval);
  }

  std::size_t size() const { return NumPoints * Intervals; }

  // ---------------------------------------
  // ------ Default Elements ---------------
  // ---------------------------------------

  static PWGLVPolynomial
  constant(const Interval &_interval,
           const Eigen::Vector<double, CoDomainDim> &_vals) {

    PWGLVPolynomial result(_interval);

    for (std::size_t i = 0; i < result.size(); i++) {
      result[i] = _vals;
    }

    return result;
  }

  template <bool F = CoDomainDim == 1>
  static std::enable_if_t<F, PWGLVPolynomial>
  identity(const Interval &_interval) {

    PWGLVPolynomial result(_interval);

    Eigen::Matrix<double, 100, 1> mat;
    for (std::size_t i = 0; i < result.size(); i++) {
      result[i](0) = result.domain_point(i);
    }

    return result;
  }
};

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomialSpace {
private:
  Interval interval_;

public:
  PWGLVPolynomialSpace(const Interval &_interval) : interval_(_interval) {}

  PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> polynomial() {
    return PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>(interval_);
  }
  template <std::size_t Deg>
  typename PWGLVPolynomial<NumPoints, Intervals,
                           CoDomainDim>::template Derivative<Deg>
  derivative();
};

} // namespace manifolds
