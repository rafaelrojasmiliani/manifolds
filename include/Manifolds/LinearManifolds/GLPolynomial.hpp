#pragma once
#include <Manifolds/LinearManifolds/GLPolynomialDetail.hpp>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>

#include <Manifolds/Interval.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <array>
#include <cstddef>
#include <functional>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

namespace manifolds {

template <long Rows>
Eigen::Vector<double, Rows> willson_barrent(detail::sparse_matrix_ref_t _mat,
                                            Eigen::Vector<double, Rows> &_vec,
                                            double mu, double tol = 1.e-9,
                                            std::size_t max_iter = 100) {
  /* Taken from Joseph Kuo et al.  Computing a projection operator onto the null
   * space of a linear imaging operator: tutorial.
   * Algorithm 1
   * This algorithm requires the computation of the pinv of _mat, we overcome
   * this problem by computing the QR solution *I belive that this minimizes the
   * norm*
   * */
  static Eigen::Vector<double, Rows> f_prev;
  static Eigen::Vector<double, Rows> f_next;

  Eigen::Vector<double, Rows> r_prev(_mat.get().rows());
  Eigen::Vector<double, Rows> r_next(_mat.get().rows());

  double alpha = 2.0 / mu;

  f_prev = _vec;
  r_prev = _mat.get() * _vec;
  Eigen::SparseQR<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                  Eigen::COLAMDOrdering<int>>
      solver;
  solver.compute(_mat.get());
  for (std::size_t i = 0; i < max_iter; i++) {
    f_next.noalias() = f_prev - alpha * solver.solve(r_prev);
    r_next.noalias() = _mat.get() * f_next;
    if (r_next.norm() < tol)
      break;
  }
  return f_next;
}

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim,
          std::size_t ContDeg, bool FixedStartPoint, bool FixedEndPoint>
class CPWGLVPolynomial;

/** Piece-wise Gauss Lobatto polynomial
 * NumPoints: number of gauss-lobato points
 * Intervals: number of intervals
 * DomainDim: Dimpension of the values of the pol
 * */
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial
    : public detail::Clonable<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          DenseLinearManifold<NumPoints * CoDomainDim * Intervals, false>,
          Map<Reals, DenseLinearManifold<CoDomainDim>>> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

public:
  IntervalPartition<Intervals> domain_partition_;
  // ----------------------------
  // ------ Types ---------------
  // ----------------------------
  template <std::size_t Deg, bool FixedStartPoint = false,
            bool FixedEndPoint = false>
  using Continuous = CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, Deg,
                                      FixedStartPoint, FixedEndPoint>;

  using interval_t = std::pair<double, double>;
  using base_t = detail::Clonable<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
      DenseLinearManifold<NumPoints * CoDomainDim * Intervals, false>,
      Map<Reals, DenseLinearManifold<CoDomainDim>>>;

  using map_t = Map<Reals, DenseLinearManifold<CoDomainDim>>;

  using interval_partition_t = IntervalPartition<Intervals>;

  using manifold_t =
      DenseLinearManifold<NumPoints * CoDomainDim * Intervals, false>;

  using base_t::base_t;
  using base_t::dimension;
  using base_t::operator=;

  struct point_value_t {
    long i0;
    long i1;
    Eigen::Map<Eigen::Vector<double, CoDomainDim>> value;
  };

  // ----------------------------
  // ------ Constants ---------------
  // ----------------------------
  static constexpr std::array<double, NumPoints> gl_points =
      manifolds::collocation::detail::compute_glp<NumPoints>();
  static constexpr std::array<double, NumPoints> gl_weights =
      manifolds::collocation::detail::compute_glw<NumPoints>();
  static constexpr std::array<double, NumPoints> barycentric_weights =
      manifolds::collocation::detail::barycentric_weights<NumPoints>(
          manifolds::collocation::detail::compute_glp<NumPoints>());

  // ----------------------------
  // ------ Live Cycle ---------------
  // ----------------------------

  PWGLVPolynomial() : base_t(), domain_partition_() {

    printf("QQdimension of PWGLVPolynomial %li ----------------\n",
           PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::dimension);
  }

  PWGLVPolynomial(double a, double b)
      : base_t(), domain_partition_(Interval(a, b)) {

    printf("QQdimension of PWGLVPolynomial %li ----------------\n",
           PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::dimension);
  }

  PWGLVPolynomial(const IntervalPartition<Intervals> &_domain)
      : base_t(), domain_partition_(_domain) {}

  PWGLVPolynomial(const Interval &_domain,
                  const Eigen::Vector<double, dimension> &val)
      : base_t(val), domain_partition_(_domain) {}

  PWGLVPolynomial(const PWGLVPolynomial &) = default;
  PWGLVPolynomial(PWGLVPolynomial &&) = default;

  PWGLVPolynomial &operator=(const PWGLVPolynomial &_in) = default;
  PWGLVPolynomial &operator=(PWGLVPolynomial &&_in) = default;

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
          evaluation(aux_vec(Eigen::seqN(i, NumPoints, CoDomainDim), 0), s);
    }
    return true;
  }

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Vector<double, CoDomainDim> &out,
                              detail::dense_matrix_ref_t _mat) const override {

    value_on_repr(_in, out);

    auto [interval, s] =
        this->domain_partition_.subinterval_index_and_canonic_value(_in);
    for (std::size_t i = 0; i < CoDomainDim; i++) {

      Eigen::Ref<const Eigen::Vector<double, CoDomainDim * NumPoints>> aux_vec =
          this->crepr().template segment<NumPoints * CoDomainDim>(
              NumPoints * CoDomainDim * interval);

      _mat(i, 0) = evaluation_derivative(
                       aux_vec(Eigen::seqN(i, NumPoints, CoDomainDim), 0), s) *
                   2.0 / domain_partition_.subinterval_length(interval);
    }
    return true;
  }

  /// ---------------------------------------------------------
  /// --------------- Operators -------------------------------
  /// ---------------------------------------------------------
  template <std::size_t Deg> class Derivative;
  class Minus;

  class ToVector;
  class FromVector;

  class Integral;

  template <typename CoDomain> class Composition;

  class functions;

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

  Eigen::MatrixXd points() {
    return this->crepr().template reshaped<Eigen::RowMajor>(
        NumPoints * Intervals, CoDomainDim);
  }

  Eigen::Ref<Eigen::Vector<double, CoDomainDim>>
  operator[](const std::size_t i) {
    return this->repr().template segment<CoDomainDim>(i * CoDomainDim);
  }

  Eigen::Ref<const Eigen::Vector<double, CoDomainDim>>
  operator[](const std::size_t i) const {
    return this->crepr().template segment<CoDomainDim>(i * CoDomainDim);
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
           (glp + 1.0) / 2.0 * domain_partition_.subinterval_length(interval);
  }

  std::size_t size() const { return NumPoints * Intervals; }

  // ---------------------------------------
  // ------ Default Elements ---------------
  // ---------------------------------------

  static PWGLVPolynomial
  constant(const IntervalPartition<Intervals> &_interval,
           const Eigen::Vector<double, CoDomainDim> &_vals) {

    printf("Constant dimension of PWGLVPolynomial %li ----------------\n",
           PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::dimension);
    PWGLVPolynomial result;
    result.domain_partition_ = IntervalPartition<Intervals>(_interval);

    for (std::size_t i = 0; i < result.size(); i++) {
      result[i] = _vals;
    }

    return result;
  }

  static PWGLVPolynomial zero(const IntervalPartition<Intervals> &_interval) {

    PWGLVPolynomial result;

    result = from_repr(_interval, PWGLVPolynomial::Representation::Zero());

    return result;
  }

  static PWGLVPolynomial random(const IntervalPartition<Intervals> &_interval) {

    return from_repr(_interval, PWGLVPolynomial::Representation::Random());
  }

  static PWGLVPolynomial
  random_continuous(const IntervalPartition<Intervals> &_interval) {

    return from_repr(_interval, PWGLVPolynomial::Representation::Random());
    return Continuous<2>::Inclusion(_interval)(
        Continuous<2>::random(_interval));
  }

  static PWGLVPolynomial
  straight_p2p(const IntervalPartition<Intervals> &_interval,
               const Eigen::Vector<double, CoDomainDim> &_a,
               const Eigen::Vector<double, CoDomainDim> &_b) {

    Eigen::Vector<double, PWGLVPolynomial::dimension> vec;

    PWGLVPolynomial result;
    result.domain_partition_ = _interval;

    Eigen::Matrix<double, 100, 1> mat;
    for (std::size_t i = 0; i < result.size(); i++) {
      result[i] =
          (result.domain_point(i) - _interval.interval().first()) * (_b - _a) +
          _a;
    }

    return result;
  }

  template <bool F = CoDomainDim == 1>
  static std::enable_if_t<F, PWGLVPolynomial>
  identity(const IntervalPartition<Intervals> &_interval) {

    PWGLVPolynomial result(_interval);

    Eigen::Matrix<double, 100, 1> mat;
    for (std::size_t i = 0; i < result.size(); i++) {
      result[i](0) = result.domain_point(i);
    }

    return result;
  }

  static PWGLVPolynomial
  from_repr(const IntervalPartition<Intervals> _domain_partision,
            const typename base_t::Representation &_in) {
    PWGLVPolynomial result = *base_t::from_repr(_in);
    result.domain_partition_ = _domain_partision;
    return result;
  }

  static PWGLVPolynomial from_function(
      const IntervalPartition<Intervals> _domain_partision,
      const std::function<typename base_t::codomain_t::Representation(double)>
          _fun) {
    PWGLVPolynomial result(_domain_partision);
    for (std::size_t i = 0; i < NumPoints * Intervals; i++) {
      result[i] = _fun(result.domain_point(i));
    }
    return result;
  }

  // ---------------------------------------
  // ------ Continuity matrix ---------------
  // ---------------------------------------
  static Eigen::SparseMatrix<double, Eigen::RowMajor>
  continuity_matrix(std::size_t deg) {

    Eigen::SparseMatrix<double, Eigen::RowMajor> result(
        (deg + 1) * CoDomainDim * (Intervals - 1), dimension);
    result.reserve(Eigen::VectorXi::Constant(dimension, 2 * NumPoints));
    // 0-continuity
    for (std::size_t k = 0; k < Intervals - 1; k++) {
      std::size_t j_lhs =
          k * CoDomainDim * NumPoints + CoDomainDim * (NumPoints - 1);
      std::size_t j_rhs = (k + 1) * CoDomainDim * NumPoints;
      for (std::size_t i = 0; i < CoDomainDim; i++) {
        result.coeffRef(k * CoDomainDim + i, j_lhs + i) = 1.0;
        result.coeffRef(k * CoDomainDim + i, j_rhs + i) = -1.0;
      }
    }
    if (deg == 0)
      return result;
    // 1-continuity
    static const Eigen::Matrix<double, NumPoints, NumPoints> dmat1 =
        PWGLVPolynomial<NumPoints, Intervals,
                        CoDomainDim>::template derivative_matrix_order_m<1>();

    for (std::size_t k = 0; k < Intervals - 1; k++) {
      std::size_t j_lhs = k * CoDomainDim * NumPoints;
      std::size_t j_rhs = (k + 1) * CoDomainDim * NumPoints;
      for (std::size_t point_block = 0; point_block < NumPoints;
           point_block++) {
        for (std::size_t i = 0; i < CoDomainDim; i++) {
          long lhs_row = (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;
          long rhs_row = (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;

          long rhs_col = j_rhs + point_block * CoDomainDim + i;
          long lhs_col = j_lhs + point_block * CoDomainDim + i;
          result.coeffRef(lhs_row, lhs_col) = dmat1(NumPoints - 1, point_block);
          result.coeffRef(rhs_row, rhs_col) = -dmat1(0, point_block);
        }
      }
    }

    if (deg == 1)
      return result;
    // 2-cotinuty
    static const Eigen::Matrix<double, NumPoints, NumPoints> dmat2 =
        PWGLVPolynomial<NumPoints, Intervals,
                        CoDomainDim>::template derivative_matrix_order_m<2>();

    for (std::size_t k = 0; k < Intervals - 1; k++) {
      std::size_t j_lhs = k * CoDomainDim * NumPoints;
      std::size_t j_rhs = (k + 1) * CoDomainDim * NumPoints;
      for (std::size_t point_block = 0; point_block < NumPoints;
           point_block++) {
        for (std::size_t i = 0; i < CoDomainDim; i++) {
          long lhs_row =
              2 * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;
          long rhs_row =
              2 * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;

          long rhs_col = j_rhs + point_block * CoDomainDim + i;
          long lhs_col = j_lhs + point_block * CoDomainDim + i;
          result.coeffRef(lhs_row, lhs_col) = dmat2(NumPoints - 1, point_block);
          result.coeffRef(rhs_row, rhs_col) = -dmat2(0, point_block);
        }
      }
    }
    // Eigen::IOFormat format(3, 0, ", ", ";\n", "", "", "[", "]");
    // std::cout << result.toDense().format(format) << "\n---\n\n\n";
    // std::cout << dmat1.format(format) << "\n---\n";
    if (deg > 2)
      throw std::invalid_argument("Continuoty matrix not implemented");
    return result;
  }
};

/** K-times continuosy Piece-wise Gauss Lobatto polynomial with fixed end poitns
 * NumPoints: number of gauss-lobato points
 * Intervals: number of intervals
 * DomainDim: Dimpension of the values of the pol
 * */
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim,
          std::size_t ContDeg, bool FixedStartPoint = false,
          bool FixedEndPoint = false>
class CPWGLVPolynomial
    : public detail::Clonable<
          CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                           FixedStartPoint, FixedEndPoint>,
          DenseLinearManifold<NumPoints * CoDomainDim * Intervals -
                                  CoDomainDim *(Intervals - 1) * (ContDeg + 1) -
                                  ((FixedEndPoint) ? CoDomainDim : 0) -
                                  ((FixedStartPoint) ? CoDomainDim : 0),
                              false>,
          Map<Reals, DenseLinearManifold<CoDomainDim>>> {

private:
  mutable std::array<double, NumPoints> evaluation_buffer_;

  IntervalPartition<Intervals> domain_partition_;

public:
  // ----------------------------
  // ------ Types ---------------
  // ----------------------------
  using interval_t = std::pair<double, double>;

  using manifold_t =
      DenseLinearManifold<NumPoints * CoDomainDim * Intervals -
                              CoDomainDim *(Intervals - 1) * (ContDeg + 1) -
                              ((FixedEndPoint) ? CoDomainDim : 0) -
                              ((FixedStartPoint) ? CoDomainDim : 0),
                          false>;

  using map_t = Map<Reals, DenseLinearManifold<CoDomainDim>>;

  using base_t = detail::Clonable<
      CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                       FixedStartPoint, FixedEndPoint>,
      DenseLinearManifold<NumPoints * CoDomainDim * Intervals -
                              CoDomainDim *(Intervals - 1) * (ContDeg + 1) -
                              ((FixedEndPoint) ? CoDomainDim : 0) -
                              ((FixedStartPoint) ? CoDomainDim : 0),
                          false>,
      Map<Reals, DenseLinearManifold<CoDomainDim>>>;
  ;
  using base_t::base_t;
  using base_t::dimension;

  struct point_value_t {
    long i0;
    long i1;
    Eigen::Map<Eigen::Vector<double, CoDomainDim>> value;
  };

  CPWGLVPolynomial() : base_t(), domain_partition_() {}

  CPWGLVPolynomial(double a, double b)
      : CPWGLVPolynomial(IntervalPartition<Intervals>(a, b)) {}

  CPWGLVPolynomial(const IntervalPartition<Intervals> &_domain)
      : base_t(), domain_partition_(_domain) {}

  CPWGLVPolynomial(const Interval &_domain,
                   const Eigen::Vector<double, dimension> &val)
      : base_t(val), domain_partition_(_domain) {}

  CPWGLVPolynomial(const CPWGLVPolynomial &) = default;
  CPWGLVPolynomial(CPWGLVPolynomial &&) = default;

  CPWGLVPolynomial &operator=(const CPWGLVPolynomial &_in) = default;
  CPWGLVPolynomial &operator=(CPWGLVPolynomial &&_in) = default;
  /// ---------------------------------------------------------
  /// --------------- Evaluation of the polynomial ------------
  /// ---------------------------------------------------------

  bool value_on_repr(const double &,
                     Eigen::Vector<double, CoDomainDim> &) const override {

    return true;
  }

  virtual bool diff_from_repr(const double &,
                              Eigen::Vector<double, CoDomainDim> &,
                              Eigen::Ref<Eigen::MatrixXd>) const override {

    return true;
  }

  static Eigen::SparseMatrix<double, Eigen::RowMajor> canonical_inclusion() {

    Eigen::SparseMatrix<double, Eigen::RowMajor> smT =
        PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::continuity_matrix(
            ContDeg)
            .transpose();

    // https://stackoverflow.com/questions/54766392/eigen-obtain-the-kernel-of-a-sparse-matrix
    Eigen::SparseQR<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                    Eigen::COLAMDOrdering<int>>
        solver;

    smT.makeCompressed();
    solver.compute(smT);
    Eigen::MatrixXd res =
        solver.matrixQ() * Eigen::MatrixXd::Identity(solver.matrixQ().rows(),
                                                     solver.matrixQ().cols())
                               .rightCols(smT.rows() - solver.rank());

    return res.sparseView();
  }

  static Eigen::SparseMatrix<double, Eigen::RowMajor> euclidean_projector() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> ci = canonical_inclusion();
    // TODO this is a pseudo inverse
    Eigen::SparseMatrix<double, Eigen::RowMajor> result =
        ((ci.transpose() * ci).toDense().inverse() * ci.transpose())
            .sparseView();
    return result;
  }

  class Inclusion;
  class EuclideanProjection;

  class ContinuityError;
  // ---------------------------------------
  // ------ Default Elements ---------------
  // ---------------------------------------

  static CPWGLVPolynomial zero(const IntervalPartition<Intervals> &_interval) {

    CPWGLVPolynomial result;

    result =
        *CPWGLVPolynomial::from_repr(CPWGLVPolynomial::Representation::Zero());

    result.domain_partition_ = IntervalPartition<Intervals>(_interval);

    return result;
  }

  static CPWGLVPolynomial
  random(const IntervalPartition<Intervals> &_interval) {

    CPWGLVPolynomial result;

    result = *CPWGLVPolynomial::from_repr(
        CPWGLVPolynomial::Representation::Random());

    result.domain_partition_ = IntervalPartition<Intervals>(_interval);

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
