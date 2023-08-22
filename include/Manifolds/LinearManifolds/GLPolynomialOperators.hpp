#pragma once
#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <omp.h>

namespace manifolds {

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <std::size_t Deg>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative
    : public detail::Clonable<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
          SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>> {

public:
  using base_t = detail::Clonable<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
      SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>>;
  using map_t =
      SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>;

  Derivative(const IntervalPartition<Intervals> &_interval)
      : base_t(), domain_partition_(_interval) {
    static auto matrix = this->get();

    this->repr() = matrix;

    for (std::size_t k = 0; k < Intervals; k += 1) {
      auto interval_block = this->repr().block(
          k * NumPoints * CoDomainDim, k * NumPoints * CoDomainDim,
          NumPoints * CoDomainDim, NumPoints * CoDomainDim);
      interval_block *=
          std::pow(2.0 / domain_partition_.subinterval_length(k), Deg);
    }
  }
  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {

    if (domain_partition_ != _in.domain_partition_)
      throw std::invalid_argument("pol has a different domain");

    _out.domain_partition_ = domain_partition_;

    this->get_repr(_out) = this->crepr() * _in.crepr();

    return true;
  }

private:
  IntervalPartition<Intervals> domain_partition_;
  static Eigen::SparseMatrix<double, Eigen::RowMajor> get() {

    static const Eigen::Matrix<double, NumPoints, NumPoints> dmat =
        PWGLVPolynomial<NumPoints, Intervals,
                        CoDomainDim>::template derivative_matrix_order_m<Deg>();

    // TODO Here the value of
    // SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>
    // Is the same of this->dimension. WHY!??
    Eigen::SparseMatrix<double, Eigen::RowMajor> result(
        base_t::codomain_t::dimension, base_t::domain_t::dimension);

    /*
    for (std::size_t k = 0; k < base_t::domain_dimension; k += NumPoints) {
      for (std::size_t i = 0; i < NumPoints; i++)
        for (std::size_t j = 0; j < NumPoints; j++)
          result.coeffRef(k + i, k + j) = dmat(i, j);
    }
    */

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(NumPoints * NumPoints * Intervals * CoDomainDim);

    result.reserve(NumPoints * NumPoints * Intervals * CoDomainDim);
    // #pragma omp parallel for num_threads(num_threads)
    for (std::size_t k = 0; k < Intervals; k += 1) {
      for (std::size_t i = 0; i < NumPoints; i++) {
        for (std::size_t j = 0; j < NumPoints; j++) {
          for (std::size_t l = 0; l < CoDomainDim; l++) {
            // #pragma omp critical
            {
              long i0 = k * NumPoints * CoDomainDim + i * CoDomainDim + l;
              long j0 = k * NumPoints * CoDomainDim + j * CoDomainDim + l;
              tripletList.emplace_back(i0, j0, dmat(i, j));
            }
          }

          /* result */
          /*     .block(k * NumPoints * CoDomainDim, k * NumPoints *
           * CoDomainDim, */
          /*            NumPoints * CoDomainDim, NumPoints * CoDomainDim) */
          /*     .block(i * CoDomainDim, j * CoDomainDim, CoDomainDim, */
          /*            CoDomainDim); */
          /* .coeffRef(l, l) = dmat(i, j); */
        }
      }
    }

    result.setFromTriplets(tripletList.begin(), tripletList.end());
    result.makeCompressed();

    return result;
  }
};

//---------------------------------------
//---------------------------------------
/// Space definition--
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <std::size_t Deg>
typename PWGLVPolynomial<NumPoints, Intervals,
                         CoDomainDim>::template Derivative<Deg>
PWGLVPolynomialSpace<NumPoints, Intervals, CoDomainDim>::derivative() {
  return PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>(
      this->interval_);
}

/// Integral of scalar polynomials
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral
    : public detail::Clonable<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral,
          DenseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                         Reals>> {
public:
  using base_t = detail::Clonable<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral,
      DenseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                     Reals>>;

  using map_t =
      DenseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>, Reals>;

  Integral(const IntervalPartition<Intervals> &_interval)
      : base_t(), domain_partition_(_interval) {

    static_assert(CoDomainDim == 1, "Cannot integrate vectors");
    static auto matrix = this->get();

    this->repr() = matrix;

    for (std::size_t k = 0; k < Intervals; k += 1) {
      this->repr().block(0, k * NumPoints, 1, NumPoints) *=
          domain_partition_.subinterval_length(k) / 2.0;
    }
  }
  virtual bool value_on_repr(const typename base_t::domain_t &_in,
                             double &_out) const override {

    if (domain_partition_ != _in.domain_partition_)
      throw std::invalid_argument("pol has a different domain");

    _out = this->crepr() * _in.crepr();

    return true;
  }

private:
  IntervalPartition<Intervals> domain_partition_;
  static Eigen::Matrix<double, 1, base_t::domain_t::dimension> get() {
    Eigen::Matrix<double, 1, base_t::domain_t::dimension> result;

    for (std::size_t k = 0; k < Intervals; k++) {
      result.template block<1, NumPoints>(0, k * NumPoints) =
          Eigen::Map<const Eigen::Vector<double, NumPoints>>(
              base_t::domain_t::gl_weights.data())
              .transpose();
    }
    return result;
  }
};

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim,
          std::size_t ContDeg, bool FixedStartPoint, bool FixedEndPoint>
class CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                       FixedStartPoint, FixedEndPoint>::ContinuityError
    : public detail::Clonable<
          CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                           FixedStartPoint, FixedEndPoint>::ContinuityError,
          SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                          DenseLinearManifold<5>>> {
private:
  IntervalPartition<Intervals> domain_partition_;

public:
  using base_t = detail::Clonable<
      CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                       FixedStartPoint, FixedEndPoint>::ContinuityError,
      SparseLinearMap<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          DenseLinearManifold<(ContDeg + 1) * CoDomainDim *(Intervals - 1)>>>;

  ContinuityError(const IntervalPartition<Intervals> &_partition)
      : base_t(), domain_partition_(_partition) {

    static auto matrix = base_t::domain_t::continuity_matrix(ContDeg);

    this->repr() = matrix;

    for (std::size_t deg = 1; deg <= ContDeg; deg++)
      for (std::size_t k = 0; k < Intervals - 1; k++) {
        std::size_t j_lhs = k * CoDomainDim * NumPoints;
        std::size_t j_rhs = (k + 1) * CoDomainDim * NumPoints;
        for (std::size_t point_block = 0; point_block < NumPoints;
             point_block++) {
          for (std::size_t i = 0; i < CoDomainDim; i++) {
            long lhs_row =
                deg * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;
            long rhs_row =
                deg * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;

            long rhs_col = j_rhs + point_block * CoDomainDim + i;
            long lhs_col = j_lhs + point_block * CoDomainDim + i;
            this->repr().coeffRef(lhs_row, lhs_col) *=
                std::pow(domain_partition_.subinterval_length(k) / 2.0, deg);
            this->repr().coeffRef(rhs_row, rhs_col) *= std::pow(
                domain_partition_.subinterval_length(k + 1) / 2.0, deg);
          }
        }
      }
  }
  virtual bool value_on_repr(const typename base_t::domain_t &_in,
                             typename base_t::codomain_t &_out) const override {

    if (domain_partition_ != _in.domain_partition_)
      throw std::invalid_argument("pol has a different domain");

    _out.domain_partition_ = domain_partition_;

    this->get_repr(_out) = this->crepr() * _in.crepr();

    return true;
  }
};

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim,
          std::size_t ContDeg, bool FixedStartPoint, bool FixedEndPoint>
class CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                       FixedStartPoint, FixedEndPoint>::Inclusion
    : public detail::Clonable<
          CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                           FixedStartPoint, FixedEndPoint>::Inclusion,
          SparseLinearMap<
              CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                               FixedStartPoint, FixedEndPoint>,
              PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>> {
private:
  IntervalPartition<Intervals> domain_partition_;

public:
  using map_t =
      SparseLinearMap<CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim,
                                       ContDeg, FixedStartPoint, FixedEndPoint>,
                      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>;
  using base_t = detail::Clonable<
      CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim, ContDeg,
                       FixedStartPoint, FixedEndPoint>::Inclusion,
      SparseLinearMap<CPWGLVPolynomial<NumPoints, Intervals, CoDomainDim,
                                       ContDeg, FixedStartPoint, FixedEndPoint>,
                      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>>;

  Inclusion(double a, double b)
      : Inclusion(IntervalPartition<Intervals>(a, b)) {}

  Inclusion(const IntervalPartition<Intervals> &_partition)
      : base_t(), domain_partition_(_partition) {

    static auto matrix = base_t::codomain_t::continuity_matrix(ContDeg);

    auto current_matrix = matrix;

    for (std::size_t deg = 1; deg <= ContDeg; deg++)
      for (std::size_t k = 0; k < Intervals - 1; k++) {
        std::size_t j_lhs = k * CoDomainDim * NumPoints;
        std::size_t j_rhs = (k + 1) * CoDomainDim * NumPoints;
        for (std::size_t point_block = 0; point_block < NumPoints;
             point_block++) {
          for (std::size_t i = 0; i < CoDomainDim; i++) {
            long lhs_row =
                deg * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;
            long rhs_row =
                deg * (Intervals - 1) * CoDomainDim + k * CoDomainDim + i;

            long rhs_col = j_rhs + point_block * CoDomainDim + i;
            long lhs_col = j_lhs + point_block * CoDomainDim + i;
            current_matrix.coeffRef(lhs_row, lhs_col) *=
                std::pow(domain_partition_.subinterval_length(k) / 2.0, deg);
            current_matrix.coeffRef(rhs_row, rhs_col) *= std::pow(
                domain_partition_.subinterval_length(k + 1) / 2.0, deg);
          }
        }
      }
    // https://stackoverflow.com/questions/54766392/eigen-obtain-the-kernel-of-a-sparse-matrix
    Eigen::SparseQR<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                    Eigen::COLAMDOrdering<int>>
        solver;

    current_matrix.makeCompressed();
    solver.compute(current_matrix.transpose());
    Eigen::MatrixXd res = solver.matrixQ() *
                          Eigen::MatrixXd::Identity(solver.matrixQ().rows(),
                                                    solver.matrixQ().cols())
                              .rightCols(current_matrix.cols() - solver.rank());
    this->repr() = res.sparseView();
  }
  virtual bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_out) const override {

    if (domain_partition_ != _in.domain_partition_)
      throw std::invalid_argument("pol has a different domain");

    _out.domain_partition_ = domain_partition_;

    this->get_repr(_out) = this->crepr() * _in.crepr();

    return true;
  }
};

//---------------------------------------

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <typename OtherCoDomain>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Composition
    : public detail::Clonable<
          PWGLVPolynomial<NumPoints, Intervals,
                          CoDomainDim>::Composition<OtherCoDomain>,
          Map<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
              PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension>,
              detail::MatrixTypeId::Sparse>> {
public:
  using base_t = detail::Clonable<
      PWGLVPolynomial<NumPoints, Intervals,
                      CoDomainDim>::Composition<OtherCoDomain>,
      Map<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
          PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension>,
          detail::MatrixTypeId::Sparse>>;

  using value_fun_t = std::function<bool(
      const typename DenseLinearManifold<CoDomainDim>::Representation &,
      typename OtherCoDomain::Representation &)>;
  using diff_fun_t = std::function<bool(
      const typename DenseLinearManifold<CoDomainDim>::Representation &,
      detail::dense_matrix_ref_t)>;

  Composition(const Map<DenseLinearManifold<CoDomainDim>, OtherCoDomain> &in)
      : map_(in.clone()) {}

  Composition(const value_fun_t &_value_fun, const diff_fun_t &_diff_fun)
      : base_t(),
        map_(std::make_unique<
             MapLifting<DenseLinearManifold<CoDomainDim>, OtherCoDomain,
                        detail::MatrixTypeId::Dense>>(_value_fun, _diff_fun)) {}

  Composition(const Composition &_that) { map_ = _that.map_->clone(); }
  Composition(Composition &&_that) { map_ = _that.map_->move_clone(); }

  bool
  value_on_repr(const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> &in,
                PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension>
                    &out) const override {

    out.domain_partition_ = in.domain_partition_;
    for (std::size_t point_index = 0; point_index < in.size(); point_index++) {
      // TODO, fix this by adding more complex reference to manifolds
      domain_buffer_ = in[point_index];
      map_->value(domain_buffer_, codomain_buffer_);
      out[point_index] = codomain_buffer_;
    }
    return true;
  }

  virtual bool diff_from_repr(
      const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> &in,
      PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension> &,
      std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>> _mat)
      const override {

    Eigen::Matrix<double, OtherCoDomain::dimension, CoDomainDim> mat;

    int i0 = 0;
    int j0 = 0;
    OtherCoDomain buff;
    for (std::size_t point_index = 0; point_index < in.size(); point_index++) {
      map_->diff(in[point_index], buff, mat);
      for (std::size_t i = 0; i < OtherCoDomain::dimension; i++)
        for (std::size_t j = 0; j < CoDomainDim; j++)
          _mat.get().coeffRef(i0 + i, j0 + j) = mat(i, j);
      i0 += OtherCoDomain::dimension;
      j0 += CoDomainDim;
    }

    return true;
  }

private:
  std::unique_ptr<Map<DenseLinearManifold<CoDomainDim>, OtherCoDomain>> map_;
  mutable Eigen::Vector<double, CoDomainDim> domain_buffer_;
  mutable Eigen::Vector<double, OtherCoDomain::dimension> codomain_buffer_;
};

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::functions {
public:
  inline static const Composition<DenseLinearManifold<1>> norm =
      Composition<DenseLinearManifold<1>>(
          [](const Eigen::Vector<double, CoDomainDim> &in,
             Eigen::Vector<double, 1> &out) {
            out(0) = in.norm();
            return true;
          },
          [](const Eigen::Matrix<double, 1, CoDomainDim> &in,
             Eigen::Ref<Eigen::MatrixXd> _diff) {
            _diff = Eigen::Vector<double, CoDomainDim>::Ones() / in.norm();
            return true;
          });
};

} // namespace manifolds
