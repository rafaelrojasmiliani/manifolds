#pragma once
#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <omp.h>

namespace manifolds {

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <std::size_t Deg>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative
    : public LinearMapInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
          SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>> {

public:
  using base_t = LinearMapInheritanceHelper<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
      SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>>;

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
        base_t::codomain::dimension, base_t::domain::dimension);

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

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral
    : public LinearMapInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral,
          DenseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                         Reals>> {
  using base_t = LinearMapInheritanceHelper<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Integral,
      DenseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                     Reals>>;

  Integral(const IntervalPartition<Intervals> &_interval)
      : base_t(), domain_partition_(_interval) {

    static_assert(CoDomainDim == 1, "Cannot integrate vectors");
    static auto matrix = this->get();

    this->repr() = matrix;

    for (std::size_t k = 0; k < Intervals; k += 1) {
      auto &interval_block = this->repr().block(k * NumPoints, 0, 1, NumPoints);
      interval_block *= domain_partition_.subinterval_length(k) / 2.0;
    }
  }

private:
  IntervalPartition<Intervals> domain_partition_;
  static Eigen::Matrix<double, 1, base_t::domain::dimension> get() {
    Eigen::Matrix<double, 1, base_t::domain::dimension> result;
    for (std::size_t k = 0; k < Intervals; k += 1) {
      auto &interval_block = result.block(k * NumPoints, 0, 1, NumPoints);
      interval_block = Eigen::Map<decltype(result)>(
          result.base_t::domain::gl_weights.data());
    }
  }
};

//---------------------------------------
/*
template <std::size_t NumPoints, std::size_t Intervals, std::size_t
CoDomainDim> template <typename OtherCoDomain> class
PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Composition : public
MapInheritanceHelper< PWGLVPolynomial<NumPoints, Intervals,
                          CoDomainDim>::Composition<OtherCoDomain>,
          Map<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
              PWGLVPolynomial<NumPoints, Intervals,
OtherCoDomain::dimension>, true>> { public: Composition(const
Map<LinearManifold<CoDomainDim>, OtherCoDomain, true> &in) :
map_(in.clone()) {}

  bool
  value_on_repr(const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>
&in, PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension> &out)
const override {

    for (std::size_t point_index = 0; point_index < in.size();
point_index++) { map_->value(in[point_index], out[point_index]);
    }
    return true;
  }

  virtual bool diff_from_repr(
      const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> &in,
      std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>>
_mat) const override {

    Eigen::Matrix<double, OtherCoDomain::dimension, CoDomainDim> mat;

    int i0 = 0;
    int j0 = 0;
    for (std::size_t point_index = 0; point_index < in.size();
point_index++) { map_->diff(in[point_index], mat); for (int i = 0; i <
OtherCoDomain::dimension; i++) for (int j = 0; j < CoDomainDim; j++)
          _mat.get().coeffRef(i0 + i, j0 + j) = mat(i, j);
      i0 += OtherCoDomain::dimension;
      j0 += CoDomainDim;
    }

    return true;
  }

private:
  std::unique_ptr<Map<LinearManifold<CoDomainDim>, OtherCoDomain, true>>
map_;
};*/
} // namespace manifolds
