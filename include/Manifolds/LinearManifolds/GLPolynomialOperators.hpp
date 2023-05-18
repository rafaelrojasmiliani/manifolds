#pragma once
#include <Manifolds/LinearManifolds/GLPolynomial.hpp>

namespace manifolds {

template <std::size_t NumPoints>
template <std::size_t Deg>
class GLPolynomial<NumPoints>::Derivative
    : public MapInheritanceHelper<
          GLPolynomial<NumPoints>::Derivative<Deg>,
          LinearMap<GLPolynomial<NumPoints>, GLPolynomial<NumPoints>>> {
public:
  using base_t = MapInheritanceHelper<
      GLPolynomial<NumPoints>::Derivative<Deg>,
      LinearMap<GLPolynomial<NumPoints>, GLPolynomial<NumPoints>>>;
  Derivative() : base_t(GLPolynomial::derivative_matrix_order_m<Deg>()) {}
};

template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <std::size_t Deg>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative
    : public LinearManifoldInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
          SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                          PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>> {

  using base_t = MapInheritanceHelper<
      PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Derivative<Deg>,
      LinearMap<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
                PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>>>;

public:
  using base_t::base_t;
  using base_t::codomain_dimension;
  using base_t::dimension;
  using base_t::domain_dimension;
  using base_t::operator=;

  Derivative(const Interval &_interval)
      : base_t(), domain_interval_(_interval) {
    static auto matrix = this->get();

    this->repr() = matrix;

    for (std::size_t k = 0; k < Intervals; k += 1) {
      auto interval_block = this->repr().block(
          k * NumPoints * CoDomainDim, k * NumPoints * CoDomainDim,
          NumPoints * CoDomainDim, NumPoints * CoDomainDim);
      interval_block *= 2.0 / domain_interval_.subinterval_length(k);
    }
  }

private:
  IntervalPartition<Intervals> domain_interval_;
  static Eigen::SparseMatrix<double> get() {
    static const Eigen::Matrix<double, NumPoints, NumPoints> dmat =
        GLPolynomial<NumPoints>::template derivative_matrix_order_m<Deg>();

    Eigen::SparseMatrix<double> result(codomain_dimension, domain_dimension);

    /*
    for (std::size_t k = 0; k < base_t::domain_dimension; k += NumPoints) {
      for (std::size_t i = 0; i < NumPoints; i++)
        for (std::size_t j = 0; j < NumPoints; j++)
          result.coeffRef(k + i, k + j) = dmat(i, j);
    }
    */
    for (std::size_t k = 0; k < Intervals; k += 1) {
      auto interval_block =
          result.block(k * NumPoints * CoDomainDim, k * NumPoints * CoDomainDim,
                       NumPoints * CoDomainDim, NumPoints * CoDomainDim);
      for (std::size_t i = 0; i < NumPoints; i++)
        for (std::size_t j = 0; j < NumPoints; j++) {
          auto point_block = interval_block.block(
              i * CoDomainDim, j * CoDomainDim, CoDomainDim, CoDomainDim);
          for (std::size_t l = 0; l < CoDomainDim; l++)
            point_block.coeffRef(l, l) = dmat(i, j);
        }
    }
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

//---------------------------------------
template <std::size_t NumPoints, std::size_t Intervals, std::size_t CoDomainDim>
template <typename OtherCoDomain>
class PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>::Composition
    : public MapInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals,
                          CoDomainDim>::Composition<OtherCoDomain>,
          Map<PWGLVPolynomial<NumPoints, Intervals, CoDomainDim>,
              PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension>,
              true>> {
public:
  Composition(const Map<LinearManifold<CoDomainDim>, OtherCoDomain, true> &in)
      : map_(in.clone()) {}

  bool
  value_on_repr(const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> &in,
                PWGLVPolynomial<NumPoints, Intervals, OtherCoDomain::dimension>
                    &out) const override {

    for (std::size_t point_index = 0; point_index < in.size(); point_index++) {
      map_->value(in[point_index], out[point_index]);
    }
    return true;
  }

  virtual bool diff_from_repr(
      const PWGLVPolynomial<NumPoints, Intervals, CoDomainDim> &in,
      std::reference_wrapper<Eigen::SparseMatrix<double>> _mat) const override {

    Eigen::Matrix<double, OtherCoDomain::dimension, CoDomainDim> mat;

    int i0 = 0;
    int j0 = 0;
    for (std::size_t point_index = 0; point_index < in.size(); point_index++) {
      map_->diff(in[point_index], mat);
      for (int i = 0; i < OtherCoDomain::dimension; i++)
        for (int j = 0; j < CoDomainDim; j++)
          _mat.get().coeffRef(i0 + i, j0 + j) = mat(i, j);
      i0 += OtherCoDomain::dimension;
      j0 += CoDomainDim;
    }

    return true;
  }

private:
  std::unique_ptr<Map<LinearManifold<CoDomainDim>, OtherCoDomain, true>> map_;
};
} // namespace manifolds
