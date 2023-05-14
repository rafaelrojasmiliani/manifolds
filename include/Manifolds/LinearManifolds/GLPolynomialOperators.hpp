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

template <std::size_t NumPoints, std::size_t Intervals, std::size_t DomainDim>
template <std::size_t Deg>
class PWGLVPolynomial<NumPoints, Intervals, DomainDim>::Derivative
    : public LinearManifoldInheritanceHelper<
          PWGLVPolynomial<NumPoints, Intervals, DomainDim>::Derivative<Deg>,
          SparseLinearMap<PWGLVPolynomial<NumPoints, Intervals, DomainDim>,
                          PWGLVPolynomial<NumPoints, Intervals, DomainDim>>> {

  using base_t = MapInheritanceHelper<
      PWGLVPolynomial<NumPoints, Intervals, DomainDim>::Derivative<Deg>,
      LinearMap<PWGLVPolynomial<NumPoints, Intervals, DomainDim>,
                PWGLVPolynomial<NumPoints, Intervals, DomainDim>>>;

  using base_t::codomain_dimension;
  using base_t::dimension;
  using base_t::domain_dimension;

  Derivative() : base_t() {
    static auto matrix = this->get();
    this->repr() = matrix;
  }

private:
  static Eigen::SparseMatrix<double> get() {
    static const Eigen::Matrix<double, NumPoints, NumPoints> dmat =
        GLPolynomial<NumPoints>::template derivative_matrix_order_m<Deg>();

    Eigen::SparseMatrix<double> result(codomain_dimension, domain_dimension);

    for (std::size_t k = 0; k < base_t::domain_dimension; k += NumPoints) {
      for (std::size_t i = 0; i < NumPoints; i++)
        for (std::size_t j = 0; j < NumPoints; j++)
          result.coeffRef(k + i, k + j) = dmat(i, j);
    }
    result.makeCompressed();
    return result;
  }
};
} // namespace manifolds
