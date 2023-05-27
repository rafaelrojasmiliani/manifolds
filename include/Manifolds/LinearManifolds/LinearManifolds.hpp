#pragma once
#include <Manifolds/Atlases/LinearManifolds.h>
#include <Manifolds/Detail.hpp>
#include <Manifolds/Manifold.hpp>
#include <iostream>
#include <random>
#include <type_traits>

namespace manifolds {

template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class LinearManifoldInheritanceHelper : public Base {
private:
  using ThisClass = LinearManifoldInheritanceHelper<Current, Base, DT>;

public:
  /// Representation type from its base
  using Representation = typename Base::Representation;

  /// Implement the constructor of its base class

  bool operator==(const Current &that) {

    double _tol = 1.0e-12;
    double err = 0;
    double lhs_max = 0;
    double rhs_max = 0;
    std::visit(
        [&err, &lhs_max, &rhs_max](const auto &lhs, const auto rhs) {
          err = (lhs - rhs).array().abs().maxCoeff();
          lhs_max = (lhs).array().abs().maxCoeff();
          rhs_max = (rhs).array().abs().maxCoeff();
        },
        this->crepr(), that.crepr());

    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }

  auto operator+(const Representation &_in) & {
    return std::visit([](auto &&_lhs, auto &&_rhs) { return _lhs + _rhs; },
                      this->crepr(), _in);
  }
  bool operator!=(const Current &that) { return not(*this == that); }

  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(LinearManifoldInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(Current, Base)
};

/// Left multiplication by scalar
template <typename C, typename B>
auto operator*(double m, const LinearManifoldInheritanceHelper<C, B> &that) {
  return m * that.crepr();
}
/* template <typename C, typename B, typename R> */
/* std::enable_if_t<std::is_base_of_v<LinearManifoldInheritanceHelper<C, B>>, R>
 */
/* operator*(double, B::Representation &&_other) { */
/*   if (not representation_) */
/*     throw std::logic_error("Trying to assign to a constnat manifold
 * element"); */
/*   *representation_ = std::move(_other); */
/*   return *this; */
/* } */

template <long Rows, long Cols>
using RealTuplesBase = Manifold<LinearManifoldAtlas<Rows, Cols>, true>;

template <long Rows, long Cols>
class MatrixManifold
    : public LinearManifoldInheritanceHelper<MatrixManifold<Rows, Cols>,
                                             RealTuplesBase<Rows, Cols>> {
public:
  static constexpr long rows = Rows;
  static constexpr long cols = Cols;

  /// Type definitonis

  using base_t = LinearManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<LinearManifoldAtlas<Rows, Cols>, true>>;

  using typename base_t::Representation;

  // Inheritance
  using base_t::base_t;
  using base_t::operator=;

  /// Live cycle
  ///
  __DEFAULT_LIVE_CYCLE(MatrixManifold)
  /// specific casting operators

  /// Getters
  Eigen::Matrix<double, Rows, Cols> &eigen_dense() {
    return std::get<Eigen::Matrix<double, Rows, Cols>>(this->repr());
  }
  const Eigen::Matrix<double, Rows, Cols> &ceigen_dense() const {
    return std::get<Eigen::Matrix<double, Rows, Cols>>(this->crepr());
  }

  Eigen::SparseMatrix<double> &eigen_sparse() {
    return std::get<Eigen::SparseMatrix<double>>(this->repr());
  }
  const Eigen::SparseMatrix<double> &ceigen_sparse() const {
    return std::get<Eigen::SparseMatrix<double>>(this->crepr());
  }

  static DenseMatrix<Cols, Rows> get_dense_random() {
    return DenseMatrix<Cols, Rows>::Random();
  }
  static SparseMatrix get_sparse_random() {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int rows = Rows;
    int cols = Cols;

    std::vector<Eigen::Triplet<double>> tripletList;
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) {
        auto v_ij = dist(gen); // generate random number
        if (v_ij < 0.1) {
          tripletList.push_back(Eigen::Triplet<double>(
              i, j, v_ij)); // if larger than treshold, insert it
        }
      }
    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
  }
};

template <long Rows, long Cols>
class DenseMatrixManifold
    : public LinearManifoldInheritanceHelper<DenseMatrixManifold<Rows, Cols>,
                                             MatrixManifold<Rows, Cols>,
                                             MatrixTypeId::Dense> {

public:
  using base_t =
      LinearManifoldInheritanceHelper<DenseMatrixManifold<Rows, Cols>,
                                      MatrixManifold<Rows, Cols>,
                                      MatrixTypeId::Dense>;
  using T = DenseMatrix<Rows, Cols>;
  DenseMatrixManifold() : base_t(T()) {}
  DenseMatrixManifold(DenseMatrixConstRef in) : base_t(in) {}
  DenseMatrixManifold(SparseMatrixConstRef in) : base_t(in.get().toDense()) {}

  DenseMatrixManifold(const DenseMatrixManifold &that) : base_t(that) {}
  DenseMatrixManifold(DenseMatrixManifold &&that) : base_t(std::move(that)) {}

  DenseMatrixManifold &operator=(DenseMatrixRef in) {
    this->repr() = in;
    return *this;
  }
  DenseMatrixManifold &operator=(const DenseMatrixManifold &in) {
    base_t::operator=(in);
    return *this;
  }
  DenseMatrixManifold &operator=(const MatrixManifold<Rows, Cols> &in) {
    base_t::operator=(std::get<T>(in.crepr()));
    return *this;
  }

private:
  using base_t::operator=;
};

template <long Rows, long Cols>
class SparseMatrixManifold
    : public LinearManifoldInheritanceHelper<DenseMatrixManifold<Rows, Cols>,
                                             MatrixManifold<Rows, Cols>,
                                             MatrixTypeId::Sparse> {

public:
  using base_t =
      LinearManifoldInheritanceHelper<DenseMatrixManifold<Rows, Cols>,
                                      MatrixManifold<Rows, Cols>,
                                      MatrixTypeId::Dense>;
  using T = SparseMatrix;
  SparseMatrixManifold() : base_t(T()) {}
  SparseMatrixManifold(DenseMatrixConstRef in) : base_t(in) {}
  SparseMatrixManifold(SparseMatrixConstRef in) : base_t(in.get().toDense()) {}

  SparseMatrixManifold(const SparseMatrixManifold &that) : base_t(that) {}
  SparseMatrixManifold(SparseMatrixManifold &&that) : base_t(std::move(that)) {}

  SparseMatrixManifold &operator=(DenseMatrixRef in) { this->repr() = in; }
  SparseMatrixManifold &operator=(const Eigen::SparseMatrix<double> &in) {
    base_t::operator=(in);
  }
  SparseMatrixManifold &operator=(const SparseMatrixManifold &in) {
    base_t::operator=(std::get<T>(in.crepr()));
  }

private:
  using base_t::operator=;
};

template <long Rows> using LinearManifold = MatrixManifold<Rows, 1>;

using R2 = DenseMatrixManifold<2, 1>;
using R3 = DenseMatrixManifold<3, 1>;
using R4 = DenseMatrixManifold<4, 1>;
using R5 = DenseMatrixManifold<5, 1>;
using R6 = DenseMatrixManifold<6, 1>;
using R7 = DenseMatrixManifold<7, 1>;

} // namespace manifolds
