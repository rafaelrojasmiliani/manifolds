#pragma once
#include <Manifolds/Atlases/LinearManifolds.h>
#include <Manifolds/Detail.hpp>
#include <Manifolds/Manifold.hpp>
#include <iostream>
#include <random>
#include <type_traits>

namespace manifolds {

class LinearManifold {};

template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class MixedLinearManifoldInheritanceHelper : public Base {
private:
  using ThisClass = MixedLinearManifoldInheritanceHelper<Current, Base, DT>;

public:
  /// Representation type from its base
  using Representation = typename Base::Representation;

  /// Implement the constructor of its base class
  auto operator+(const Representation &_in) & {
    return std::visit([](auto &&_lhs, auto &&_rhs) { return _lhs + _rhs; },
                      this->crepr(), _in);
  }

  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(MixedLinearManifoldInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(ThisClass, Current, Base)
  __DEFAULT_REF(Current, Base)
};

template <typename Current, typename Base>
class LinearManifoldInheritanceHelper : public Base {
private:
  using ThisClass = LinearManifoldInheritanceHelper<Current, Base>;

public:
  /// Representation type from its base
  using Representation = typename Base::Representation;

  /// Implement the constructor of its base class

  inline auto operator+(const Current &_in) & {
    return this->crepr() + _in.crepr();
  }

  inline auto operator+(Current &&_in) & {
    return this->crepr() + std::move(_in.crepr());
  }

  inline auto operator+(const Current &_in) && {
    return std::move(this->repr()) + _in.crepr();
  }

  inline auto operator+(Current &&_in) && {
    return std::move(this->repr()) + std::move(_in.crepr());
  }

  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(LinearManifoldInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(ThisClass, Current, Base)
  __DEFAULT_REF(Current, Base)
};
/// Left multiplication by scalar
template <typename C, typename B>
auto operator*(double m,
               const MixedLinearManifoldInheritanceHelper<C, B> &that) {
  return m * that.crepr();
}
/* template <typename C, typename B, typename R> */
/* std::enable_if_t<std::is_base_of_v<MixedLinearManifoldInheritanceHelper<C,
 * B>>, R>
 */
/* operator*(double, B::Representation &&_other) { */
/*   if (not representation_) */
/*     throw std::logic_error("Trying to assign to a constnat manifold
 * element"); */
/*   *representation_ = std::move(_other); */
/*   return *this; */
/* } */

template <long Rows, long Cols>
using MixedTuplesBase = Manifold<LinearManifoldAtlas<Rows, Cols>, true>;

template <long Rows, long Cols, bool F>
using DenseRealTuplesBase = Manifold<DenseLinearManifoldAtlas<Rows, Cols>, F>;

template <long Rows, long Cols, bool F>
using SparseRealTuplesBase = Manifold<SparseLinearManifoldAtlas<Rows, Cols>, F>;

template <long Rows, long Cols>
class MixedMatrixManifold
    : public MixedLinearManifoldInheritanceHelper<
          MixedMatrixManifold<Rows, Cols>, MixedTuplesBase<Rows, Cols>>,
      public LinearManifold {
public:
  static constexpr long rows = Rows;
  static constexpr long cols = Cols;

  /// Type definitonis

  using base_t = MixedLinearManifoldInheritanceHelper<
      MixedMatrixManifold<Rows, Cols>,
      Manifold<LinearManifoldAtlas<Rows, Cols>, true>>;

  using typename base_t::Representation;

  // Inheritance
  using base_t::base_t;
  using base_t::operator=;

  /// Live cycle
  ///
  __DEFAULT_LIVE_CYCLE(MixedMatrixManifold)

  MixedMatrixManifold &operator=(const Eigen::Matrix<double, Rows, Cols> &in) {
    this->repr() = in;
    return *this;
  }

  MixedMatrixManifold &
  operator=(const Eigen::SparseMatrix<double, Eigen::RowMajor> &in) {
    this->repr() = in;
    return *this;
  }
  /// specific casting operators

  /// Getters
  Eigen::Matrix<double, Rows, Cols> &eigen_dense() {
    return std::get<Eigen::Matrix<double, Rows, Cols>>(this->repr());
  }
  const Eigen::Matrix<double, Rows, Cols> &ceigen_dense() const {
    return std::get<Eigen::Matrix<double, Rows, Cols>>(this->crepr());
  }

  Eigen::SparseMatrix<double, Eigen::RowMajor> &eigen_sparse() {
    return std::get < Eigen::SparseMatrix < double,
           Eigen::RowMajor >>> (this->repr());
  }
  const Eigen::SparseMatrix<double, Eigen::RowMajor> &ceigen_sparse() const {
    return std::get<Eigen::SparseMatrix<double, Eigen::RowMajor>>(
        this->crepr());
  }

  bool is_dense() const {
    return std::holds_alternative<Eigen::Matrix<double, Rows, Cols>>(
        this->crepr());
  }

  bool is_sparse() const {
    return std::holds_alternative < Eigen::SparseMatrix < double,
           Eigen::RowMajor >>> (this->crepr());
  }
  static DenseMatrix<Rows, Cols> get_dense_random() {
    return DenseMatrix<Rows, Cols>::Random();
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
class DenseMatrixManifold : public LinearManifoldInheritanceHelper<
                                DenseMatrixManifold<Rows, Cols>,
                                DenseRealTuplesBase<Rows, Cols, true>>,
                            public LinearManifold {

public:
  using base_t =
      LinearManifoldInheritanceHelper<DenseMatrixManifold<Rows, Cols>,
                                      DenseRealTuplesBase<Rows, Cols, true>>;
  using Representation = typename base_t::Representation;
  using base_t::base_t;
  using base_t::operator=;
  __DEFAULT_LIVE_CYCLE(DenseMatrixManifold);
};

template <long Rows, long Cols>
class SparseMatrixManifold : public LinearManifoldInheritanceHelper<
                                 SparseMatrixManifold<Rows, Cols>,
                                 SparseRealTuplesBase<Rows, Cols, true>>,
                             public LinearManifold {

public:
  using base_t =
      LinearManifoldInheritanceHelper<SparseMatrixManifold<Rows, Cols>,
                                      SparseRealTuplesBase<Rows, Cols, true>>;
  using Representation = typename base_t::Representation;
  using base_t::base_t;
  using base_t::operator=;
  __DEFAULT_LIVE_CYCLE(SparseMatrixManifold);
};

template <long Rows> using MixedLinearManifold = MixedMatrixManifold<Rows, 1>;
template <long Rows> using DenseLinearManifold = DenseMatrixManifold<Rows, 1>;
template <long Rows> using SparseLinearManifold = SparseMatrixManifold<Rows, 1>;

using R2 = DenseMatrixManifold<2, 1>;
using R3 = DenseMatrixManifold<3, 1>;
using R4 = DenseMatrixManifold<4, 1>;
using R5 = DenseMatrixManifold<5, 1>;
using R6 = DenseMatrixManifold<6, 1>;
using R7 = DenseMatrixManifold<7, 1>;

} // namespace manifolds
