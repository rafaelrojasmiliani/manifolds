#pragma once
#include <Manifolds/Atlases/LinearManifolds.h>
#include <Manifolds/Manifold.hpp>
#include <type_traits>

namespace manifolds {

template <typename Current, typename Base>
class LinearManifoldInheritanceHelper : public Base {
private:
  using ThisClass = LinearManifoldInheritanceHelper<Current, Base>;

public:
  /// Representation type from its base
  using Representation = typename Base::Representation;

  /// Implement the constructor of its base class
  using Base::Base;

  /// Copy constructor
  LinearManifoldInheritanceHelper(const ThisClass &_in) : Base(_in) {}

  /// Move constructor
  LinearManifoldInheritanceHelper(ThisClass &&_in) : Base(std::move(_in)) {}

  /// Default constructor
  virtual ~LinearManifoldInheritanceHelper() = default;

  /// Clone
  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  /// Move clone
  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

  /// Implement assigment operator of the base
  using Base::operator=;

  /// Assigment operator
  ThisClass &operator=(const ThisClass &that) {
    Base::operator=(that);
    return *this;
  }

  /// Assigment move operator
  ThisClass &operator=(ThisClass &&that) {
    Base::operator=(std::move(that));
    return *this;
  }

  /// Sum operation
  inline auto operator+(const Representation &that) const & {
    return Base::crepr() + that;
  }

  /// Sum operation
  inline auto operator+(const Representation &that) && {
    return std::move(Base::mrepr()) + that;
  }

  inline auto operator+(Representation &&that) const & {
    return Base::crepr() + std::move(that);
  }

  inline auto operator+(Representation &&that) && {
    return std::move(Base::mrepr()) + std::move(that);
  }

  inline void operator+=(const Representation &that) { Base::repr() += that; }

  inline void operator+=(Representation &&that) { Base::repr() += that; }

  inline auto operator*(double that) const & { return Base::crepr() * that; }

  inline auto operator*(double that) && {
    return std::move(Base::mrepr()) * that;
  }

  inline void operator*=(double that) { Base::repr() *= that; }

  bool operator==(const Current &that) {

    static typename Current::Representation diff_buffer;
    double _tol = 1.0e-12;
    diff_buffer = this->crepr() - that.crepr();
    double err = 0;
    double lhs_max = 0;
    double rhs_max = 0;
    if constexpr (std::is_same_v<typename Current::Representation, double>) {
      err = std::fabs(diff_buffer);
      lhs_max = std::fabs(this->crepr());
      rhs_max = std::fabs(that.crepr());
    } else {
      err = diff_buffer.array().abs().maxCoeff();
      lhs_max = (this->crepr()).array().abs().maxCoeff();
      rhs_max = (that.crepr()).array().abs().maxCoeff();
    }

    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }

  bool operator!=(const Current &that) { return not(*this == that); }

protected:
  virtual ManifoldBase *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual ManifoldBase *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
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
using SparseRealTuplesBase =
    Manifold<SparseLinearManifoldAtlas<Rows, Cols>, true>;

template <long Rows, long Cols>
class MatrixManifold
    : public LinearManifoldInheritanceHelper<MatrixManifold<Rows, Cols>,
                                             RealTuplesBase<Rows, Cols>> {
public:
  using base_t = LinearManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<LinearManifoldAtlas<Rows, Cols>, true>>;
  using base_t::base_t;
  using base_t::operator=;
  const MatrixManifold &operator=(const MatrixManifold &that) {
    base_t::operator=(that);
    return *this;
  }
  const MatrixManifold &operator=(MatrixManifold &&that) {
    base_t::operator=(std::move(that));
    return *this;
  }
  MatrixManifold(const MatrixManifold &_in) : base_t(_in) {}
  MatrixManifold(MatrixManifold &&_in) : base_t(std::move(_in)) {}
  virtual ~MatrixManifold() = default;
};

template <long Rows, long Cols>
class SparseMatrixManifold
    : public LinearManifoldInheritanceHelper<SparseMatrixManifold<Rows, Cols>,
                                             SparseRealTuplesBase<Rows, Cols>> {
public:
  using base_t = LinearManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<LinearManifoldAtlas<Rows, Cols>, true>>;
  using base_t::base_t;
  using base_t::operator=;
  const SparseMatrixManifold &operator=(const SparseMatrixManifold &that) {
    base_t::operator=(that);
    return *this;
  }
  const SparseMatrixManifold &operator=(SparseMatrixManifold &&that) {
    base_t::operator=(std::move(that));
    return *this;
  }
  SparseMatrixManifold(const SparseMatrixManifold &_in) : base_t(_in) {}
  SparseMatrixManifold(SparseMatrixManifold &&_in) : base_t(std::move(_in)) {}
  virtual ~SparseMatrixManifold() = default;
};

template <long Rows> using LinearManifold = MatrixManifold<Rows, 1>;

using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;
using R4 = MatrixManifold<4, 1>;
using R5 = MatrixManifold<5, 1>;
using R6 = MatrixManifold<6, 1>;
using R7 = MatrixManifold<7, 1>;
} // namespace manifolds
