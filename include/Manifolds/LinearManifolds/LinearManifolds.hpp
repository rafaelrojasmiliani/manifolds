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
  using Representation = typename Base::Representation;
  using Base::Base;
  LinearManifoldInheritanceHelper(const ThisClass &_in) : Base(_in) {}
  LinearManifoldInheritanceHelper(ThisClass &&_in) : Base(std::move(_in)) {}
  virtual ~LinearManifoldInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

  using Base::operator=;

  ThisClass &operator=(const ThisClass &that) {
    Base::operator=(that);
    return *this;
  }
  ThisClass &operator=(ThisClass &&that) {
    Base::operator=(std::move(that));
    return *this;
  }

  inline auto operator+(const Representation &that) const & {
    return Base::crepr() + that;
  }
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
class MatrixManifold : public LinearManifoldInheritanceHelper<
                           MatrixManifold<Rows, Cols>,
                           Manifold<LinearManifoldAtlas<Rows, Cols>, true>> {
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

template <long Rows> using LinearManifold = MatrixManifold<Rows, 1>;

using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;
using R4 = MatrixManifold<4, 1>;
using R5 = MatrixManifold<5, 1>;
using R6 = MatrixManifold<6, 1>;
using R7 = MatrixManifold<7, 1>;
} // namespace manifolds
