#pragma once
#include <Manifolds/Manifold.hpp>

namespace manifolds {

template <long Rows, long Cols>
class MatrixManifold : public ManifoldInheritanceHelper<
                           MatrixManifold<Rows, Cols>,
                           Manifold<Eigen::Matrix<double, Rows, Cols>,
                                    Rows * Cols, Rows * Cols, true>> {

public:
  using base_class_t =
      ManifoldInheritanceHelper<MatrixManifold<Rows, Cols>,
                                Manifold<Eigen::Matrix<double, Rows, Cols>,
                                         Rows * Cols, Rows * Cols, true>>;
  using ManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<Eigen::Matrix<double, Rows, Cols>, Rows * Cols, Rows * Cols,
               true>>::ManifoldInheritanceHelper;

  virtual ~MatrixManifold() = default;

  MatrixManifold(const MatrixManifold &_that) : base_class_t(_that) {}
  MatrixManifold(MatrixManifold &&_that) : base_class_t(std::move(_that)) {}

  virtual std::size_t get_dim() const override { return this->crepr().size(); }
  virtual std::size_t get_tanget_repr_dim() const override {
    return this->crepr().size();
  }
  using base_class_t::operator=;
};

template <int N> struct footype { typedef MatrixManifold<N, 1> type; };
template <> struct footype<1> { typedef Manifold<double, 1, 1, true> type; };

using Rn = MatrixManifold<Eigen::Dynamic, 1>;

template <long N = 1> using Reals = typename footype<N>::type;

using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;

} // namespace manifolds
