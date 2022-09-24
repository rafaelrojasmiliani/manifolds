#pragma once
#include <Manifolds/Manifold.hpp>

namespace manifolds {

template <long Rows, long Cols>
class MatrixManifold : public ManifoldInheritanceHelper<
                           MatrixManifold<Rows, Cols>,
                           Manifold<Eigen::Matrix<double, Rows, Cols>,
                                    Rows * Cols, Rows * Cols, true>> {

public:
  using ManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<Eigen::Matrix<double, Rows, Cols>, Rows * Cols, Rows * Cols,
               true>>::ManifoldInheritanceHelper;

  virtual std::size_t get_dim() const override { return this->crepr().size(); }
  virtual std::size_t get_tanget_repr_dim() const override {
    return this->crepr().size();
  }
};

using Rn = MatrixManifold<Eigen::Dynamic, 1>;
using R = MatrixManifold<1, 1>;
using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;

} // namespace manifolds
