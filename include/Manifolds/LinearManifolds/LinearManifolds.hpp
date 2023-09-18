#pragma once
#include <Manifolds/Atlases/LinearManifolds.h>
#include <Manifolds/Detail.hpp>
#include <Manifolds/Manifold.hpp>

namespace manifolds {

template <long Rows, long Cols, bool F = true>
using DenseMatrixManifold = Manifold<DenseLinearManifoldAtlas<Rows, Cols>, F>;

template <long Rows, long Cols, bool F = true>
using SparseMatrixManifold = Manifold<SparseLinearManifoldAtlas<Rows, Cols>, F>;

template <long Rows, bool F = true>
using DenseLinearManifold = DenseMatrixManifold<Rows, 1, F>;
template <long Rows, bool F = true>
using SparseLinearManifold = SparseMatrixManifold<Rows, 1, F>;

using R2 = DenseLinearManifold<2>;
using R3 = DenseLinearManifold<3>;
using R4 = DenseLinearManifold<4>;
using R5 = DenseLinearManifold<5>;
using R6 = DenseLinearManifold<6>;
using R7 = DenseLinearManifold<7>;

} // namespace manifolds
