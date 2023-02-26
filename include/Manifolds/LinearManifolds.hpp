#pragma once
#include <Manifolds/Atlases/LinearManifolds.h>
#include <Manifolds/Manifold.hpp>

namespace manifolds {

template <long Rows, long Cols>
using MatrixManifold = Manifold<LinearManifoldAtlas<Rows, Cols>, true>;

using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;
using R4 = MatrixManifold<4, 1>;
using R5 = MatrixManifold<5, 1>;
using R6 = MatrixManifold<6, 1>;
using R7 = MatrixManifold<7, 1>;

} // namespace manifolds
