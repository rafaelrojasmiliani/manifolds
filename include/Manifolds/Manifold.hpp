
#pragma once
#include <Manifolds/Topology/Open.hpp>
namespace manifolds {

template <typename Current, long Dim>
class Manifold : public topology::Open<Manifold<Current, Dim>> {
private:
public:
  static const long dim = Dim;
  Manifold()
      : topology::Open<Manifold<Current, Dim>>(
            [](const Manifold<Current, Dim> &) { return true; }) {}
};

} // namespace manifolds
