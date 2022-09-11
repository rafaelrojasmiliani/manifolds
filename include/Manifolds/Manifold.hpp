
#pragma once
#include <Manifolds/Topology/Open.hpp>
namespace manifolds {

class ManifoldBase {};

template <typename R, long Dim, typename T, long TDim>
class Manifold : public ManifoldBase {
protected:
  R representation_;

public:
  template <typename... Ts>
  Manifold(Ts &&... args) : representation_(args...) {}
  static const long dim = Dim;
  static const long tangent_repr_dim = TDim;
  typedef R Representation;
  typedef T TangentRepresentation;
  const R &repr() const { return representation_; };
};

} // namespace manifolds
