#pragma once
#include <Manifolds/Atlases/Reals.h>
#include <Manifolds/Detail.hpp>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <functional>

namespace manifolds {

class Reals
    : public LinearManifoldInheritanceHelper<Reals,
                                             Manifold<RealsAtlas, true>> {
public:
  using base_t =
      LinearManifoldInheritanceHelper<Reals, Manifold<RealsAtlas, true>>;
  using Representation = typename base_t::Representation;
  using base_t::base_t;
  using base_t::operator=;
  __DEFAULT_LIVE_CYCLE(Reals)

  class Lifting;
};

/* class Reals::Lifting */
/*     : public MapInheritanceHelper<Reals::Lifting, Map<Reals, Reals, false>> {
 */
/* public: */
/*   Lifting(const std::function<double(double)> &_value, */
/*           const std::function<double(double)> &_derivative) */
/*       : value_(_value), derivative_(_derivative) {} */

/* private: */
/*   std::function<double(double)> value_; */
/*   std::function<double(double)> derivative_; */

/*   bool value_on_repr(const double &_in, double &_result) const override { */
/*     _result = value_(_in); */
/*     return true; */
/*   } */
/*   virtual bool diff_from_repr( */
/*       const double &_in, */
/*       detail::DifferentialReprRef_t<false, 1, 1> _mat) const override { */
/*     _mat(0, 0) = derivative_(_in); */
/*     return true; */
/*   } */
/* }; */

} // namespace manifolds
