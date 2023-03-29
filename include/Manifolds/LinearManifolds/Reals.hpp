#pragma once
#include <Manifolds/Atlases/Reals.h>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <functional>

namespace manifolds {

class Reals : public Manifold<RealsAtlas, true> {
  template <typename T, typename U> friend class Map;
  template <typename T, typename U> friend class MapComposition;

public:
  using Manifold<RealsAtlas, true>::Manifold;
  class Lifting;
};

class Reals::Lifting
    : public MapInheritanceHelper<Reals::Lifting, Map<Reals, Reals>> {
public:
  Lifting(const std::function<double(double)> &_value,
          const std::function<double(double)> &_derivative)
      : value_(_value), derivative_(_derivative) {}

private:
  std::function<double(double)> value_;
  std::function<double(double)> derivative_;

  bool value_on_repr(const double &_in, double &_result) const override {
    _result = value_(_in);
    return true;
  }
  virtual bool diff_from_repr(const double &_in,
                              DifferentialReprRefType _mat) const override {
    std::get<0>(_mat)(0, 0) = derivative_(_in);
    return true;
  }
};

} // namespace manifolds
