#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <functional>

namespace manifolds {

template <std::size_t N>
class CanonicPolynomial
    : public ManifoldInheritanceHelper<CanonicPolynomial<N>, LinearManifold<N>>,
      public MapInheritanceHelper<CanonicPolynomial<N>, Map<Reals, Reals>> {
  template <typename T, typename U> friend class Map;
  template <typename T, typename U> friend class MapComposition;

  using base_t =
      ManifoldInheritanceHelper<CanonicPolynomial<N>, LinearManifold<N>>;

public:
  using base_t::base_t;
  bool value_on_repr(const double &_in, double &_result) const override {
    int j;

    const auto &c = this->crepr();
    double p = c[j = this->crepr().size() - 1];
    while (j > 0)
      p = p * _in + c[--j];
    _result = p;
    return true;
  }
  virtual bool
  diff_from_repr(const double &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {

    int j = this->crepr().size() - 1;
    const auto &c = this->crepr();
    double p = j * c[j];
    j--;
    while (j > 1) {
      p = p * _in + j * c[j];
      j--;
    }
    _mat(0, 0) = p;
    return true;
  }
};

} // namespace manifolds
