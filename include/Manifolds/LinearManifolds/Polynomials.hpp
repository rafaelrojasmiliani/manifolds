#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <functional>

namespace manifolds {

template <std::size_t Dim>
class CanonicPolynomial
    : public ManifoldInheritanceHelper<CanonicPolynomial<Dim>,
                                       LinearManifold<Dim>>,
      public MapInheritanceHelper<CanonicPolynomial<Dim>,
                                  Map<Reals, Reals, false>> {
  template <typename T, typename U, bool is> friend class Map;
  template <typename T, typename U, bool is> friend class MapComposition;

  using base_t =
      ManifoldInheritanceHelper<CanonicPolynomial<Dim>, LinearManifold<Dim>>;

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
  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {

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

template <std::size_t Dim, std::size_t N>
class PWCanonicPolynomial
    : public ManifoldInheritanceHelper<PWCanonicPolynomial<Dim, N>,
                                       LinearManifold<Dim * N>>,
      public MapInheritanceHelper<PWCanonicPolynomial<Dim, N>,
                                  Map<Reals, Reals, false>> {
  template <typename T, typename U, bool is> friend class Map;
  template <typename T, typename U, bool is> friend class MapComposition;

  using base_t = ManifoldInheritanceHelper<PWCanonicPolynomial<Dim, N>,
                                           LinearManifold<Dim>>;

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

  virtual bool diff_from_repr(const double &_in,
                              Eigen::Ref<Eigen::MatrixXd> _mat) const override {

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
