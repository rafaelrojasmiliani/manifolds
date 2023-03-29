#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
namespace manifolds {

template <std::size_t DomainDim, std::size_t CodomainDim>
class LinearMap
    : public LinearManifoldInheritanceHelper<
          LinearMap<DomainDim, CodomainDim>,
          MatrixManifold<CodomainDim, DomainDim>>,
      public MapInheritanceHelper<
          LinearMap<DomainDim, CodomainDim>,
          Map<LinearManifold<DomainDim>, LinearManifold<CodomainDim>>> {
public:
  using base_t =
      LinearManifoldInheritanceHelper<LinearMap<DomainDim, CodomainDim>,
                                      MatrixManifold<CodomainDim, DomainDim>>;

  using Representation = typename base_t::Representation;
  using DomainRepresentation =
      typename LinearManifold<DomainDim>::Representation;
  using CodomainRepresentation =
      typename LinearManifold<CodomainDim>::Representation;

  using base_t::base_t;
  using base_t::operator=;
  const LinearMap &operator=(const LinearMap &that) {
    base_t::operator=(that);
    return *this;
  }
  const LinearMap &operator=(LinearMap &&that) {
    base_t::operator=(std::move(that));
    return *this;
  }

  LinearMap(const LinearMap &_in) : base_t(_in) {}

  LinearMap(LinearMap &&_in) : base_t(std::move(_in)) {}

  virtual bool
  value_on_repr(const Eigen::Matrix<double, DomainDim, 1> &in,
                Eigen::Matrix<double, DomainDim, 1> &out) const override {

    out = this->crepr() * in;
    return true;
  }

  virtual bool diff_from_repr(const Eigen::Matrix<double, DomainDim, 1> &,
                              DifferentialReprRefType _mat) const override {

    std::get<0>(_mat) = this->crepr();
    return true;
  }
  virtual ~LinearMap() = default;

  // ----------------------------------------------------
  // -------------------  operators  --------------------
  // ----------------------------------------------------

  inline auto operator*(const Representation &that) const & {
    return base_t::crepr() * that;
  }

  inline auto operator*(const Representation &that) && {
    return std::move(base_t::mrepr()) * that;
  }

  inline auto operator*(Representation &&that) const & {
    return base_t::crepr() * std::move(that);
  }
  inline auto operator*(Representation &&that) && {
    return std::move(base_t::mrepr()) * std::move(that);
  }

  inline void operator*=(const Representation &that) { base_t::repr() *= that; }

  inline void operator*=(Representation &&that) { base_t::repr() *= that; }

  inline auto operator*(const DomainRepresentation &that) {
    return base_t::crepr() * that;
  }
  inline auto operator*(DomainRepresentation &&that) {
    return base_t::crepr() * std::move(that);
  }

  template <std::size_t OtherDomainDim>
  LinearMap<DomainDim, OtherDomainDim>
  compose(const LinearMap<DomainDim, OtherDomainDim> &_in) const {
    return LinearMap<DomainDim, OtherDomainDim>(base_t::crepr() * _in.crepr());
  }
};

using End3 = LinearMap<3, 3>;
} // namespace manifolds
