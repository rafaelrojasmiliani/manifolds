#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <type_traits>
namespace manifolds {

template <typename Domain, typename Codomain>
class LinearMap : public LinearManifoldInheritanceHelper<
                      LinearMap<Domain, Codomain>,
                      MatrixManifold<Codomain::dimension, Domain::dimension>>,
                  public MapInheritanceHelper<LinearMap<Domain, Codomain>,
                                              Map<Domain, Codomain, false>> {

  // ------------------------
  // --- Static assertions --
  // ------------------------
  static_assert(std::is_base_of_v<LinearManifold<Domain::dimension>, Domain>,
                "Linear map must act between vector spaces");

  static_assert(
      std::is_base_of_v<LinearManifold<Codomain::dimension>, Codomain>,
      "Linear map must act between vector spaces");

public:
  using base_t = LinearManifoldInheritanceHelper<
      LinearMap<Domain, Codomain>,
      MatrixManifold<Codomain::dimension, Domain::dimension>>;

  using Representation = typename base_t::Representation;
  using DomainRepresentation = typename Domain::Representation;
  using CodomainRepresentation = typename Codomain::Representation;

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

  virtual bool value_on_repr(
      const Eigen::Matrix<double, Domain::dimension, 1> &in,
      Eigen::Matrix<double, Codomain::dimension, 1> &out) const override {

    out = this->crepr() * in;
    return true;
  }

  virtual bool
  diff_from_repr(const Eigen::Matrix<double, Domain::dimension, 1> &,
                 detail::DifferentialReprRef_t<false> _mat) const override {

    _mat = this->crepr();
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

  template <typename OtherDomain>
  LinearMap<Codomain, OtherDomain>
  compose(const LinearMap<OtherDomain, Codomain> &_in) const {
    return LinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }
};

using End3 = LinearMap<R3, R3>;

template <typename Domain, typename Codomain>
class SparseLinearMap
    : public LinearManifoldInheritanceHelper<
          SparseLinearMap<Domain, Codomain>,
          SparseMatrixManifold<Codomain::dimension, Domain::dimension>>,
      public MapInheritanceHelper<SparseLinearMap<Domain, Codomain>,
                                  Map<Domain, Codomain, false>> {

  // ------------------------
  // --- Static assertions --
  // ------------------------
  static_assert(std::is_base_of_v<LinearManifold<Domain::dimension>, Domain>,
                "Linear map must act between vector spaces");

  static_assert(
      std::is_base_of_v<LinearManifold<Codomain::dimension>, Codomain>,
      "Linear map must act between vector spaces");

public:
  using base_t = LinearManifoldInheritanceHelper<
      SparseLinearMap<Domain, Codomain>,
      SparseMatrixManifold<Codomain::dimension, Domain::dimension>>;

  using Representation = typename base_t::Representation;
  using DomainRepresentation = typename Domain::Representation;
  using CodomainRepresentation = typename Codomain::Representation;

  using base_t::base_t;
  using base_t::operator=;
  const SparseLinearMap &operator=(const SparseLinearMap &that) {
    base_t::operator=(that);
    return *this;
  }
  const SparseLinearMap &operator=(SparseLinearMap &&that) {
    base_t::operator=(std::move(that));
    return *this;
  }

  SparseLinearMap(const SparseLinearMap &_in) : base_t(_in) {}

  SparseLinearMap(SparseLinearMap &&_in) : base_t(std::move(_in)) {}

  virtual bool value_on_repr(
      const Eigen::Matrix<double, Domain::dimension, 1> &in,
      Eigen::Matrix<double, Codomain::dimension, 1> &out) const override {

    out = this->crepr() * in;
    return true;
  }

  virtual bool
  diff_from_repr(const Eigen::Matrix<double, Domain::dimension, 1> &,
                 detail::DifferentialReprRef_t<false> _mat) const override {

    _mat = this->crepr();
    return true;
  }
  virtual ~SparseLinearMap() = default;

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

  template <typename OtherDomain>
  SparseLinearMap<Codomain, OtherDomain>
  compose(const SparseLinearMap<OtherDomain, Codomain> &_in) const {
    return SparseLinearMap<Codomain, OtherDomain>(base_t::crepr() *
                                                  _in.crepr());
  }
};
} // namespace manifolds
