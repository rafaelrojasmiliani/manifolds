#pragma once

#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/MapBase.hpp>
#include <Manifolds/Maps/MapComposition.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <memory>

namespace manifolds {

template <typename DomainType, typename CoDomainType>
class Map : virtual public MapBase {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>,
                "The codomain must inherit from ManifoldBase");
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>,
                "The domain must intherit from ManifoldBase");

public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;

  using Differential_t =
      Eigen::Matrix<double, Codomain_t::tangent_repr_dimension,
                    Domain_t::tangent_repr_dimension>;

  // Default lifecycle
  Map() = default;
  Map(const Map &_that) = default;
  Map(Map &&_that) = default;
  virtual ~Map() = default;

  // Clone
  std::unique_ptr<Map> clone() const {
    return std::unique_ptr<Map>(clone_impl());
  }

  std::unique_ptr<Map> move_clone() {
    return std::unique_ptr<Map>(move_clone_impl());
  }

  // Defintioon of out value(in)
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType> value(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType>
  value(const typename DomainType::Representation &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  value(const typename DomainType::Representation &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }
  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  value(const DomainType &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }
  // ---------------------------------
  // Defintion of bool value(in, out)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool> value(const DomainType &_in,
                                  CoDomainType &_out) const {
    return value_on_repr(_in.crepr(), _out.repr());
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::Representation &_in,
        CoDomainType &_out) const {
    return value_on_repr(_in, _out.repr());
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::Representation &_in,
        typename CoDomainType::Representation &_out) const {
    return value_on_repr(_in, _out);
  }
  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const DomainType &_in,
        typename CoDomainType::Representation &_out) const {
    return value_on_repr(_in.crepr(), _out);
  }

  // ---------------------------------
  // Defintion of bool operator(in, out)
  // ---------------------------------
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, bool> operator()(const DomainType &_in,
                                       CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }

  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const typename DomainType::Representation &_in,
             CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const typename DomainType::Representation &_in,
             typename CoDomainType::Representation &_out) const {
    return value(_in, _out);
  }

  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const DomainType &_in,
             typename CoDomainType::Representation &_out) const {
    return value(_in, _out);
  }

  // ---------------------------------
  // Defintion of out operator(in)
  // ---------------------------------
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType> operator()(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType>
  operator()(const typename DomainType::Representation &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  operator()(const typename DomainType::Representation &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }

  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  operator()(const DomainType &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }

  // ---------------------------------
  // Defintion of out diff(in)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull)>
  std::enable_if_t<F, typename Eigen::MatrixXd>
  diff(const DomainType &_in) const {
    Eigen::MatrixXd result = this->linearization_buffer();
    diff(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull)>
  std::enable_if_t<F, typename Eigen::MatrixXd>
  diff(const typename DomainType::Representation &_in) const {
    Eigen::MatrixXd result = this->linearization_buffer();
    diff(_in, result);
    return result;
  }

  // ---------------------------------
  // Defintion of bool diff(in, out)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull)>
  std::enable_if_t<F, bool> diff(const DomainType &_in,
                                 Eigen::Ref<Eigen::MatrixXd> _out) const {
    return diff_from_repr(_in.crepr(), _out);
  }
  template <bool F = DomainType::is_faithfull>
  std::enable_if_t<F, bool> diff(const typename DomainType::Representation &_in,
                                 Eigen::Ref<Eigen::MatrixXd> _out) const {
    return diff_from_repr(_in, _out);
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const Map<DomainType, OtherDomainType> &_in) const {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

    auto a = MapComposition<CoDomainType, DomainType>(*this);

    return a.compose(_in);
  }

  virtual std::size_t get_dom_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::dimension;
  }
  virtual std::size_t get_codom_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::dimension;
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::tangent_repr_dimension;
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::tangent_repr_dimension;
  }

protected:
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {

    return value_on_repr(static_cast<const DomainType *>(_in)->crepr(),
                         static_cast<CoDomainType *>(_other)->repr());
  }
  bool diff_impl(const ManifoldBase *_in,
                 Eigen::Ref<Eigen::MatrixXd> _mat) const override {

    return diff_from_repr(static_cast<const DomainType *>(_in)->crepr(), _mat);
  }
  virtual ManifoldBase *domain_buffer_impl() const override {
    return new Domain_t();
  }
  virtual ManifoldBase *codomain_buffer_impl() const override {
    return new Codomain_t();
  }

protected:
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const = 0;
  virtual bool diff_from_repr(const typename DomainType::Representation &_in,
                              Eigen::Ref<Eigen::MatrixXd> &_mat) const = 0;
};

template <typename Set>
class Identity : public MapInheritanceHelper<Identity<Set>, Map<Set, Set>> {
public:
  static_assert(std::is_base_of_v<ManifoldBase, Set>);
  Identity() = default;
  Identity(const Identity &_that) = default;
  Identity(Identity &&_that) = default;

protected:
  bool value_on_repr(const typename Set::Representation &_in,
                     typename Set::Representation &_result) const override {

    _result = _in;
    return true;
  }
  bool diff_from_repr(const typename Set::Representation &,
                      Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    _mat.noalias() = Map<Set, Set>::Differential_t::Identity(
        Set::tangent_repr_dimension, Set::tangent_repr_dimension);
    return true;
  }
};

}; // namespace manifolds
