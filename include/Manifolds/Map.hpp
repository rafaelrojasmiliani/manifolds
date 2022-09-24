#pragma once

#include <Manifolds/Manifold.hpp>
#include <Manifolds/MapBase.hpp>
//#include <Manifolds/MapComposition.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <memory>

namespace manifolds {

template <typename DomainType, typename CoDomainType> class MapComposition;

template <typename DomainType, typename CoDomainType>
class Map : public AbstractMapInheritanceHelper<Map<DomainType, CoDomainType>,
                                                MapBase> {
public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;

  // Default lifecycle
  Map() = default;
  Map(const Map &_that) = default;
  Map(Map &&_that) = default;
  virtual ~Map() = default;

  // behaviur
  CoDomainType value(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  bool value(const DomainType &_in, CoDomainType &_out) const {
    return value_on_repr(_in.crepr(), _out.repr());
  }

  // template<bool IF= not DomainType::is_faithfull>
  CoDomainType operator()(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  bool operator()(const DomainType &_in, CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }
  Eigen::MatrixXd diff(const DomainType &_in) const {
    Eigen::MatrixXd result = this->linearization_buffer();
    diff(_in, result);
    return result;
  }

  bool diff(const DomainType &_in, Eigen::MatrixXd &_out) const {
    return diff_from_repr(_in.crepr(), _out);
  }
  /*
    template <typename OtherDomainType>
    MapComposition<CoDomainType, OtherDomainType>
    compose(const Map<DomainType, OtherDomainType> _in) const {
      return MapComposition<CoDomainType, DomainType>(*this).compose(_in);
    }
    */

  virtual std::size_t get_dom_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::dim;
  }
  virtual std::size_t get_codom_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::dim;
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::tangent_repr_dim;
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::tangent_repr_dim;
  }

private:
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {

    return value_on_repr(static_cast<const DomainType *>(_in)->crepr(),
                         static_cast<CoDomainType *>(_other)->repr());
  }
  bool diff_impl(const ManifoldBase *_in,
                 Eigen::MatrixXd &_mat) const override {

    return diff_from_repr(static_cast<const DomainType *>(_in)->crepr(), _mat);
  }
  virtual Domain_t *domain_buffer_impl() const override {
    return new Domain_t();
  }
  virtual Codomain_t *codomain_buffer_impl() const override {
    return new Codomain_t();
  }

protected:
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const = 0;
  virtual bool diff_from_repr(const typename DomainType::Representation &_in,
                              Eigen::MatrixXd &_mat) const = 0;
};

}; // namespace manifolds
