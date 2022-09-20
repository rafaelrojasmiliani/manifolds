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
class Map : public MapBase {
public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;

  // Default lifecycle
  Map() = default;
  Map(const Map &_that) = default;
  Map(Map &&_that) = default;
  virtual ~Map() = default;

  // Default clone
  std::unique_ptr<Map<DomainType, CoDomainType>> clone() const {
    return std::unique_ptr<MapBase>(clone_impl());
  }

  std::unique_ptr<Map<DomainType, CoDomainType>> move_clone() {
    return std::unique_ptr<MapBase>(move_clone_impl());
  }
  // behaviur
  CoDomainType value(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  bool value(const DomainType &_in, CoDomainType &_out) const {
    return value_on_repr(_in.crepr(), _out.repr());
  }

  CoDomainType operator()(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  bool operator()(const DomainType &_in, CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }
  /*
    template <typename OtherDomainType>
    MapComposition<CoDomainType, OtherDomainType>
    compose(const Map<DomainType, OtherDomainType> _in) const {
      return MapComposition<CoDomainType, DomainType>(*this).compose(_in);
    }
    */

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
  Domain_t *domain_buffer_impl() const override { return new Domain_t(); }
  Codomain_t *codomain_buffer_impl() const override { return new Codomain_t(); }

protected:
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const = 0;
  virtual bool diff_from_repr(const typename DomainType::Representation &_in,
                              Eigen::MatrixXd &_mat) const = 0;
};
/*
template <typename DomainType, typename CoDomainType>
class MapComposition
    : public MapInheritanceHelper<MapComposition<DomainType, CoDomainType>,
                                  Map<DomainType, CoDomainType>>,
      public MapBaseComposition {
public:
  using MapBaseComposition::MapBaseComposition;
  virtual ~MapComposition() = default;

  bool value(const DomainType &_in) const {
    return std::unique_ptr<CoDomainType>(static_cast<CoDomainType *>(
        this->value_ptr_impl(static_cast<const ManifoldBase &>(_in))));
  }
  virtual bool value(const DomainType &_in, const CoDomainType &_result) {}

  CoDomainType operator()(const DomainType &_in) const {
    return *static_cast<CoDomainType *>(
        this->value_ptr_impl(static_cast<const ManifoldBase &>(_in)));
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const MapComposition<DomainType, OtherDomainType> &_in) const & {
    MapComposition<CoDomainType, OtherDomainType> result(*this);
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(MapComposition<DomainType, OtherDomainType> &&_in) const & {
    MapComposition<CoDomainType, OtherDomainType> result(*this);
    result.append(std::move(_in));
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const MapComposition<DomainType, OtherDomainType> &_in) && {
    MapComposition<CoDomainType, OtherDomainType> result(std::move(*this));
    result.append(_in);
    return result;
  }
  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(MapComposition<DomainType, OtherDomainType> &&_in) && {
    MapComposition<CoDomainType, OtherDomainType> result(std::move(*this));
    result.append(std::move(_in));
    return result;
  }
};
*/
//  virtual Chart<M> inverse() const = 0;
}; // namespace manifolds
