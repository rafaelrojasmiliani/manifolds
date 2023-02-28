#pragma once
////#include <Manifolds/Map.hpp>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/MapBaseComposition.hpp>

namespace manifolds {

template <typename DomainType, typename CoDomainType> class Map;

template <typename DomainType, typename CoDomainType>
class MapComposition : public virtual Map<DomainType, CoDomainType>,
                       public virtual MapBaseComposition {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>);
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {
    return MapBaseComposition::value_impl(_in, _other);
  }
  bool diff_impl(const ManifoldBase *_in,
                 Eigen::Ref<Eigen::MatrixXd> _mat) const override {
    return MapBaseComposition::diff_impl(_in, _mat);
  }

public:
  std::size_t get_dom_dim() const override {
    return MapBaseComposition::get_dom_dim();
  }
  std::size_t get_codom_dim() const override {
    return MapBaseComposition::get_codom_dim();
  }

  std::size_t get_dom_tangent_repr_dim() const override {
    return MapBaseComposition::get_dom_tangent_repr_dim();
  }
  std::size_t get_codom_tangent_repr_dim() const override {
    return MapBaseComposition::get_codom_tangent_repr_dim();
  }

  ManifoldBase *domain_buffer_impl() const override {
    return Map<DomainType, CoDomainType>::domain_buffer_impl();
  }
  ManifoldBase *codomain_buffer_impl() const override {
    return Map<DomainType, CoDomainType>::codomain_buffer_impl();
  }

  MapComposition(const Map<DomainType, CoDomainType> &m)
      : MapBaseComposition(m) {}
  virtual ~MapComposition() = default;

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const Map<DomainType, OtherDomainType> &_in) const & {
    MapComposition<DomainType, CoDomainType> result(*this);
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

protected:
  virtual MapComposition *clone_impl() const override {
    return new MapComposition(*this);
  }

  virtual MapComposition *move_clone_impl() override {
    return new MapComposition(std::move(*(this)));
  }
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const override {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);
    auto m1 = DomainType::Ref(_in);
    auto m2 = CoDomainType::Ref(_result);
    return value_impl(&m1, &m2);
  }
  virtual bool
  diff_from_repr(const typename DomainType::Representation &_in,
                 Eigen::Ref<Eigen::MatrixXd> &_mat) const override {
    DomainType m1(_in);
    return diff_impl(&m1, _mat);
  }
  MapComposition() = default;
};
} // namespace manifolds
