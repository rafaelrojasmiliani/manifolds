#pragma once
#include <Manifolds/Map.hpp>
#include <Manifolds/MapBaseComposition.hpp>
namespace manifolds {

template <typename DomainType, typename CoDomainType>
class MapComposition : public AbstractMapInheritanceHelper<
                           MapComposition<DomainType, CoDomainType>,
                           Map<DomainType, CoDomainType>>,
                       public MapBaseComposition {
public:
  using MapBaseComposition::MapBaseComposition;

  virtual ~MapComposition() = default;

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
} // namespace manifolds
