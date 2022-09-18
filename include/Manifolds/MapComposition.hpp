#pragma once
#include <Manifolds/MapBase.hpp>
namespace manifolds {

class MapBaseComposition : public MapBase {
public:
  MapBaseComposition() = delete;
  MapBaseComposition(const MapBaseComposition &_that) {
    for (const auto &map : _that.maps_)
      maps_.push_back(map->clone());
  }
  MapBaseComposition(MapBaseComposition &&_that) {
    for (const auto &map : _that.maps_)
      maps_.push_back(map->move_clone());
  }

  MapBaseComposition(const MapBase &_in) : MapBase(), maps_() {
    maps_.push_back(_in.clone());
  }

  MapBaseComposition(MapBase &&_in) : MapBase(), maps_() {
    maps_.push_back(_in.move_clone());
  }

  MapBaseComposition(const std::vector<MapBase> &_in) : MapBase(), maps_() {
    for (const auto &map : _in)
      maps_.push_back(map.clone());
  }

  MapBaseComposition(std::vector<MapBase> &&_in) : MapBase(), maps_() {
    for (auto &map : _in)
      maps_.push_back(map.move_clone());
  }

  virtual ~MapBaseComposition() = default;

  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {

    auto map_it = maps_.end();
    auto codomain_it = codomain_buffers_.end();
    auto domain_it = codomain_buffers_.end();
    std::prev(map_it, 1);
    std::prev(codomain_it, 1);
    std::prev(domain_it, 1);
    (*map_it)->value_impl(_in, codomain_it->get());

    do {
      --map_it;
      --codomain_it;
      (*map_it)->value(*domain_it, *codomain_it);
      --domain_it;
    } while (map_it != maps_.begin());

    _other->assign(*codomain_it);
    return true;
  }

  void append(const MapBase &_in) { maps_.push_back(_in.clone()); }
  void append(MapBase &&_in) { maps_.push_back(_in.move_clone()); }

  Eigen::MatrixXd diff(const ManifoldBase &) const override {
    return Eigen::MatrixXd::Identity(4, 4);
  }

protected:
  std::vector<std::unique_ptr<MapBase>> maps_;
  mutable std::vector<std::unique_ptr<ManifoldBase>> codomain_buffers_;
};

} // namespace manifolds
