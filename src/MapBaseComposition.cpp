#include <Manifolds/MapBaseComposition.hpp>
namespace manifolds {

MapBaseComposition::MapBaseComposition(const MapBaseComposition &_that) {
  for (const auto &map : _that.maps_) {
    maps_.push_back(map->clone());
    codomain_buffers_.push_back(map->codomain_buffer());
    matrix_buffers_.push_back(map->linearization_buffer());
  }
}
MapBaseComposition::MapBaseComposition(MapBaseComposition &&_that) {
  for (const auto &map : _that.maps_) {
    codomain_buffers_.push_back(map->codomain_buffer());
    matrix_buffers_.push_back(map->linearization_buffer());
    maps_.push_back(map->move_clone());
  }
}

MapBaseComposition::MapBaseComposition(const MapBase &_in)
    : MapInheritanceHelper(), maps_() {
  maps_.push_back(_in.clone());
  codomain_buffers_.push_back(_in.codomain_buffer());
  matrix_buffers_.push_back(_in.linearization_buffer());
}

MapBaseComposition::MapBaseComposition(MapBase &&_in)
    : MapInheritanceHelper(), maps_() {
  codomain_buffers_.push_back(_in.codomain_buffer());
  matrix_buffers_.push_back(_in.linearization_buffer());
  maps_.push_back(_in.move_clone());
}

MapBaseComposition::MapBaseComposition(const std::vector<MapBase> &_in)
    : MapInheritanceHelper(), maps_() {
  std::vector<MapBase> res;
  for (const auto &map : _in) {
    maps_.push_back(map.clone());
    matrix_buffers_.push_back(map.linearization_buffer());
    codomain_buffers_.push_back(map.codomain_buffer());
  }
}

MapBaseComposition::MapBaseComposition(std::vector<MapBase> &&_in)
    : MapInheritanceHelper(), maps_() {
  for (auto &map : _in) {
    codomain_buffers_.push_back(map.codomain_buffer());
    matrix_buffers_.push_back(map.linearization_buffer());
    maps_.push_back(map.move_clone());
  }
}

bool MapBaseComposition::value_impl(const ManifoldBase *_in,
                                    ManifoldBase *_other) const {

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

void MapBaseComposition::append(const MapBase &_in) {
  maps_.push_back(_in.clone());
}
void MapBaseComposition::append(MapBase &&_in) {
  maps_.push_back(_in.move_clone());
}

bool MapBaseComposition::diff_impl(const ManifoldBase *_in,
                                   Eigen::MatrixXd &_mat) const {
  auto map_it = maps_.end();
  auto codomain_it = codomain_buffers_.end();
  auto matrix_it = matrix_buffers_.end();
  auto domain_it = codomain_buffers_.end();
  std::prev(map_it, 1);
  std::prev(codomain_it, 1);
  std::prev(domain_it, 1);
  std::prev(matrix_it, 1);

  (*map_it)->value_impl(_in, codomain_it->get());
  (*map_it)->diff_impl(_in, *matrix_it);

  do {
    --map_it;
    --codomain_it;
    --matrix_it;
    (*map_it)->value(*domain_it, *codomain_it);
    (*map_it)->diff(*domain_it, *matrix_it);
    --domain_it;
  } while (map_it != maps_.begin());

  _mat.noalias() = *matrix_it;
  return true;
}
std::size_t MapBaseComposition::get_dom_dim() const {
  return maps_.back()->get_dom_dim();
}
std::size_t MapBaseComposition::get_codom_dim() const {
  return maps_.front()->get_codom_dim();
}
std::size_t MapBaseComposition::get_dom_tangent_repr_dim() const {
  return maps_.back()->get_dom_dim();
}
std::size_t MapBaseComposition::get_codom_tangent_repr_dim() const {
  return maps_.front()->get_codom_dim();
}

ManifoldBase *MapBaseComposition::domain_buffer_impl() const {
  return maps_.back()->domain_buffer().release();
}
ManifoldBase *MapBaseComposition::codomain_buffer_impl() const {
  return maps_.front()->codomain_buffer().release();
}
} // namespace manifolds
