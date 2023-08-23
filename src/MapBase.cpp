#include <Manifolds/Detail.hpp>
#include <Manifolds/Maps/MapBase.hpp>
namespace manifolds {

std::unique_ptr<ManifoldBase> MapBase::value(const ManifoldBase &_in) const {
  std::unique_ptr<ManifoldBase> result = codomain_buffer();
  value_impl(&_in, result.get());
  return result;
}

MapBase::~MapBase() = default;

// value gets a buffer
bool MapBase::value(const std::unique_ptr<ManifoldBase> &_in,
                    std::unique_ptr<ManifoldBase> &_other) const {
  return value_impl(_in.get(), _other.get());
}

// other stuff
std::unique_ptr<MapBase> MapBase::clone() const {
  return std::unique_ptr<MapBase>(clone_impl());
}

std::unique_ptr<MapBase> MapBase::move_clone() {
  return std::unique_ptr<MapBase>(move_clone_impl());
}

bool MapBase::diff(const ManifoldBase &_in,
                   detail::mixed_matrix_ref_t _mat) const {
  return diff_impl(&_in, nullptr, _mat);
}
detail::mixed_matrix_t MapBase::diff(const ManifoldBase &_in) const {
  auto result = this->mixed_linearization_buffer();
  diff_impl(&_in, nullptr, detail::mixed_matrix_to_ref(result));
  return result;
}

std::unique_ptr<MapBase> MapBase::operator|(const MapBase &_in) const {
  return std::unique_ptr<MapBase>(this->pipe_impl(_in));
}

std::unique_ptr<MapBase> MapBase::operator|(MapBase &&_in) const {
  return std::unique_ptr<MapBase>(this->pipe_move_impl(std::move(_in)));
}

// return manifold buffers
std::unique_ptr<ManifoldBase> MapBase::codomain_buffer() const {
  return std::unique_ptr<ManifoldBase>(codomain_buffer_impl());
}
std::unique_ptr<ManifoldBase> MapBase::domain_buffer() const {
  return std::unique_ptr<ManifoldBase>(domain_buffer_impl());
}

} // namespace manifolds
