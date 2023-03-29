#include <Manifolds/Maps/MapBase.hpp>
namespace manifolds {

std::unique_ptr<ManifoldBase>
MapBase::value(const std::unique_ptr<ManifoldBase> &_in) const {
  std::unique_ptr<ManifoldBase> result = codomain_buffer();
  value_impl(_in.get(), result.get());
  return result;
}

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

bool MapBase::diff(const std::unique_ptr<ManifoldBase> &_in,
                   DifferentialReprRefType _mat) const {
  return diff_impl(_in.get(), _mat);
}

// return manifold buffers
std::unique_ptr<ManifoldBase> MapBase::codomain_buffer() const {
  return std::unique_ptr<ManifoldBase>(codomain_buffer_impl());
}
std::unique_ptr<ManifoldBase> MapBase::domain_buffer() const {
  return std::unique_ptr<ManifoldBase>(domain_buffer_impl());
}

Eigen::MatrixXd MapBase::linearization_buffer() const {
  return Eigen::MatrixXd::Zero(get_codom_tangent_repr_dim(),
                               get_dom_tangent_repr_dim());
}
} // namespace manifolds
