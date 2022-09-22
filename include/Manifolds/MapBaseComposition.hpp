#pragma once
#include <Manifolds/MapBase.hpp>
namespace manifolds {

class MapBaseComposition
    : public MapInheritanceHelper<MapBaseComposition, MapBase> {
public:
  MapBaseComposition() = delete;
  MapBaseComposition(const MapBaseComposition &_that);
  MapBaseComposition(MapBaseComposition &&_that);

  MapBaseComposition(const MapBase &_in);

  MapBaseComposition(MapBase &&_in);

  MapBaseComposition(const std::vector<MapBase> &_in);

  MapBaseComposition(std::vector<MapBase> &&_in);

  virtual ~MapBaseComposition() = default;

  void append(const MapBase &_in);
  void append(MapBase &&_in);

protected:
  std::vector<std::unique_ptr<MapBase>> maps_;
  mutable std::vector<std::unique_ptr<ManifoldBase>> codomain_buffers_;
  mutable std::vector<Eigen::MatrixXd> matrix_buffers_;

private:
  bool value_impl(const ManifoldBase *_in, ManifoldBase *_other) const override;
  bool diff_impl(const ManifoldBase *_in, Eigen::MatrixXd &_mat) const override;

  std::size_t get_dom_dim() const override;
  std::size_t get_codom_dim() const override;

  std::size_t get_dom_tangent_repr_dim() const override;
  std::size_t get_codom_tangent_repr_dim() const override;

  ManifoldBase *domain_buffer_impl() const override;
  ManifoldBase *codomain_buffer_impl() const override;
};

} // namespace manifolds
