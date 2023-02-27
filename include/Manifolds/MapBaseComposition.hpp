#pragma once
#include <Manifolds/MapBase.hpp>
namespace manifolds {

class MapBaseComposition : virtual public MapBase {
public:
  MapBaseComposition() = delete;
  MapBaseComposition(const MapBaseComposition &_that);
  MapBaseComposition(MapBaseComposition &&_that);

  MapBaseComposition(const MapBase &_in);

  MapBaseComposition(MapBase &&_in);

  MapBaseComposition(const std::vector<std::unique_ptr<MapBase>> &_in);

  MapBaseComposition(std::vector<std::unique_ptr<MapBase>> &&_in);

  MapBaseComposition(const std::unique_ptr<MapBase> &_in);

  MapBaseComposition(std::unique_ptr<MapBase> &&_in);

  virtual ~MapBaseComposition() = default;

  void append(const MapBase &_in);
  void append(MapBase &&_in);

  std::unique_ptr<MapBaseComposition> clone() const {
    return std::unique_ptr<MapBaseComposition>(
        reinterpret_cast<MapBaseComposition *>(clone_impl()));
  }

  std::unique_ptr<MapBaseComposition> move_clone() {
    return std::unique_ptr<MapBaseComposition>(move_clone_impl());
  }
  std::size_t get_dom_dim() const override;
  std::size_t get_codom_dim() const override;

  std::size_t get_dom_tangent_repr_dim() const override;
  std::size_t get_codom_tangent_repr_dim() const override;

protected:
  std::vector<std::unique_ptr<MapBase>> maps_;
  mutable std::vector<std::unique_ptr<ManifoldBase>> codomain_buffers_;
  mutable std::vector<Eigen::MatrixXd> matrix_buffers_;

  bool value_impl(const ManifoldBase *_in, ManifoldBase *_other) const override;
  bool diff_impl(const ManifoldBase *_in,
                 Eigen::Ref<Eigen::MatrixXd> _mat) const override;

  ManifoldBase *domain_buffer_impl() const override;
  ManifoldBase *codomain_buffer_impl() const override;

  virtual MapBaseComposition *clone_impl() const override {
    return new MapBaseComposition(*this);
  }

  virtual MapBaseComposition *move_clone_impl() override {
    return new MapBaseComposition(std::move(*(this)));
  }
};

} // namespace manifolds
