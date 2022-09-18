#pragma once
#include <Manifolds/Manifold.hpp>
#include <memory>
#include <variant>

namespace manifolds {

class MapBaseComposition;
class MapBase {
  friend class MapBaseComposition;

private:
  virtual bool value_impl(const ManifoldBase *_in,
                          ManifoldBase *_other) const = 0;
  virtual bool diff_impl(const ManifoldBase *_in,
                         Eigen::MatrixXd &_mat) const = 0;

public:
  // Default lifecycle
  MapBase(const MapBase &) = default;
  MapBase(MapBase &&) = default;
  MapBase() = default;
  virtual ~MapBase() = default;
  // value returns
  std::unique_ptr<ManifoldBase>
  value(const std::unique_ptr<ManifoldBase> &_in) const {
    std::unique_ptr<ManifoldBase> result = codomain_buffer();
    value_impl(_in.get(), result.get());
    return result;
  }

  // value gets a buffer
  bool value(const std::unique_ptr<ManifoldBase> &_in,
             std::unique_ptr<ManifoldBase> &_other) const {
    return value_impl(_in.get(), _other.get());
  }

  // other stuff
  std::unique_ptr<MapBase> clone() const {
    return std::unique_ptr<MapBase>(clone_impl());
  }

  std::unique_ptr<MapBase> move_clone() {
    return std::unique_ptr<MapBase>(move_clone_impl());
  }

  bool diff(const std::unique_ptr<ManifoldBase> &_in,
            Eigen::MatrixXd &_mat) const {
    return diff_impl(_in.get(), _mat);
  }

  // return manifold buffers
  virtual std::unique_ptr<ManifoldBase> codomain_buffer() const {
    return std::unique_ptr<ManifoldBase>(codomain_buffer_impl());
  }
  std::unique_ptr<ManifoldBase> domain_buffer() const {
    return std::unique_ptr<ManifoldBase>(domain_buffer_impl());
  }

protected:
  virtual MapBase *clone_impl() const = 0;
  virtual MapBase *move_clone_impl() = 0;
  virtual ManifoldBase *domain_buffer_impl() const = 0;
  virtual ManifoldBase *codomain_buffer_impl() const = 0;
};

template <typename Current, typename Base>
class MapInheritanceHelper : public Base {
public:
  using Base::Base;
  virtual ~MapInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

protected:
  virtual MapBase *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual MapBase *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
};

} // namespace manifolds
