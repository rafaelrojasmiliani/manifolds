#pragma once
#include <Eigen/Core>
#include <memory>
namespace manifolds {

class MapBase;
class MapBaseComposition;

class ManifoldBase {
private:
  friend class MapBase;
  friend class MapBaseComposition;
  virtual void assign(const std::unique_ptr<ManifoldBase> &_other) = 0;

public:
  using Representation = void *;
  std::unique_ptr<ManifoldBase> clone() const {
    return std::unique_ptr<ManifoldBase>(clone_impl());
  }

  std::unique_ptr<ManifoldBase> move_clone() {
    return std::unique_ptr<ManifoldBase>(move_clone_impl());
  }

  virtual std::size_t get_dim() const = 0;
  virtual std::size_t get_tanget_repr_dim() const = 0;

  virtual ~ManifoldBase() = default;
  ManifoldBase(const ManifoldBase &) = default;
  ManifoldBase(ManifoldBase &&) = default;
  ManifoldBase() = default;

protected:
  virtual ManifoldBase *clone_impl() const = 0;
  virtual ManifoldBase *move_clone_impl() = 0;
};

} // namespace manifolds
