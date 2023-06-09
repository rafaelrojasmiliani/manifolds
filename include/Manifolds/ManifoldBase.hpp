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

  virtual bool has_value() const = 0;
  virtual void assign(const std::unique_ptr<ManifoldBase> &_other) = 0;

  template <typename T> bool is_same() const {
    return dynamic_cast<const std::decay_t<T> *>(this) != nullptr;
  }

  virtual bool is_equal(const std::unique_ptr<ManifoldBase> &_other) const = 0;

protected:
  virtual ManifoldBase *clone_impl() const = 0;
  virtual ManifoldBase *move_clone_impl() = 0;
};

} // namespace manifolds
