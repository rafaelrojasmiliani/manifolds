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

template <typename Current, typename Base>
class ManifoldInheritanceHelper : public Base {
private:
  using ThisClass = ManifoldInheritanceHelper<Current, Base>;

public:
  using Base::Base;
  ManifoldInheritanceHelper(const ThisClass &_in) : Base(_in) {}
  ManifoldInheritanceHelper(ThisClass &&_in) : Base(std::move(_in)) {}
  virtual ~ManifoldInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(static_cast<Current *>(clone_impl()));
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(static_cast<Current *>(move_clone_impl()));
  }

  // virtual bool is_faithfull() const = 0;
  using Base::operator=;

protected:
  virtual ThisClass *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual ThisClass *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
};

} // namespace manifolds
