
#pragma once
#include <Eigen/Core>
#include <Manifolds/ManifoldBase.hpp>
#include <memory>
namespace manifolds {

template <typename R, long Dim, long TDim, bool Faithfull = false>
class Manifold
    : public ManifoldInheritanceHelper<Manifold<R, Dim, TDim>, ManifoldBase> {
  template <typename T, typename U> friend class Map;

protected:
  R representation_;

public:
  // Generic constructor
  template <typename... Ts>
  Manifold(Ts &&... args) : representation_(std::forward<Ts>(args)...) {}
  // Static const calues
  static const long dim = Dim;
  static const long tangent_repr_dim = TDim;
  static const bool is_faithfull = Faithfull;
  typedef R Representation;
  // Get const referece to the representtion
  const std::decay_t<R> &crepr() const { return representation_; };
  // Dynamically get the dimension of manifold and its tanget representation
  virtual std::size_t get_dim() const override { return dim; }
  virtual std::size_t get_tanget_repr_dim() const override {
    return tangent_repr_dim;
  }

  // SNIFAE for Faithfull Manifolds
  template <bool F = Faithfull>
  operator const std::enable_if_t<F, R> &() const & {
    return representation_;
  }

  template <bool F = Faithfull> operator std::enable_if_t<F, R>() & {
    return representation_;
  }

  template <bool F = Faithfull> operator std::enable_if_t<F, R> &&() && {
    return std::move(representation_);
  }

  /*
   // Signment operator
   const Manifold &operator=(const Manifold<R, Dim, TDim, Faithfull> &_other) {
     representation_ = _other.crepr();
   }
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<R, Dim, TDim, Faithfull>> &
  operator=(const std::decay_t<R> &_other) {
    representation_ = _other;
  }
 */

private:
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    representation_ =
        static_cast<Manifold<R, Dim, TDim> *>(_other.get())->crepr();
  }
  std::decay_t<R> &repr() { return representation_; };
};

} // namespace manifolds
