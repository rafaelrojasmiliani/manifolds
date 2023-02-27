#pragma once
#include <Eigen/Core>
#include <Manifolds/ManifoldBase.hpp>
#include <memory>
#include <type_traits>
namespace manifolds {

template <typename Atlas, bool Faithfull = false>
class Manifold : public ManifoldInheritanceHelper<Manifold<Atlas, Faithfull>,
                                                  ManifoldBase>,
                 public Atlas {
  template <typename T, typename U> friend class Map;
  template <typename T, typename U> friend class MapComposition;

  using base_t =
      ManifoldInheritanceHelper<Manifold<Atlas, Faithfull>, ManifoldBase>;

public:
  using Representation = std::decay_t<typename Atlas::Representation>;

protected:
  Representation representation_;

private:
public:
  // Generic livecycle
  Manifold() : base_t(), representation_() {}
  Manifold(const Manifold &that)
      : base_t(that), representation_(that.representation_) {}
  Manifold(Manifold &&that)
      : base_t(std::move(that)),
        representation_(std::move(that.representation_)) {}
  virtual ~Manifold() = default;
  // Static const calues
  static const bool is_faithfull = Faithfull;
  // Get const referece to the representtion
  const Representation &crepr() const { return representation_; };

  // SNIFAE for Faithfull Manifolds
  template <bool F = Faithfull>
  operator const std::enable_if_t<F, Representation> &() const & {
    return representation_;
  }

  template <bool F = Faithfull>
  operator std::enable_if_t<F, Representation> &() & {
    return representation_;
  }

  template <bool F = Faithfull>
  operator std::enable_if_t<F, Representation> &&() && {
    return std::move(representation_);
  }

  // Assignment with manifold operator
  const Manifold &operator=(const Manifold &_other) {
    representation_ = _other.crepr();
    return *this;
  }
  // Assignment with manifold operator
  const Manifold &operator=(Manifold &&_other) {
    representation_ = std::move(_other.representation_);
    return *this;
  }

  // Assignment wirth representation
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(const Representation &_other) {
    representation_ = _other;
    return *this;
  }

  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(Representation &&_other) {
    representation_ = std::move(_other);
    return *this;
  }
  std::size_t get_dim() const override { return Atlas::dimension; }
  std::size_t get_tanget_repr_dim() const override {
    return Atlas::tangent_repr_dimension;
  }

  /// SNIFAE of constructors: Make faithfull public
  template <bool F = Faithfull>
  Manifold(const std::enable_if_t<F, Representation> &_in)
      : representation_(_in) {}

  template <bool F = Faithfull>
  Manifold(const std::enable_if_t<F, Representation> &&_in)
      : representation_(_in) {}

private:
  /// SNIFAE of constructors: Make non faithfull private
  template <bool F = Faithfull>
  Manifold(const std::enable_if_t<not F, Representation> &_in)
      : representation_(_in) {}

  template <bool F = Faithfull>
  Manifold(std::enable_if_t<not F, Representation> &&_in)
      : representation_(std::move(_in)) {}

  /// override assign but private
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    representation_ =
        static_cast<Manifold<Atlas, Faithfull> *>(_other.get())->crepr();
  }
  Representation &repr() { return representation_; };
};

} // namespace manifolds
