#pragma once
#include <Eigen/Core>
#include <Manifolds/ManifoldBase.hpp>
#include <exception>
#include <memory>
#include <stdexcept>
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
  Representation *representation_;
  const Representation *const_representation_;

private:
  bool onwing_;

public:
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++   Live cycle    +++++++++++++++++++++++++++++++
  // Livecycle Constructor copy, move constructor, assigment and destructor
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Manifold()
      : base_t(), representation_(new Representation()),
        const_representation_(representation_), onwing_(true) {}
  Manifold(const Manifold &that)
      : base_t(that),
        representation_(new Representation(*that.representation_)),
        const_representation_(representation_), onwing_(true) {}
  Manifold(Manifold &&that)
      : base_t(std::move(that)),
        representation_(new Representation(std::move(*that.representation_))),
        const_representation_(representation_), onwing_(true) {}

  virtual ~Manifold() {
    if (onwing_)
      delete representation_;
    representation_ = nullptr;
    const_representation_ = nullptr;
  }

  // Assignment with manifold operator
  const Manifold &operator=(const Manifold &_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = _other.crepr();
    return *this;
  }
  // Assignment with manifold operator
  const Manifold &operator=(Manifold &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = std::move(*_other.representation_);
    return *this;
  }

  // Assignment with representation
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(const Representation &_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = _other;
    return *this;
  }

  // Assignment with representation
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(Representation &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = std::move(_other);
    return *this;
  }

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++   Static const +++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  static constexpr bool is_faithfull = Faithfull;

  // Get a manifold representation of a piece of data
  static constexpr Manifold Ref(Representation *in) { return Manifold(in); }
  // Get a manifold representation of a piece of data
  static constexpr Manifold Ref(Representation &in) { return Manifold(&in); }
  // Get a manifold representation of a piece of data
  static constexpr Manifold Ref(const Representation *in) {
    return Manifold(in);
  }
  // Get a manifold representation of a piece of data
  static constexpr Manifold Ref(const Representation &in) {
    return Manifold(&in);
  }

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++ Casting  and implicid Cast consturcotrs  +++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  template <bool F = Faithfull>
  constexpr operator const std::enable_if_t<F, Representation> &() const & {
    return *const_representation_;
  }

  template <bool F = Faithfull>
  constexpr operator std::enable_if_t<F, Representation> &() & {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    return *representation_;
  }

  template <bool F = Faithfull>
  constexpr operator std::enable_if_t<F, Representation> &&() && {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    return std::move(*representation_);
  }

  template <bool F = Faithfull>
  constexpr Manifold(const std::enable_if_t<F, Representation> &_in)
      : base_t(), representation_(new Representation(_in)),
        const_representation_(representation_), onwing_(true) {}

  template <bool F = Faithfull>
  constexpr Manifold(const std::enable_if_t<F, Representation> &&_in)
      : base_t(), representation_(new Representation(std::move(_in))),
        const_representation_(representation_), onwing_(true) {}

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++ Getterns ++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  /// Get the dimension of the manifold
  std::size_t get_dim() const override { return Atlas::dimension; }
  /// Get the dimension of the representatino of the tangent space
  std::size_t get_tanget_repr_dim() const override {
    return Atlas::tangent_repr_dimension;
  }
  /// Get const referece to the representtion
  const Representation &crepr() const { return *const_representation_; };

private:
  /// SNIFAE of constructors: Make non faithfull private
  template <bool F = Faithfull>
  constexpr Manifold(const std::enable_if_t<not F, Representation> &_in)
      : base_t(), representation_(new Representation(_in)),
        const_representation_(representation_), onwing_(true) {}

  template <bool F = Faithfull>
  constexpr Manifold(std::enable_if_t<not F, Representation> &&_in)
      : base_t(), representation_(new Representation(std::move(_in))),
        const_representation_(representation_), onwing_(true) {}

  /// Reference constructor
  constexpr Manifold(Representation *_in)
      : base_t(), representation_(_in), const_representation_(_in),
        onwing_(false) {}
  /// Reference const constructor
  constexpr Manifold(const Representation *_in)
      : base_t(), representation_(nullptr), const_representation_(_in),
        onwing_(false) {}

  /// override assign but private
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ =
        static_cast<Manifold<Atlas, Faithfull> *>(_other.get())->crepr();
  }
  Representation &repr() {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    return *representation_;
  };
};

template <typename M> static constexpr bool manifold_sanity_check() {
  return std::is_base_of_v<ManifoldBase, M>;
  return true;
}
} // namespace manifolds
