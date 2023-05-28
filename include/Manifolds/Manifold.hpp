#pragma once
#include <Eigen/Core>
#include <Manifolds/ManifoldBase.hpp>
#include <exception>
#include <iostream>
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
  using RepresentationRef = typename Atlas::RepresentationRef;
  using RepresentationConstRef = typename Atlas::RepresentationConstRef;
  using Atlas::cref_to_type;
  using Atlas::ctype_to_ref;
  using Atlas::ref_to_type;
  using Atlas::type_to_ref;

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
  /// Default constructor. calls default constructor of its representation
  Manifold()
      : base_t(), representation_(new Representation()),
        const_representation_(representation_), onwing_(true) {}
  /// Copy constructor. Copy construct the representation into the holding
  /// pointer
  Manifold(const Manifold &that)
      : base_t(that),
        representation_(new Representation(*that.representation_)),
        const_representation_(representation_), onwing_(true) {}
  /// Move constructor. Move construct the representation into the holding
  /// pointer
  Manifold(Manifold &&that)
      : base_t(std::move(that)),
        representation_(new Representation(std::move(*that.representation_))),
        const_representation_(representation_), onwing_(true) {
    that.representation_ = nullptr;
    that.const_representation_ = nullptr;
    that.onwing_ = false;
  }

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
  // Move assignment with manifold operator
  const Manifold &operator=(Manifold &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = std::move(*_other.representation_);
    delete _other.representation_;
    _other.representation_ = nullptr;
    _other.const_representation_ = nullptr;
    _other.onwing_ = false;
    return *this;
  }

  // **REMARK**
  // Assignemtns from representation are not necessray, because we can construct
  // a temporal Manifold from Representation and use the move assign from a
  // Manifold Assignment with representation for faithfull manifolds
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(const Representation &_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    *representation_ = _other;
    return *this;
  }

  // Move assignment with representation for faithfull manifolds
  template <bool F = Faithfull>
  std::enable_if_t<F, Manifold<Atlas, Faithfull>> &
  operator=(Representation &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifol element");
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
  static constexpr Manifold Ref(const Representation *in) {
    return Manifold(in);
  }
  // Get a manifold representation of a piece of data
  static constexpr Manifold Ref(Representation &in) { return Manifold(&in); }

  static constexpr Manifold Ref(const Representation &in) {
    return Manifold(&in);
  }

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++ Casting  and implicid Cast consturcotrs  +++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  constexpr operator const Representation &() const & {
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
  constexpr const Representation &crepr() const {
    return *const_representation_;
  };

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++ Charts, Params and change of coordinates ++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  class Chart;
  class Parametrization;
  class ChangeOfCoordinates;

private:
  /// SNIFAE of constructors: Make non faithfull private. If the manifold is
  /// not faithfull, the following constructor we be private, This is to avoid
  /// constructing objects from its representation

  // Cast constructor from representation
  template <bool F = Faithfull>
  constexpr Manifold(const std::enable_if_t<not F, Representation> &_in)
      : base_t(), representation_(new Representation(_in)),
        const_representation_(representation_), onwing_(true) {}

  // Cast-move constructor from representation
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

protected:
  constexpr Representation &repr() {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    return *representation_;
  };
  constexpr Representation &&mrepr() {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constnat manifold element");
    return std::move(*representation_);
  };
};

template <typename M> static constexpr bool manifold_sanity_check() {
  return std::is_base_of_v<ManifoldBase, M>;
  return true;
}
} // namespace manifolds
