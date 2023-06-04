#pragma once
#include <Eigen/Core>
#include <Manifolds/Detail.hpp>
#include <Manifolds/ManifoldBase.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace manifolds {

/// Inheritance helper
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
  __DEFAULT_REF(Current, Base)
};

template <typename Atlas, bool Faithfull = false>
class Manifold : public ManifoldBase, public Atlas {
  template <typename T, typename U> friend class Map;
  template <typename T, typename U> friend class MapComposition;

  using base_t = ManifoldBase;

public:
  using Representation = std::decay_t<typename Atlas::Representation>;
  using RepresentationRef = typename Atlas::RepresentationRef;

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
    if (that.onwing_)
      delete that.representation_;
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
  // static constexpr Manifold Ref(Representation *in) { return Manifold(in); }
  // Get a manifold representation of a piece of data
  // static constexpr Manifold Ref(const Representation *in) {
  // return Manifold(in);
  //}
  // Get a manifold representation of a piece of data
  // static constexpr Manifold Ref(Representation &in) { return Manifold(&in); }

  // static constexpr Manifold Ref(const Representation &in) {
  //   return Manifold(&in);
  // }

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
    auto &&temp = std::move(*representation_);
    delete representation_;
    representation_ = nullptr;
    const_representation_ = nullptr;
    return std::move(temp);
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

protected:
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

public:
  std::unique_ptr<Manifold> clone() const {
    return std::unique_ptr<Manifold>(static_cast<Manifold *>(clone_impl()));
  }

  std::unique_ptr<Manifold> move_clone() {
    return std::unique_ptr<Manifold>(
        static_cast<Manifold *>(move_clone_impl()));
  }

protected:
  virtual Manifold *clone_impl() const override {

    return new Manifold(*static_cast<const Manifold *>(this));
  }

  virtual Manifold *move_clone_impl() override {
    return new Manifold(std::move(*static_cast<Manifold *>(this)));
  }
};

template <typename M> static constexpr bool manifold_sanity_check() {
  return std::is_base_of_v<ManifoldBase, M>;
  return true;
}
} // namespace manifolds
