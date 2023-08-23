#pragma once
#include <Eigen/Core>
#include <Manifolds/Detail.hpp>
#include <Manifolds/ManifoldBase.hpp>
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
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(ManifoldInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(ManifoldInheritanceHelper, Current, Base)
  __DEFAULT_REF(Current, Base)
};

template <typename Atlas, bool Faithful = false>
class Manifold : public ManifoldBase {
  template <typename T, typename U, detail::MatrixTypeId DT> friend class Map;
  template <typename T, typename U, detail::MatrixTypeId DT>
  friend class MapComposition;

  using base_t = ManifoldBase;

public:
  using Representation = std::decay_t<typename Atlas::Representation>;
  using RepresentationRef = typename Atlas::RepresentationRef;
  using facade_t =
      std::conditional_t<Faithful, Representation, Manifold<Atlas, Faithful>>;
  static constexpr std::size_t dimension = Atlas::dimension;
  static constexpr std::size_t tangent_repr_dimension =
      Atlas::tangent_repr_dimension;

  using atlas = Atlas;

protected:
  Representation *representation_;
  const Representation *const_representation_;

private:
  bool owning_;

public:
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++   Live cycle    +++++++++++++++++++++++++++++++
  // Livecycle Constructor copy, move constructor, assigment and destructor
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  /// Default constructor. calls default constructor of its representation
  Manifold()
      : base_t(),
        representation_(new Representation(Atlas::random_projection())),
        const_representation_(representation_), owning_(true) {}
  /// Copy constructor. Copy construct the representation into the holding
  /// pointer
  Manifold(const Manifold &that)
      : base_t(that),
        representation_(new Representation(*that.representation_)),
        const_representation_(representation_), owning_(true) {}
  /// Move constructor. Move construct the representation into the holding
  /// pointer
  Manifold(Manifold &&that)
      : base_t(std::move(that)),
        representation_(new Representation(std::move(*that.representation_))),
        const_representation_(representation_), owning_(true) {
    if (that.owning_)
      delete that.representation_;
    that.representation_ = nullptr;
    that.const_representation_ = nullptr;
    that.owning_ = false;
  }

  virtual ~Manifold() override {
    if (owning_)
      delete representation_;
    representation_ = nullptr;
    const_representation_ = nullptr;
  }

  // Assignment with manifold operator
  const Manifold &operator=(const Manifold &_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constant manifold element");
    *representation_ = _other.crepr();
    return *this;
  }
  // Move assignment with manifold operator
  const Manifold &operator=(Manifold &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constant manifold element");
    *representation_ = std::move(*_other.representation_);
    delete _other.representation_;
    _other.representation_ = nullptr;
    _other.const_representation_ = nullptr;
    _other.owning_ = false;
    return *this;
  }

  static constexpr Manifold Ref(typename Manifold::Representation *in) {
    return Manifold(in);
  }
  static constexpr Manifold CRef(const typename Manifold::Representation *in) {
    return Manifold(in);
  }
  static constexpr Manifold Ref(typename Manifold::Representation &in) {
    return Manifold(&in);
  }

  static constexpr Manifold CRef(const typename Manifold::Representation &in) {
    return Manifold(&in);
  }

  template <typename T>
  static constexpr T from_repr(const typename Manifold::Representation &in) {
    return T(in);
  }

  // **REMARK**
  // Assignments from representation are not necessary, because we can construct
  // a temporal Manifold from Representation and use the move assign from a
  // Manifold Assignment with representation for faithful manifolds
  template <bool F = Faithful>
  std::enable_if_t<F, Manifold<Atlas, Faithful>> &
  operator=(const Representation &_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constant manifold element");
    *representation_ = _other;
    return *this;
  }

  // Move assignment with representation for faithful manifolds
  template <bool F = Faithful>
  std::enable_if_t<F, Manifold<Atlas, Faithful>> &
  operator=(Representation &&_other) {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constant manifold element");
    *representation_ = std::move(_other);
    return *this;
  }

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++   Static const +++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  static constexpr bool is_faithful = Faithful;

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

  template <bool F = Faithful>
  constexpr operator std::enable_if_t<F, Representation> &() & {
    if (not representation_)
      throw std::logic_error("Trying to assign to a constant manifold element");
    return *representation_;
  }

  template <bool F = Faithful>
  constexpr operator std::enable_if_t<F, Representation>() && {
    if (not representation_)
      throw std::logic_error("Trying to move a constant manifold element");
    Representation temp = std::move(*representation_);
    delete representation_;
    representation_ = nullptr;
    const_representation_ = nullptr;
    return temp;
  }

  template <bool F = Faithful>
  explicit constexpr Manifold(const std::enable_if_t<F, Representation> &_in)
      : base_t(), representation_(new Representation(_in)),
        const_representation_(representation_), owning_(true) {}

  template <bool F = Faithful>
  constexpr Manifold(const std::enable_if_t<F, Representation> &&_in)
      : base_t(), representation_(new Representation(std::move(_in))),
        const_representation_(representation_), owning_(true) {}

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++ Getterns ++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  /// Get the dimension of the manifold
  std::size_t get_dim() const override { return Atlas::dimension; }
  /// Get the dimension of the representatino of the tangent space
  std::size_t get_tanget_repr_dim() const override {
    return Atlas::tangent_repr_dimension;
  }

  static Representation random_projection() {
    return Atlas::random_projection();
  }
  static Manifold random() { return Manifold(Atlas::random_projection()); }

  bool has_value() const override {
    return const_representation_ != nullptr or representation_ != nullptr;
  }

  /// Get const referece to the representtion
  constexpr const Representation &crepr() const {
    return *const_representation_;
  };

  template <bool F = Faithful>
  constexpr std::enable_if_t<F, Representation> &repr() {
    if (not representation_)
      throw std::logic_error(
          "Trying to get a non constant reference to a constant manifold");
    return *representation_;
  };

  template <bool F = Faithful>
  constexpr std::enable_if_t<F, Representation> &&mrepr() {
    if (not representation_)
      throw std::logic_error(
          "Trying to get a non constant reference to a constant manifold");
    return std::move(*representation_);
  };

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++ Comparison  +++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  virtual bool
  is_equal(const std::unique_ptr<ManifoldBase> &_other) const override {
    if (this == _other.get())
      return true;
    if (not _other->is_same<Manifold<Atlas, Faithful>>())
      throw std::logic_error(
          "You are trying to compare elements from different manifolds");
    return *this == *dynamic_cast<Manifold<Atlas, Faithful> *>(_other.get());
  }

  bool operator==(const Manifold &_that) const {
    return Atlas::comparison(this->crepr(), _that.crepr());
  }

  bool operator!=(const Manifold &_that) const { return not operator==(_that); }

  /*
  template <bool F = Faithful>
  std::enable_if_t<F, bool> operator==(const Representation &_that) {
    return Atlas::comparison(this->crepr(), _that);
  }

  template <bool F = Faithful>
  std::enable_if_t<F, bool> operator!=(const Representation &_that) {
    return not Atlas::comparison(this->crepr(), _that);
  }
  */
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // ++++++++++++++++++ Charts, Params and change of coordinates ++++++++++++++
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  class Chart;
  class Parametrization;
  class ChangeOfCoordinates;

protected:
  /// SNIFAE of constructors: Make non faithful private. If the manifold is
  /// not faithful, the following constructor we be private, This is to avoid
  /// constructing objects from its representation

  // Cast constructor from representation
  template <bool F = Faithful>
  explicit constexpr Manifold(
      const std::enable_if_t<not F, Representation> &_in)
      : base_t(), representation_(new Representation(_in)),
        const_representation_(representation_), owning_(true) {}

  // Cast-move constructor from representation
  template <bool F = Faithful>
  constexpr Manifold(std::enable_if_t<not F, Representation> &&_in)
      : base_t(), representation_(new Representation(std::move(_in))),
        const_representation_(representation_), owning_(true) {}

  /// Reference constructor
  constexpr Manifold(Representation *_in)
      : base_t(), representation_(_in), const_representation_(_in),
        owning_(false) {}
  /// Reference const constructor
  constexpr Manifold(const Representation *_in)
      : base_t(), representation_(nullptr), const_representation_(_in),
        owning_(false) {}

  /// override assign but private
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    if (representation_ == nullptr and const_representation_ != nullptr)
      throw std::logic_error("Trying to assign to a constant manifold");

    if (dynamic_cast<Manifold<Atlas, Faithful> *>(_other.get()) == nullptr)
      throw std::logic_error(
          "You are trying to assign differenty types of manifold");

    if (not representation_ and not const_representation_)
      *representation_ = Representation();

    *representation_ =
        static_cast<Manifold<Atlas, Faithful> *>(_other.get())->crepr();
    const_representation_ = representation_;
  }

protected:
  template <bool F = Faithful>
  constexpr std::enable_if_t<not F, Representation> &repr() {
    if (not representation_)
      throw std::logic_error(
          "Trying to get a non constant reference to a constant manifold");
    return *representation_;
  };
  template <bool F = Faithful>
  constexpr std::enable_if_t<not F, Representation> &&mrepr() {
    if (not representation_)
      throw std::logic_error(
          "Trying to get a non constant reference to a constant manifold");
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
  virtual Manifold *clone_impl() const override { return new Manifold(*this); }

  virtual Manifold *move_clone_impl() override {
    return new Manifold(std::move(*this));
  }
};

template <typename M> static constexpr bool manifold_sanity_check() {
  return std::is_base_of_v<ManifoldBase, M>;
  return true;
}
} // namespace manifolds
