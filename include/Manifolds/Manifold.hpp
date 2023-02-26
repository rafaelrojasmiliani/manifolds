
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

public:
  using Representation = std::decay_t<typename Atlas::Representation>;

protected:
  Representation representation_;

public:
  // Generic constructor
  Manifold() : representation_() {}
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

  /*
   // Signment operator
   const Manifold &operator=(const Manifold<Representation, Dim, TDim,
   Faithfull> &_other) { representation_ = _other.crepr();
   }
 */
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
    return Atlas::tanget_repr_dimension;
  }

private:
  Manifold(const Representation &_in) : representation_(_in) {}
  Manifold(Representation &&_in) : representation_(std::move(_in)) {}
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    representation_ =
        static_cast<Manifold<Atlas, Faithfull> *>(_other.get())->crepr();
  }
  Representation &repr() { return representation_; };
};

} // namespace manifolds
