
#pragma once
#include <Manifolds/Topology/Open.hpp>
#include <memory>
namespace manifolds {

class MapBase;
class MapBaseComposition;
template <typename T, typename U> class Map;
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
public:
  using Base::Base;
  virtual ~ManifoldInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

protected:
  virtual ManifoldBase *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual ManifoldBase *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
};

template <typename R, long Dim, long TDim>
class Manifold
    : public ManifoldInheritanceHelper<Manifold<R, Dim, TDim>, ManifoldBase> {
  template <typename T, typename U> friend class Map;

protected:
  R representation_;

public:
  template <typename... Ts>
  Manifold(Ts &&... args) : representation_(std::forward<Ts>(args)...) {}
  static const long dim = Dim;
  static const long tangent_repr_dim = TDim;
  typedef R Representation;
  const std::decay_t<R> &repr() const { return representation_; };
  virtual std::size_t get_dim() const override { return dim; }

private:
  void assign(const std::unique_ptr<ManifoldBase> &_other) override {
    representation_ =
        static_cast<Manifold<R, Dim, TDim> *>(_other.get())->repr();
  }
  std::decay_t<R> &repr() { return representation_; };
};

template <long Rows, long Cols>
class MatrixManifold : public ManifoldInheritanceHelper<
                           MatrixManifold<Rows, Cols>,
                           Manifold<Eigen::Matrix<double, Rows, Cols>,
                                    Rows * Cols, Rows * Cols>> {

public:
  using ManifoldInheritanceHelper<
      MatrixManifold<Rows, Cols>,
      Manifold<Eigen::Matrix<double, Rows, Cols>, Rows * Cols,
               Rows * Cols>>::ManifoldInheritanceHelper;

  operator const Eigen::Matrix<double, Rows, Cols> &() const & {
    return this->representation_;
  }
  operator Eigen::Matrix<double, Rows, Cols> &() & {
    return this->representation_;
  }
  operator Eigen::Matrix<double, Rows, Cols> &&() && {
    return std::move(this->representation_);
  }
};

template <long Rows, long Cols>
class MatrixManifoldConstRef
    : public ManifoldInheritanceHelper<
          MatrixManifoldConstRef<Rows, Cols>,
          const Manifold<Eigen::Matrix<double, Rows, Cols> &, Rows * Cols,
                         Rows * Cols>> {

public:
  using ManifoldInheritanceHelper<
      MatrixManifoldConstRef<Rows, Cols>,
      const Manifold<Eigen::Matrix<double, Rows, Cols> &, Rows * Cols,
                     Rows * Cols>>::ManifoldInheritanceHelper;

  operator const Eigen::Matrix<double, Rows, Cols> &() const & {
    return this->representation_;
  }
  operator Eigen::Matrix<double, Rows, Cols> &() & {
    return this->representation_;
  }
  operator Eigen::Matrix<double, Rows, Cols> &&() && {
    return std::move(this->representation_);
  }
};

using Rn = MatrixManifold<Eigen::Dynamic, 1>;
using R = MatrixManifold<1, 1>;
using R2 = MatrixManifold<2, 1>;
using R3 = MatrixManifold<3, 1>;
using RnConstRef = MatrixManifoldConstRef<Eigen::Dynamic, 1>;
using RConstRef = MatrixManifoldConstRef<1, 1>;
using R2ConstRef = MatrixManifoldConstRef<2, 1>;
using R3ConstRef = MatrixManifoldConstRef<3, 1>;

} // namespace manifolds
