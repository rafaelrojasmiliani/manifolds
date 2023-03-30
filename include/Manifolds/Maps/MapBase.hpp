#pragma once
#include <Eigen/Sparse>
#include <Manifolds/ManifoldBase.hpp>
#include <functional>
#include <memory>
#include <variant>

namespace manifolds {

using DifferentialReprRefType =
    std::variant<Eigen::Ref<Eigen::MatrixXd>,
                 std::reference_wrapper<Eigen::SparseMatrix<double>>>;

using DifferentialReprType =
    std::variant<Eigen::MatrixXd, Eigen::SparseMatrix<double>>;

class MapBaseComposition;
/** Dynamic and type-agnostic function representation
 * **/
class MapBase {
  friend class MapBaseComposition;

private:
  virtual bool value_impl(const ManifoldBase *_in,
                          ManifoldBase *_other) const = 0;
  // Here, change to Variant of dense and sparse matrix
  virtual bool diff_impl(const ManifoldBase *_in,
                         DifferentialReprRefType _mat) const = 0;

public:
  // Default lifecycle
  MapBase(const MapBase &) = default;
  MapBase(MapBase &&) = default;
  MapBase() = default;
  virtual ~MapBase() = default;
  // value returns
  std::unique_ptr<ManifoldBase>
  value(const std::unique_ptr<ManifoldBase> &_in) const;

  // value gets a buffer
  bool value(const std::unique_ptr<ManifoldBase> &_in,
             std::unique_ptr<ManifoldBase> &_other) const;

  // other stuff
  std::unique_ptr<MapBase> clone() const;

  std::unique_ptr<MapBase> move_clone();

  // Here, change to Variant of dense and sparse matrix
  bool diff(const std::unique_ptr<ManifoldBase> &_in,
            DifferentialReprRefType _mat) const;

  // return manifold buffers
  virtual std::unique_ptr<ManifoldBase> codomain_buffer() const;
  std::unique_ptr<ManifoldBase> domain_buffer() const;

  virtual std::size_t get_dom_dim() const = 0;
  virtual std::size_t get_dom_tangent_repr_dim() const = 0;
  virtual std::size_t get_codom_dim() const = 0;
  virtual std::size_t get_codom_tangent_repr_dim() const = 0;

  // Here, change to Variant of dense and sparse matrix
  virtual DifferentialReprType linearization_buffer() const = 0;
  virtual bool is_differential_sparse() const = 0;

protected:
  virtual MapBase *clone_impl() const = 0;
  virtual MapBase *move_clone_impl() = 0;
  virtual ManifoldBase *domain_buffer_impl() const = 0;
  virtual ManifoldBase *codomain_buffer_impl() const = 0;
};

template <typename Current, typename Base>
class MapInheritanceHelper : public Base {
public:
  using Base::Base;
  virtual ~MapInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

protected:
  virtual Base *clone_impl() const override {

    return new Current(*static_cast<const Current *>(this));
  }

  virtual Base *move_clone_impl() override {
    return new Current(std::move(*static_cast<Current *>(this)));
  }
};

template <typename Current, typename Base>
class AbstractMapInheritanceHelper : public Base {
public:
  using Base::Base;
  AbstractMapInheritanceHelper(const AbstractMapInheritanceHelper &) = default;
  AbstractMapInheritanceHelper(AbstractMapInheritanceHelper &&) = default;
  virtual ~AbstractMapInheritanceHelper() = default;

  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(clone_impl());
  }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(move_clone_impl());
  }

protected:
  virtual Base *clone_impl() const = 0;
  virtual Base *move_clone_impl() = 0;
};

} // namespace manifolds
