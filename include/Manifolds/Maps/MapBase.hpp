#pragma once
#include <Eigen/Sparse>
#include <Manifolds/Detail.hpp>
#include <Manifolds/ManifoldBase.hpp>
#include <functional>
#include <memory>
#include <variant>

namespace manifolds {

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

  bool diff(const std::unique_ptr<ManifoldBase> &_in,
            DifferentialReprRefType _mat) const;

  // return manifold buffers
  virtual std::unique_ptr<ManifoldBase> codomain_buffer() const;
  std::unique_ptr<ManifoldBase> domain_buffer() const;

  virtual std::size_t get_dom_dim() const = 0;
  virtual std::size_t get_dom_tangent_repr_dim() const = 0;
  virtual std::size_t get_codom_dim() const = 0;
  virtual std::size_t get_codom_tangent_repr_dim() const = 0;

  virtual DifferentialReprType linearization_buffer() const = 0;
  virtual MatrixTypeId differential_type() const = 0;

protected:
  virtual MapBase *clone_impl() const = 0;
  virtual MapBase *move_clone_impl() = 0;
  virtual ManifoldBase *domain_buffer_impl() const = 0;
  virtual ManifoldBase *codomain_buffer_impl() const = 0;
};

} // namespace manifolds
