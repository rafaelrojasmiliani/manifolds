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

protected:
  virtual bool value_impl(const ManifoldBase *_in,
                          ManifoldBase *_other) const = 0;
  // Here, change to Variant of dense and sparse matrix
  virtual bool diff_impl(const ManifoldBase *_in,
                         detail::mixed_matrix_ref_t _mat) const = 0;

public:
  // Default lifecycle
  virtual ~MapBase();

  std::unique_ptr<MapBase> clone() const;

  std::unique_ptr<MapBase> move_clone();

  // --------------------
  // Evaluation
  // --------------------

  // Evaluation
  std::unique_ptr<ManifoldBase> value(const ManifoldBase &_in) const;

  // Evaluation in pre-allocated buffer
  bool value(const std::unique_ptr<ManifoldBase> &_in,
             std::unique_ptr<ManifoldBase> &_other) const;

  // differentiation in pre-allocated buffer
  bool diff(const ManifoldBase &_in, detail::mixed_matrix_ref_t _mat) const;

  // differentiation
  detail::mixed_matrix_t diff(const ManifoldBase &_in) const;

  virtual std::unique_ptr<MapBaseComposition>
  pre_compose_ptr(const std::unique_ptr<MapBase> &) = 0;
  // return manifold buffers
  virtual std::unique_ptr<ManifoldBase> codomain_buffer() const;
  std::unique_ptr<ManifoldBase> domain_buffer() const;

  virtual std::size_t get_dom_dim() const = 0;
  virtual std::size_t get_dom_tangent_repr_dim() const = 0;
  virtual std::size_t get_codom_dim() const = 0;
  virtual std::size_t get_codom_tangent_repr_dim() const = 0;

  virtual detail::mixed_matrix_t linearization_buffer() const = 0;
  virtual detail::MatrixTypeId differential_type() const = 0;

protected:
  virtual MapBase *clone_impl() const = 0;
  virtual MapBase *move_clone_impl() = 0;
  virtual ManifoldBase *domain_buffer_impl() const = 0;
  virtual ManifoldBase *codomain_buffer_impl() const = 0;
};

} // namespace manifolds
