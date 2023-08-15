#pragma once

#include <Manifolds/Detail.hpp>
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/MapBase.hpp>
#include <Manifolds/Maps/MapComposition.hpp>

#include <Eigen/Core>
#include <algorithm>
#include <list>
#include <memory>

namespace manifolds {
//
//
//

/// Map typed class.
template <typename DomainType, typename CoDomainType, detail::MatrixTypeId DT>
class Map : virtual public MapBase {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>,
                "The codomain_t must inherit from ManifoldBase");
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>,
                "The domain_t must intherit from ManifoldBase");

public:
  using domain_t = DomainType;
  using codomain_t = CoDomainType;

  using domain_facade_t =
      std::conditional_t<domain_t::is_faithful,
                         typename domain_t::Representation, domain_t>;
  using codomain_facade_t =
      std::conditional_t<codomain_t::is_faithful,
                         typename codomain_t::Representation, codomain_t>;

  using differential_ref_t =
      std::conditional_t<DT == detail::MatrixTypeId::Dense,
                         detail::dense_matrix_ref_t,
                         std::conditional_t<DT == detail::MatrixTypeId::Sparse,
                                            detail::sparse_matrix_ref_t,
                                            detail::mixed_matrix_ref_t>>;
  // FIXME adapt to dynamic type
  using differential_t = std::conditional_t<
      DT == detail::MatrixTypeId::Dense,
      detail::dense_matrix_t<codomain_t::tangent_repr_dimension,
                             domain_t::tangent_repr_dimension>,
      detail::sparse_matrix_t>;

  static constexpr detail::MatrixTypeId differential_type_id = DT;

  // FIXME adapt to dynamic type
  static constexpr std::size_t domain_dimension = DomainType::dimension;
  // FIXME adapt to dynamic type
  static constexpr std::size_t codomain_dimension = CoDomainType::dimension;

  // -------------------------------------
  // Default lifecycle
  // -------------------------------------
  virtual ~Map() = default;

  // Clone
  std::unique_ptr<Map> clone() const {
    return std::unique_ptr<Map>(static_cast<Map *>(this->clone_impl()));
  }

  std::unique_ptr<Map> move_clone() {
    return std::unique_ptr<Map>(static_cast<Map *>(this->move_clone_impl()));
  }

  // ---------------------------------
  // Definition of bool value(in, out)
  // ---------------------------------
  bool value(const domain_facade_t &_in, codomain_facade_t &_out) const {
    return value_on_repr(_in, _out);
  }

  // ---------------------------------
  // Definition of bool operator(in, out)
  // ---------------------------------
  bool operator()(const domain_facade_t &_in, codomain_facade_t &_out) const {
    return value_on_repr(_in, _out);
  }

  codomain_t operator()(const domain_facade_t &_in) const {
    static auto out = codomain_t();
    value_on_repr(_in, out);
    return out;
  }

  // ---------------------------------
  // Definition of bool diff(in, out)
  // ---------------------------------
  bool diff(const domain_facade_t &_in, differential_ref_t _out) const {
    return diff_from_repr(_in, _out);
  }

  differential_t diff(const domain_facade_t &_in) const {
    differential_t out;
    diff_from_repr(_in, out);
    return out;
  }

  // ---------------------------------
  // Definition of Composition
  // ---------------------------------
  template <typename OtherCodomain, detail::MatrixTypeId OtherDT>
  auto operator|(const Map<codomain_t, OtherCodomain, OtherDT> &_in) const & {

    constexpr detail::MatrixTypeId mt =
        (OtherDT == DT and DT == detail::MatrixTypeId::Sparse)
            ? detail::MatrixTypeId::Sparse
            : detail::MatrixTypeId::Dense;

    MapBaseComposition result({*this, _in});

    return MapComposition<domain_t, OtherCodomain, mt>(result);
  }

  // ---------------------------------
  // Getters
  // ---------------------------------
  virtual std::size_t get_dom_dim() const override {
    // FIXME adapt to dynamic type
    return DomainType::dimension;
  }
  virtual std::size_t get_codom_dim() const override {
    std::cout << "\n----  here -\n\n;";
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    // FIXME adapt to dynamic type
    return CoDomainType::dimension;
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    // FIXME adapt to dynamic type
    return DomainType::tangent_repr_dimension;
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    // FIXME adapt to dynamic type
    return CoDomainType::tangent_repr_dimension;
  }

  detail::mixed_matrix_t linearization_buffer() const override {
    // FIXME adapt to dynamic type
    if constexpr (DT == detail::MatrixTypeId::Dense)
      return Eigen::MatrixXd(codomain_t::tangent_repr_dimension,
                             domain_t::tangent_repr_dimension);
    return detail::sparse_matrix_t(codomain_t::tangent_repr_dimension,
                                   domain_t::tangent_repr_dimension);
  }
  virtual detail::MatrixTypeId differential_type() const override { return DT; }

protected:
  /// Implementation of differentiation of for MapBase. This is a
  /// representation-agnostic evaluation.
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {

    const domain_facade_t *input_ptr = nullptr;
    codomain_facade_t *output_ptr = nullptr;

    if constexpr (DomainType::is_faithful)
      input_ptr = &static_cast<const DomainType *>(_in)->crepr();
    else
      input_ptr = static_cast<const DomainType *>(_in);

    if constexpr (CoDomainType::is_faithful)
      output_ptr = &static_cast<CoDomainType *>(_other)->repr();
    else
      output_ptr = static_cast<CoDomainType *>(_other);

    return value_on_repr(*input_ptr, *output_ptr);
  }

  /// Implementation of differentiation of for MapBase. This is a
  /// representaton-agnostic evaluation.
  bool diff_impl(const ManifoldBase *_in,
                 detail::mixed_matrix_ref_t _mat) const override {

    const domain_facade_t *input_ptr = nullptr;

    if constexpr (DomainType::is_faithful)
      input_ptr = &static_cast<const DomainType *>(_in)->crepr();
    else
      input_ptr = static_cast<const DomainType *>(_in);

    if constexpr (DT == detail::MatrixTypeId::Dense)
      return diff_from_repr(*input_ptr,
                            std::get<detail::dense_matrix_ref_t>(_mat));
    if constexpr (DT == detail::MatrixTypeId::Sparse)
      return diff_from_repr(*input_ptr,
                            std::get<detail::sparse_matrix_ref_t>(_mat).get());
  }

  /// Get a pointer with an instance of the domain_t. Override if you have a
  /// custom domain_t
  virtual ManifoldBase *domain_buffer_impl() const override {
    return new domain_t();
  }

  /// Get a pointer with an instance of the codomain_t. Override if you have
  /// custom codomain_t
  virtual ManifoldBase *codomain_buffer_impl() const override {
    return new codomain_t();
  }

protected:
  /// Function to copmute the result of the map using just the representation
  /// types
  virtual bool value_on_repr(const domain_facade_t &_in,
                             codomain_facade_t &_result) const = 0;

  /// Function to copmute the result of the map differential using just the
  /// representation types
  virtual bool diff_from_repr(const domain_facade_t &_in,
                              differential_ref_t _mat) const = 0;

  virtual Map *clone_impl() const override = 0;
  virtual Map *move_clone_impl() override = 0;

  template <typename Atlas, bool F>
  static auto &get_repr(Manifold<Atlas, F> &man) {
    return man.repr();
  }

private:
  using MapBase::operator|;

  virtual MapBaseComposition *pipe_impl(const MapBase &_that) const override {
    return new MapBaseComposition({*this, _that});
  }
  virtual MapBaseComposition *pipe_move_impl(MapBase &&_that) const override {

    return new MapBaseComposition({*this, _that});
  }
};

// ------------------------
// ---- Identity ---------
template <typename Set>
class Identity : public detail::Clonable<Identity<Set>, Map<Set, Set>> {
public:
  static_assert(std::is_base_of_v<ManifoldBase, Set>);
  Identity() = default;
  Identity(const Identity &_that) = default;
  Identity(Identity &&_that) = default;

protected:
  bool value_on_repr(const Set &_in, Set &_result) const override {

    _result = _in;
    return true;
  }
  bool diff_from_repr(const typename Set::Representation &,
                      Eigen::Ref<Eigen::MatrixXd>) const override {
    return true;
  }
};

template <typename Domain, typename Codomain, detail::MatrixTypeId DT>

class MapLifting : public detail::Clonable<MapLifting<Domain, Codomain, DT>,
                                           Map<Domain, Codomain, DT>> {

public:
  using base_t = detail::Clonable<MapLifting<Domain, Codomain, DT>,
                                  Map<Domain, Codomain, DT>>;

  using value_fun_t =
      std::function<bool(const typename Domain::Representation &,
                         typename Codomain::Representation &)>;
  using diff_fun_t = std::function<bool(const typename Domain::Representation &,
                                        typename base_t::differential_ref_t)>;
  MapLifting(const value_fun_t &_value_fun, const diff_fun_t &_diff_fun)
      : base_t(), value_fun_(_value_fun), diff_fun_(_diff_fun) {}

  bool
  value_on_repr(const typename base_t::domain_facade_t &_in,
                typename base_t::codomain_facade_t &_result) const override {

    return value_fun_(_in, _result);
  }

  bool diff_from_repr(const typename base_t::domain_facade_t &_in,
                      typename base_t::differential_ref_t _mat) const override {
    return diff_fun_(_in, _mat);
  }

private:
  value_fun_t value_fun_;
  diff_fun_t diff_fun_;
};

} // namespace manifolds
