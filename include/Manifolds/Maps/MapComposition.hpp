#pragma once
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/MapBaseComposition.hpp>

namespace manifolds {

template <typename DomainType, typename CoDomainType,
          detail::MatrixTypeId DT = detail::MatrixTypeId::Dense>
class Map;

/// Typed composition of maps.
template <typename DomainType, typename CoDomainType, detail::MatrixTypeId DT>
class MapComposition : public Map<DomainType, CoDomainType, DT>,
                       public MapBaseComposition {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

protected:
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

  using map_t = Map<DomainType, CoDomainType, DT>;

  template <typename A, typename B, detail::MatrixTypeId C> friend class Map;
  template <typename A, typename B, detail::MatrixTypeId C>
  friend class MapComposition;

  std::size_t get_dom_dim() const override {
    return MapBaseComposition::get_dom_dim();
  }
  std::size_t get_codom_dim() const override {
    return MapBaseComposition::get_codom_dim();
  }

  std::size_t get_dom_tangent_repr_dim() const override {
    return MapBaseComposition::get_dom_tangent_repr_dim();
  }
  std::size_t get_codom_tangent_repr_dim() const override {
    return MapBaseComposition::get_codom_tangent_repr_dim();
  }
  virtual detail::MatrixTypeId differential_type() const override { return DT; }

  ManifoldBase *domain_buffer_impl() const override {
    return map_t::domain_buffer_impl();
  }
  ManifoldBase *codomain_buffer_impl() const override {
    return map_t::codomain_buffer_impl();
  }

  detail::mixed_matrix_t linearization_buffer() const override {
    // FIXME adapt to dynamic type
    if constexpr (DT == detail::MatrixTypeId::Dense)
      return Eigen::MatrixXd(codomain_t::tangent_repr_dimension,
                             domain_t::tangent_repr_dimension);
    return detail::sparse_matrix_t(codomain_t::tangent_repr_dimension,
                                   domain_t::tangent_repr_dimension);
  }
  // -------------------------------------------
  // -------- Live cycle -----------------------
  // -------------------------------------------

  /// Constructor from a typed map
  // template <typename M> MapComposition(const M &m) : MapBaseComposition(m) {}
  template <typename M> MapComposition(const M &m) : MapBaseComposition(m) {

    static_assert(std::is_base_of_v<MapBase, std::decay_t<M>>);
    static_assert(std::is_same_v<typename M::codomain_t, codomain_t>);
    static_assert(std::is_same_v<typename M::domain_t, domain_t>);
  }
  /// Move-Constructor from a typed map
  // template <typename M>
  // MapComposition(M &&m) : MapBaseComposition(std::move(m)) {
  //   static_assert(std::is_base_of_v<MapBase, std::decay_t<M>>);
  //   static_assert(std::is_same_v<typename std::decay_t<M>::codomain_t,
  //   codomain_t>); static_assert(std::is_same_v<typename
  //   std::decay_t<M>::domain_t, domain_t>);
  // }

  /// Copy constructor from a typed map
  MapComposition(const MapComposition &m) : MapBaseComposition(m) {}

  /// Move constructor from a typed map
  MapComposition(MapComposition &&m) : MapBaseComposition(std::move(m)) {}

  /// Default destructor.
  virtual ~MapComposition() = default;

  MapComposition operator=(const MapComposition &that) {
    MapBaseComposition::operator=(that);
    return *this;
  }

  MapComposition operator=(MapComposition &&that) {
    MapBaseComposition::operator=(std::move(that));
    return *this;
  }
  // -------------------------------------------
  // -------- Compose ------------------------
  // -------------------------------------------

  template <typename OtherCodomain, detail::MatrixTypeId OtherDT>
  auto operator|(const Map<codomain_t, OtherCodomain, OtherDT> &_in) const & {

    constexpr detail::MatrixTypeId mt =
        (OtherDT == DT and DT == detail::MatrixTypeId::Sparse)
            ? detail::MatrixTypeId::Sparse
            : detail::MatrixTypeId::Dense;

    MapBaseComposition result(*this);
    result.append(_in);

    return MapComposition<domain_t, OtherCodomain, mt>(result);
  }

  template <typename OtherCodomain, detail::MatrixTypeId OtherDT>
  auto operator|(const Map<codomain_t, OtherCodomain, OtherDT> &_in) && {

    constexpr detail::MatrixTypeId mt =
        (OtherDT == DT and DT == detail::MatrixTypeId::Sparse)
            ? detail::MatrixTypeId::Sparse
            : detail::MatrixTypeId::Dense;

    MapBaseComposition result(std::move(*this));
    result.append(_in);

    return MapComposition<domain_t, OtherCodomain, mt>(result);
  }

protected:
  // -------------------------------------------
  // -------- Interface -----------------------
  // -------------------------------------------
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {
    return MapBaseComposition::value_impl(_in, _other);
  }
  bool diff_impl(const ManifoldBase *_in, ManifoldBase *_out,
                 detail::mixed_matrix_ref_t _mat) const override {
    return MapBaseComposition::diff_impl(_in, _out, _mat);
  }
  virtual MapComposition *clone_impl() const override {
    return new MapComposition(*this);
  }

  virtual MapComposition *move_clone_impl() override {
    return new MapComposition(std::move(*(this)));
  }
  virtual bool value_on_repr(const domain_facade_t &_in,
                             codomain_facade_t &_out) const override {
    if constexpr (domain_t::is_faithful and codomain_t::is_faithful) {
      domain_t dom = domain_t::CRef(_in);
      codomain_t codom = codomain_t::Ref(_out);
      return value_impl(&dom, &codom);
    } else if constexpr (domain_t::is_faithful and
                         not codomain_t::is_faithful) {
      domain_t dom = domain_t::CRef(_in);
      return value_impl(&dom, &_out);
    } else if constexpr (not domain_t::is_faithful and
                         codomain_t::is_faithful) {
      codomain_t codom = codomain_t::Ref(_out);
      return value_impl(&_in, &codom);
    } else
      return value_impl(&_in, &_out);
  }

  virtual bool diff_from_repr(const domain_facade_t &_in,
                              codomain_facade_t &_out,
                              differential_ref_t _mat) const override {
    if constexpr (domain_t::is_faithful and codomain_t::is_faithful) {
      domain_t dom = domain_t::CRef(_in);
      codomain_t codom = codomain_t::Ref(_out);
      return diff_impl(&dom, &codom, _mat);
    } else if constexpr (domain_t::is_faithful and
                         not codomain_t::is_faithful) {
      domain_t dom = domain_t::CRef(_in);
      return diff_impl(&dom, &_out, _mat);
    } else if constexpr (not domain_t::is_faithful and
                         codomain_t::is_faithful) {
      codomain_t codom = codomain_t::Ref(_out);
      return diff_impl(&_in, &codom, _mat);
    } else
      return diff_impl(&_in, &_out, _mat);
  }
  MapComposition() = default;

private:
  using MapBase::operator|;
  MapComposition(const MapBaseComposition &in) : MapBaseComposition(in) {}

  MapComposition(MapBaseComposition &&in) : MapBaseComposition(std::move(in)) {}

  virtual MapBaseComposition *pipe_impl(const MapBase &_that) const override {
    return new MapBaseComposition({*this, _that});
  }

  virtual MapBaseComposition *pipe_move_impl(MapBase &&_that) const override {
    return new MapBaseComposition({*this, _that});
  }
};

template <typename T>
MapComposition(const T &m)
    -> MapComposition<typename T::domain, typename T::codomain,
                      T::differential_type_id>;
template <typename T>
MapComposition(T &&m)
    -> MapComposition<typename T::domain, typename T::codomain,
                      T::differential_type_id>;
} // namespace manifolds
