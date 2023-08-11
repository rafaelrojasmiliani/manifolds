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
  // FIXME adapt to dynamic type
  using differential_t = std::conditional_t<
      DT == detail::MatrixTypeId::Dense,
      detail::dense_matrix_t<codomain_t::tangent_repr_dimension,
                             domain_t::tangent_repr_dimension>,
      detail::sparse_matrix_t>;

  using map_t = Map<DomainType, CoDomainType, DT>;

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
  template <typename M> MapComposition(const M &m) : MapBaseComposition(m) {}
  /// Move-Constructor from a typed map
  template <typename M>
  MapComposition(M &&m) : MapBaseComposition(std::move(m)) {}

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
  // -------- Modifiers ------------------------
  // -------------------------------------------
  std::unique_ptr<MapBaseComposition>
  pre_compose_ptr(const std::unique_ptr<MapBase> &_other) override {
    auto foo = std::vector<std::unique_ptr<MapBase>>{};
    foo.push_back(this->clone());
    foo.push_back(_other->clone());
    return std::make_unique<MapBaseComposition>(foo);
  }

  template <typename OtherDomainType, detail::MatrixTypeId OtherDT>
  auto compose(const Map<OtherDomainType, DomainType, OtherDT> &_in) const & {
    constexpr detail::MatrixTypeId mt =
        (OtherDT == DT and DT == detail::MatrixTypeId::Sparse)
            ? detail::MatrixTypeId::Sparse
            : detail::MatrixTypeId::Dense;
    MapBaseComposition result(*this);
    result.append(_in);

    return MapComposition<OtherDomainType, CoDomainType, mt>(result);
  }

  template <typename OtherDomainType, detail::MatrixTypeId OtherDT>
  auto
  compose(MapComposition<OtherDomainType, DomainType, OtherDT> &&_in) const & {
    constexpr detail::MatrixTypeId mt =
        (OtherDT == DT and DT == detail::MatrixTypeId::Sparse)
            ? detail::MatrixTypeId::Sparse
            : detail::MatrixTypeId::Dense;
    MapBaseComposition result(*this);
    result.append(_in);
    return MapComposition<OtherDomainType, CoDomainType, mt>(result);
  }

protected:
  // -------------------------------------------
  // -------- Interface -----------------------
  // -------------------------------------------
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {
    return MapBaseComposition::value_impl(_in, _other);
  }
  bool diff_impl(const ManifoldBase *_in,
                 detail::mixed_matrix_ref_t _mat) const override {
    return MapBaseComposition::diff_impl(_in, _mat);
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
                              differential_ref_t _mat) const override {
    if constexpr (domain_t::is_faithful) {
      DomainType m1 = DomainType::CRef(_in);
      return diff_impl(&m1, _mat);
    }
    return diff_impl(&_in, _mat);
  }
  MapComposition() = default;

private:
  MapComposition(const MapBaseComposition &in) : MapBaseComposition(in) {}

  MapComposition(MapBaseComposition &&in) : MapBaseComposition(std::move(in)) {}
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
