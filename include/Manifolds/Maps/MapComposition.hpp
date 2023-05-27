#pragma once
#include <Manifolds/Manifold.hpp>
#include <Manifolds/Maps/MapBaseComposition.hpp>

namespace manifolds {

namespace detail_composition {

// -------------------------------------
/// Diferential type snifae
// -------------------------------------
template <bool Val, std::size_t DomainDim, std::size_t CodomainDim>
struct DT {};
template <std::size_t DomainDim, std::size_t CodomainDim>
struct DT<true, DomainDim, CodomainDim> {
  using Type = Eigen::SparseMatrix<double>;
  using RefType = std::reference_wrapper<Eigen::SparseMatrix<double>>;
};
template <std::size_t DomainDim, std::size_t CodomainDim>
struct DT<false, DomainDim, CodomainDim> {
  using Type = Eigen::Matrix<double, CodomainDim, DomainDim>;
  using RefType = Eigen::Ref<Eigen::MatrixXd>;
};

template <bool IsDiffSparse, std::size_t DomainDim, std::size_t CodomainDim>
using DifferentialRepr_t =
    typename DT<IsDiffSparse, DomainDim, CodomainDim>::Type;
template <bool IsDiffSparse, std::size_t DomainDim = 0,
          std::size_t CodomainDim = 0>
using DifferentialReprRef_t =
    typename DT<IsDiffSparse, DomainDim, CodomainDim>::RefType;

} // namespace detail_composition
template <typename DomainType, typename CoDomainType> class Map;

/// Typed composition of maps.
template <typename DomainType, typename CoDomainType>
class MapComposition : public Map<DomainType, CoDomainType>,
                       public MapBaseComposition {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

protected:
public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;

  using map_t = Map<DomainType, CoDomainType>;

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

  ManifoldBase *domain_buffer_impl() const override {
    return map_t::domain_buffer_impl();
  }
  ManifoldBase *codomain_buffer_impl() const override {
    return map_t::codomain_buffer_impl();
  }

  // -------------------------------------------
  // -------- Live cycle -----------------------
  // -------------------------------------------

  /// Constructor from a typed map
  MapComposition(const map_t &m) : MapBaseComposition(m) {}
  /// Move-Constructor from a typed map
  MapComposition(map_t &&m) : MapBaseComposition(std::move(m)) {}

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
  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const Map<DomainType, OtherDomainType> &_in) const & {
    MapComposition<DomainType, CoDomainType> result(*this);
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(MapComposition<DomainType, OtherDomainType> &&_in) const & {
    MapComposition<CoDomainType, OtherDomainType> result(*this);
    result.append(std::move(_in));
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const MapComposition<DomainType, OtherDomainType> &_in) && {
    MapComposition<CoDomainType, OtherDomainType> result(std::move(*this));
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<CoDomainType, OtherDomainType>
  compose(MapComposition<DomainType, OtherDomainType> &&_in) && {
    MapComposition<CoDomainType, OtherDomainType> result(std::move(*this));
    result.append(std::move(_in));
    return result;
  }

  // -------------------------------------------
  // -------- Buffer of the differential -------
  // -------------------------------------------
  DifferentialReprType linearization_buffer() const override {
    return Differential_t(CoDomainType::tangent_repr_dimension,
                          DomainType::tangent_repr_dimension);
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
                 DifferentialReprRefType _mat) const override {
    return MapBaseComposition::diff_impl(_in, _mat);
  }
  virtual MapComposition *clone_impl() const override {
    return new MapComposition(*this);
  }

  virtual MapComposition *move_clone_impl() override {
    return new MapComposition(std::move(*(this)));
  }
  virtual bool value_on_repr(
      const typename DomainType::RepresentationRef _in,
      typename CoDomainType::RepresentationRef _result) const override {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);
    auto m1 = DomainType::Ref(_in);
    auto m2 = CoDomainType::Ref(_result);
    return value_impl(&m1, &m2);
  }
  virtual bool diff_from_repr(const typename DomainType::RepresentationRef _in,
                              DifferentialReprRefType _mat) const override {
    typename DomainType::Ref m1(_in);
    return diff_impl(&m1, _mat);
  }
  MapComposition() = default;
};
} // namespace manifolds
