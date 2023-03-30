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
};
template <std::size_t DomainDim, std::size_t CodomainDim>
struct DT<false, DomainDim, CodomainDim> {
  using Type = Eigen::Matrix<double, CodomainDim, DomainDim>;
};
} // namespace detail_composition
template <typename DomainType, typename CoDomainType, bool IsDiffSparse>
class Map;

/// Typed composition of maps.
template <typename DomainType, typename CoDomainType, bool IsDiffSparse>
class MapComposition : public Map<DomainType, CoDomainType, IsDiffSparse>,
                       public MapBaseComposition {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

protected:
public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;
  using Differential_t =
      typename detail_composition::DT<IsDiffSparse,
                                      Codomain_t::tangent_repr_dimension,
                                      Domain_t::tangent_repr_dimension>::Type;
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
    return Map<DomainType, CoDomainType, IsDiffSparse>::domain_buffer_impl();
  }
  ManifoldBase *codomain_buffer_impl() const override {
    return Map<DomainType, CoDomainType, IsDiffSparse>::codomain_buffer_impl();
  }

  // -------------------------------------------
  // -------- Live cycle -----------------------
  // -------------------------------------------

  /// Constructor from a typed map
  MapComposition(const Map<DomainType, CoDomainType, IsDiffSparse> &m)
      : MapBaseComposition(m) {}
  /// Move-Constructor from a typed map
  MapComposition(Map<DomainType, CoDomainType, IsDiffSparse> &&m)
      : MapBaseComposition(std::move(m)) {}

  /// Copy constructor from a typed map
  MapComposition(
      const MapComposition<DomainType, CoDomainType, IsDiffSparse> &m)
      : MapBaseComposition(m) {}

  /// Move constructor from a typed map
  MapComposition(MapComposition<DomainType, CoDomainType, IsDiffSparse> &&m)
      : MapBaseComposition(std::move(m)) {}

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
  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType,
                 IsDiffSparse && OtherDiffIsSparse>
  compose(
      const Map<DomainType, OtherDomainType, OtherDiffIsSparse> &_in) const & {
    MapComposition<DomainType, CoDomainType, IsDiffSparse && OtherDiffIsSparse>
        result(*this);
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType,
                 IsDiffSparse && OtherDiffIsSparse>
  compose(MapComposition<DomainType, OtherDomainType, OtherDiffIsSparse> &&_in)
      const & {
    MapComposition<CoDomainType, OtherDomainType,
                   IsDiffSparse && OtherDiffIsSparse>
        result(*this);
    result.append(std::move(_in));
    return result;
  }

  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType,
                 IsDiffSparse && OtherDiffIsSparse>
  compose(const MapComposition<DomainType, OtherDomainType, OtherDiffIsSparse>
              &_in) && {
    MapComposition<CoDomainType, OtherDomainType,
                   IsDiffSparse && OtherDiffIsSparse>
        result(std::move(*this));
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType,
                 IsDiffSparse && OtherDiffIsSparse>
  compose(
      MapComposition<DomainType, OtherDomainType, OtherDiffIsSparse> &&_in) && {
    MapComposition<CoDomainType, OtherDomainType,
                   IsDiffSparse && OtherDiffIsSparse>
        result(std::move(*this));
    result.append(std::move(_in));
    return result;
  }

  DifferentialReprType linearization_buffer() const override {
    if constexpr (IsDiffSparse) {
      return Differential_t(CoDomainType::tangent_repr_dimension,
                            DomainType::tangent_repr_dimension);
    } else {
      return Differential_t();
    }
  }
  bool is_differential_sparse() const { return IsDiffSparse; }

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
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const override {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);
    auto m1 = DomainType::Ref(_in);
    auto m2 = CoDomainType::Ref(_result);
    return value_impl(&m1, &m2);
  }
  virtual bool diff_from_repr(const typename DomainType::Representation &_in,
                              DifferentialReprRefType _mat) const override {
    DomainType m1(_in);
    return diff_impl(&m1, _mat);
  }
  MapComposition() = default;
};
} // namespace manifolds
