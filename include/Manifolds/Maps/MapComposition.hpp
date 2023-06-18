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
  using Type = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using RefType =
      std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>>;
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

  template <typename A, typename B> friend class MapComposition;

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
  template <typename T>
  auto compose(const T &_in)
      const & -> MapComposition<typename T::domain, CoDomainType> {
    static_assert(
        std::is_base_of_v<Map<typename T::domain, typename T::codomain>, T>);
    MapBaseComposition result(_in.clone());
    result.append(_in);
    return result;
  }

  template <typename OtherDomainType>
  MapComposition<OtherDomainType, CoDomainType>
  compose(MapComposition<OtherDomainType, DomainType> &&_in) const & {
    MapComposition<OtherDomainType, CoDomainType> result(*this);
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
    DomainType m1 = DomainType::CRef(&_in);
    CoDomainType m2 = CoDomainType::Ref(&_result);
    return value_impl(&m1, &m2);
  }
  virtual bool diff_from_repr(const typename DomainType::Representation &_in,
                              DifferentialReprRefType _mat) const override {
    DomainType m1 = DomainType::CRef(_in);
    return diff_impl(&m1, _mat);
  }
  MapComposition() = default;

private:
  MapComposition(const MapBaseComposition &in) : MapBaseComposition(in) {}

  MapComposition(MapBaseComposition &&in) : MapBaseComposition(std::move(in)) {}
};
template <typename T>
MapComposition(const T &m)
    -> MapComposition<typename T::domain, typename T::codomain>;
template <typename T>
MapComposition(T &&m)
    -> MapComposition<typename T::domain, typename T::codomain>;
/// Move-Constructor from a typed map
} // namespace manifolds
