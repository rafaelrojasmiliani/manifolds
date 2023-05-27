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
namespace detail {

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

} // namespace detail
template <typename DomainType, typename CoDomainType>
class Map : virtual public MapBase {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>,
                "The codomain must inherit from ManifoldBase");
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>,
                "The domain must intherit from ManifoldBase");

public:
  using domain = DomainType;
  using codomain = CoDomainType;

  static constexpr std::size_t domain_dimension = DomainType::dimension;
  static constexpr std::size_t codomain_dimension = CoDomainType::dimension;

  // -------------------------------------
  // Default lifecycle
  // -------------------------------------
  virtual ~Map() = default;

  // Clone
  std::unique_ptr<Map> clone() const {
    return std::unique_ptr<Map>(clone_impl());
  }

  std::unique_ptr<Map> move_clone() {
    return std::unique_ptr<Map>(move_clone_impl());
  }

  // Defintioon of out value(in)
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType> value(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType>
  value(const typename DomainType::RepresentationRef _in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  value(const typename DomainType::RepresentationRef _in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }
  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  value(const DomainType &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }
  // ---------------------------------
  // Defintion of bool value(in, out)
  // ---------------------------------
  bool value(const DomainType &_in, CoDomainType &_out) const {
    return value_on_repr(_in.crepr_ref(), _out.repr_ref());
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::RepresentationRef _in,
        CoDomainType &_out) const {
    return value_on_repr(_in, _out.repr());
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::RepresentationRef _in,
        typename CoDomainType::RepresentationRef _out) const {
    return value_on_repr(_in, _out);
  }
  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const DomainType &_in,
        typename CoDomainType::RepresentationRef _out) const {
    return value_on_repr(_in.crepr(), _out);
  }

  // ---------------------------------
  // Defintion of bool operator(in, out)
  // ---------------------------------
  bool operator()(const DomainType &_in, CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }

  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(typename DomainType::RepresentationConstRef _in,
             CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(typename DomainType::RepresentationConstRef _in,
             typename CoDomainType::RepresentationRef _out) const {
    return value(_in, _out);
  }

  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const DomainType &_in,
             typename CoDomainType::RepresentationRef _out) const {
    return value(_in, _out);
  }

  // ---------------------------------
  // Defintion of out operator(in)
  // ---------------------------------
  CoDomainType operator()(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType>
  operator()(const typename DomainType::RepresentationRef _in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  operator()(const typename DomainType::RepresentationRef _in) const {
    CoDomainType result;
    value(_in, result.repr_ref());
    return result.repr();
  }

  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  operator()(const DomainType &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
  }

  // ---------------------------------
  // Differntial
  // ---------------------------------

  // ---------------------------------
  // Defintion of bool diff(in, out)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull)>
  std::enable_if_t<F, bool> diff(const DomainType &_in,
                                 DifferentialReprRefType _out) const {
    return diff_from_repr(_in.crepr(), _out);
  }

  template <bool F = DomainType::is_faithfull>
  std::enable_if_t<F, bool>
  diff(const typename DomainType::RepresentationRef _in,
       DifferentialReprRefType _out) const {
    return diff_from_repr(_in, _out);
  }

  // ---------------------------------
  // Defintion of Composition
  // ---------------------------------
  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType>
  compose(const Map<DomainType, OtherDomainType> &_in) const {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

    auto a = MapComposition<CoDomainType, DomainType>(*this);

    return a.compose(_in);
  }

  // ---------------------------------
  // Getters
  // ---------------------------------
  virtual std::size_t get_dom_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::dimension;
  }
  virtual std::size_t get_codom_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::dimension;
  }
  virtual std::size_t get_dom_tangent_repr_dim() const override {
    // if (DomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return DomainType::tangent_repr_dimension;
  }
  virtual std::size_t get_codom_tangent_repr_dim() const override {
    // if (CoDomainType::dim == Eigen::Dynamic)
    //    throw std::invalid_input
    return CoDomainType::tangent_repr_dimension;
  }

protected:
  /// Implementation of differentiation of for MapBase. This is a
  /// representaiton-agnostic evaluation.
  bool value_impl(const ManifoldBase *_in,
                  ManifoldBase *_other) const override {

    return value_on_repr(static_cast<const DomainType *>(_in)->crepr_ref(),
                         static_cast<CoDomainType *>(_other)->repr_ref());
  }

  /// Implementation of differentiation of for MapBase. This is a
  /// representaiton-agnostic evaluation.
  bool diff_impl(const ManifoldBase *_in,
                 DifferentialReprRefType _mat) const override {
    return diff_from_repr(static_cast<const DomainType *>(_in)->crepr_ref(),
                          _mat);
  }

  /// Get a pointer with an instance of the domain. Override if you have a
  /// custom domain
  virtual ManifoldBase *domain_buffer_impl() const override {
    return new domain();
  }

  /// Get a pointer with an instance of the codomain. Override if you have
  /// custom codomain
  virtual ManifoldBase *codomain_buffer_impl() const override {
    return new codomain();
  }

protected:
  /// Function to copmute the result of the map using just the representation
  /// types
  virtual bool
  value_on_repr(typename DomainType::RepresentationConstRef _in,
                typename CoDomainType::RepresentationRef _result) const = 0;

  /// Function to copmute the result of the map differential using just the
  /// representation types
  virtual bool diff_from_repr(typename DomainType::RepresentationConstRef _in,
                              DifferentialReprRefType _mat) const = 0;
};

// ------------------------------
// Inheritance Helper -----------
// ------------------------------
//
//

/// Definition of inheritance helper for maps
template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class MapInheritanceHelper {};

/// Specialization for dense matrices
template <typename Current, typename Base>
class MapInheritanceHelper<Current, Base, MatrixTypeId::Dense> : public Base {
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(MapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(Current, Base)

private:
  virtual bool
  diff_from_repr(const typename Current::DomainType::RepresentationRef _in,
                 DifferentialReprRefType _mat) const override {
    return diff_from_repr(_in, std::get<1>(_mat));
  }

public:
  virtual bool
  diff_from_repr(const typename Current::DomainType::RepresentationRef _in,
                 Eigen::Ref<Eigen::MatrixXd> _mat) const = 0;

  virtual MatrixTypeId differential_type() const override {
    return MatrixTypeId::Dense;
  }

  virtual DifferentialReprType linearization_buffer() const override {
    return Eigen::Matrix<double, Current::domain::tangent_repr_dimension,
                         Current::codomain::tangent_repr_dimension>();
  }
};

/// Specialization for sparce matrices
template <typename Current, typename Base>
class MapInheritanceHelper<Current, Base, MatrixTypeId::Sparse> : public Base {
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(MapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(Current, Base)

private:
  virtual bool
  diff_from_repr(const typename Current::DomainType::RepresentationRef _in,
                 DifferentialReprRefType _mat) const override {
    return diff_from_repr(_in, std::get<0>(_mat));
  }

public:
  virtual bool diff_from_repr(
      const typename Current::DomainType::RepresentationRef _in,
      std::reference_wrapper<Eigen::SparseMatrix<double>> _mat) const = 0;

  virtual MatrixTypeId differential_type() const override {
    return MatrixTypeId::Sparse;
  }

  virtual DifferentialReprType linearization_buffer() const override {
    Eigen::SparseMatrix<double> result;
    result.resize(Current::domain::tangent_repr_dimension,
                  Current::codomain::tangent_repr_dimension);
    return result;
  }
};

/// Specialization for mixed type of matrices (variant)
template <typename Current, typename Base>
class MapInheritanceHelper<Current, Base, MatrixTypeId::Mixed> : public Base {
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(MapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(Current, Base)

public:
  virtual MatrixTypeId differential_type() const override {
    return MatrixTypeId::Mixed;
  }
};

// ------------------------
// ---- Identity ---------
template <typename Set>
class Identity : public MapInheritanceHelper<Identity<Set>, Map<Set, Set>> {
public:
  static_assert(std::is_base_of_v<ManifoldBase, Set>);
  Identity() = default;
  Identity(const Identity &_that) = default;
  Identity(Identity &&_that) = default;

protected:
  bool value_on_repr(const typename Set::RepresentationRef _in,
                     typename Set::RepresentationRef _result) const override {

    _result = _in;
    return true;
  }
  bool diff_from_repr(const typename Set::RepresentationRef,
                      Eigen::Ref<Eigen::MatrixXd>) const override {
    return true;
  }
};

}; // namespace manifolds
