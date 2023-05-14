#pragma once

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
template <typename DomainType, typename CoDomainType, bool IsDiffSparse>
class Map : virtual public MapBase {
  static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>,
                "The codomain must inherit from ManifoldBase");
  static_assert(std::is_base_of_v<ManifoldBase, DomainType>,
                "The domain must intherit from ManifoldBase");

public:
  using Domain_t = DomainType;
  using Codomain_t = CoDomainType;

  static constexpr std::size_t domain_dimension = DomainType::dimension;
  static constexpr std::size_t codomain_dimension = CoDomainType::dimension;

  using Differential_t =
      typename detail::DT<IsDiffSparse, Codomain_t::tangent_repr_dimension,
                          Domain_t::tangent_repr_dimension>::Type;

  // -------------------------------------
  // Default lifecycle
  // -------------------------------------
  Map() = default;
  Map(const Map &_that) = default;
  Map(Map &&_that) = default;
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
  value(const typename DomainType::Representation &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  value(const typename DomainType::Representation &_in) const {
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
  template <bool F = (not DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool> value(const DomainType &_in,
                                  CoDomainType &_out) const {
    return value_on_repr(_in.crepr(), _out.repr());
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::Representation &_in,
        CoDomainType &_out) const {
    return value_on_repr(_in, _out.repr());
  }

  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const typename DomainType::Representation &_in,
        typename CoDomainType::Representation &_out) const {
    return value_on_repr(_in, _out);
  }
  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  value(const DomainType &_in,
        typename CoDomainType::Representation &_out) const {
    return value_on_repr(_in.crepr(), _out);
  }

  // ---------------------------------
  // Defintion of bool operator(in, out)
  // ---------------------------------
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, bool> operator()(const DomainType &_in,
                                       CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }

  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const typename DomainType::Representation &_in,
             CoDomainType &_out) const {
    value(_in, _out);
    return true;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const typename DomainType::Representation &_in,
             typename CoDomainType::Representation &_out) const {
    return value(_in, _out);
  }

  template <bool F = (not DomainType::is_faithfull and
                      CoDomainType::is_faithfull)>
  std::enable_if_t<F, bool>
  operator()(const DomainType &_in,
             typename CoDomainType::Representation &_out) const {
    return value(_in, _out);
  }

  // ---------------------------------
  // Defintion of out operator(in)
  // ---------------------------------
  template <bool F = (not CoDomainType::is_faithfull and
                      not DomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType> operator()(const DomainType &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and
                      not CoDomainType::is_faithfull)>
  std::enable_if_t<F, CoDomainType>
  operator()(const typename DomainType::Representation &_in) const {
    CoDomainType result;
    value(_in, result);
    return result;
  }
  template <bool F = (DomainType::is_faithfull and CoDomainType::is_faithfull)>
  std::enable_if_t<F, typename CoDomainType::Representation>
  operator()(const typename DomainType::Representation &_in) const {
    typename CoDomainType::Representation result;
    value(_in, result);
    return result;
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
  DifferentialReprType linearization_buffer() const override {
    if constexpr (IsDiffSparse) {
      return Differential_t(Codomain_t::tangent_repr_dimension,
                            Domain_t::tangent_repr_dimension);
    } else {
      return Differential_t();
    }
  }
  bool is_differential_sparse() const override { return IsDiffSparse; }

  // ---------------------------------
  // Defintion of out diff(in)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull)>
  std::enable_if_t<F, Differential_t> diff(const DomainType &_in) const {
    DifferentialReprType result = this->linearization_buffer();
    if constexpr (IsDiffSparse) {
      diff(_in, std::get<1>(result));
      return std::get<1>(result);
    } else {
      diff(_in, std::get<0>(result));
      return std::get<0>(result);
    }
  }

  template <bool F = (DomainType::is_faithfull)>
  std::enable_if_t<F, Differential_t>
  diff(const typename DomainType::Representation &_in) const {
    DifferentialReprType result = this->linearization_buffer();
    if constexpr (IsDiffSparse) {
      diff(_in, std::get<1>(result));
      return std::get<1>(result);
    } else {
      diff(_in, std::get<0>(result));
      return std::get<0>(result);
    }
  }

  // ---------------------------------
  // Defintion of bool diff(in, out)
  // ---------------------------------
  template <bool F = (not DomainType::is_faithfull)>
  std::enable_if_t<F, bool> diff(const DomainType &_in,
                                 Differential_t _out) const {
    if constexpr (IsDiffSparse) {
      return diff_from_repr(_in.crepr(), std::get<1>(_out));
    } else
      return diff_from_repr(_in.crepr(), std::get<0>(_out));
  }
  template <bool F = DomainType::is_faithfull>
  std::enable_if_t<F, bool> diff(const typename DomainType::Representation &_in,
                                 DifferentialReprRefType _out) const {
    if constexpr (IsDiffSparse) {
      return diff_from_repr(_in, std::get<1>(_out));
    } else
      return diff_from_repr(_in, std::get<0>(_out));
  }

  template <typename OtherDomainType, bool OtherDiffIsSparse>
  MapComposition<CoDomainType, OtherDomainType,
                 IsDiffSparse && OtherDiffIsSparse>
  compose(
      const Map<DomainType, OtherDomainType, OtherDiffIsSparse> &_in) const {
    static_assert(std::is_base_of_v<ManifoldBase, CoDomainType>);
    static_assert(std::is_base_of_v<ManifoldBase, DomainType>);

    auto a = MapComposition < CoDomainType, DomainType,
         IsDiffSparse && OtherDiffIsSparse > (*this);

    return a.compose(_in);
  }

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

    return value_on_repr(static_cast<const DomainType *>(_in)->crepr(),
                         static_cast<CoDomainType *>(_other)->repr());
  }

  /// Implementation of differentiation of for MapBase. This is a
  /// representaiton-agnostic evaluation.
  bool diff_impl(const ManifoldBase *_in,
                 DifferentialReprRefType _mat) const override {
    if constexpr (IsDiffSparse)
      return diff_from_repr(static_cast<const DomainType *>(_in)->crepr(),
                            std::get<1>(_mat));
    else
      return diff_from_repr(static_cast<const DomainType *>(_in)->crepr(),
                            std::get<0>(_mat));
  }

  /// Get a pointer with an instance of the domain. Override if you have a
  /// custom domain
  virtual ManifoldBase *domain_buffer_impl() const override {
    return new Domain_t();
  }

  /// Get a pointer with an instance of the codomain. Override if you have
  /// custom codomain
  virtual ManifoldBase *codomain_buffer_impl() const override {
    return new Codomain_t();
  }

protected:
  /// Function to copmute the result of the map using just the representation
  /// types
  virtual bool
  value_on_repr(const typename DomainType::Representation &_in,
                typename CoDomainType::Representation &_result) const = 0;

  /// Function to copmute the result of the map differential using just the
  /// representation types
  virtual bool diff_from_repr(
      const typename DomainType::Representation &_in,
      detail::DifferentialReprRef_t<IsDiffSparse, Domain_t::dimension,
                                    Codomain_t::dimension>
          _mat) const = 0;
};

template <typename Set>
class Identity
    : public MapInheritanceHelper<Identity<Set>, Map<Set, Set, false>> {
public:
  static_assert(std::is_base_of_v<ManifoldBase, Set>);
  Identity() = default;
  Identity(const Identity &_that) = default;
  Identity(Identity &&_that) = default;

protected:
  bool value_on_repr(const typename Set::Representation &_in,
                     typename Set::Representation &_result) const override {

    _result = _in;
    return true;
  }
  bool diff_from_repr(const typename Set::Representation &,
                      Eigen::Ref<Eigen::MatrixXd> _mat) const override {
    _mat.noalias() = Map<Set, Set, false>::Differential_t::Identity(
        Set::tangent_repr_dimension, Set::tangent_repr_dimension);
    return true;
  }
};

}; // namespace manifolds
