#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <type_traits>
namespace manifolds {

template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class MixedLinearMapInheritanceHelper : public Base {

public:
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(MixedLinearMapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(MixedLinearMapInheritanceHelper, Current, Base)

private:
  virtual bool diff_from_repr(const typename Base::domain::Representation &,
                              DifferentialReprRefType _mat) const override {

    const auto &this_mat = static_cast<const Current *>(this)->crepr();
    std::visit(
        [](auto &res, const auto &in) { matrix_manifold_assing(res, in); },
        _mat, this_mat);
    return true;
  }

  virtual bool value_on_repr(
      const typename Base::domain::Representation &_in,
      typename Base::codomain::Representation &_result) const override {
    const auto &this_mat = static_cast<const Current *>(this)->crepr();
    std::visit(
        [](const auto &mat, const auto &in, auto &res) {
          auto &result = matrix_manifold_ref_to_type(res);
          const auto &matrix = matrix_manifold_ref_to_type(mat);
          const auto &vec = matrix_manifold_ref_to_type(in);

          if constexpr (
              std::is_same_v<std::decay_t<decltype(result)>,
                             Eigen::SparseMatrix<double, Eigen::RowMajor>> and
              (std::is_same_v<std::decay_t<decltype(matrix)>,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>> xor
               std::is_same_v<std::decay_t<decltype(vec)>,
                              Eigen::SparseMatrix<double, Eigen::RowMajor>>)) {
            result = (matrix * vec).sparseView();
          } else if constexpr (
              std::is_same_v<std::decay_t<decltype(result)>,
                             Eigen::SparseMatrix<double, Eigen::RowMajor>> and
              (not std::is_same_v<
                   std::decay_t<decltype(matrix)>,
                   Eigen::SparseMatrix<double, Eigen::RowMajor>> and
               not std::is_same_v<
                   std::decay_t<decltype(vec)>,
                   Eigen::SparseMatrix<double, Eigen::RowMajor>>)) {
            result = (matrix * vec).sparseView();
          } else
            result = matrix * vec;
        },
        this_mat, _in, _result);
    return true;
  }

public:
  virtual MatrixTypeId differential_type() const override { return DT; }
};

template <typename Current, typename Base>

class LinearMapInheritanceHelper : public Base {

public:
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(LinearMapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(LinearMapInheritanceHelper, Current, Base)

private:
  virtual bool diff_from_repr(const typename Base::domain::Representation &,
                              DifferentialReprRefType _mat) const override {

    std::get<0>(_mat) = static_cast<const Current *>(this)->crepr();

    return true;
  }

  virtual bool value_on_repr(
      const typename Base::domain::Representation &_in,
      typename Base::codomain::Representation &_result) const override {
    _result = static_cast<const Current *>(this)->crepr() * _in;
    return true;
  }

public:
  virtual MatrixTypeId differential_type() const override {
    if constexpr (std::is_base_of_v<Eigen::Dense,
                                    typename Current::Representation>)
      return MatrixTypeId::Dense;
    else
      return MatrixTypeId::Sparse;
  }
  virtual DifferentialReprType linearization_buffer() const override {
    return typename Current::Representation();
  }
};

template <typename Domain, typename Codomain>
class MixedLinearMap
    : public MixedLinearManifoldInheritanceHelper<
          MixedLinearMap<Domain, Codomain>,
          MixedMatrixManifold<Codomain::dimension, Domain::dimension>>,
      public MixedLinearMapInheritanceHelper<MixedLinearMap<Domain, Codomain>,
                                             Map<Domain, Codomain>> {

  // ------------------------
  // --- Static assertions --
  // ------------------------
  static_assert(std::is_base_of_v<LinearManifold, Domain>,
                "Linear map must act between vector spaces");

  static_assert(std::is_base_of_v<LinearManifold, Codomain>,
                "Linear map must act between vector spaces");

  using DenseMatrixType = DenseMatrix<Codomain::tangent_repr_dimension,
                                      Domain::tangent_repr_dimension>;
  using DenseMatrixTypeRef = DenseMatrixRef;
  using DenseMatrixTypeConstRef = DenseMatrixConstRef;
  using SparseMatrixType = Eigen::SparseMatrix<double, Eigen::RowMajor>;

public:
  // ----------------------------------------------------
  // -------------------  Taype definitions -------------------
  // ----------------------------------------------------
  using base_t = MixedLinearManifoldInheritanceHelper<
      MixedLinearMap<Domain, Codomain>,
      MixedMatrixManifold<Codomain::dimension, Domain::dimension>>;

  using Representation = typename base_t::Representation;
  using DomainRepresentation = typename Domain::Representation;
  using DomainRepresentationRef = typename Domain::Representation &;
  using DomainRepresentationConstRef = const typename Domain::Representation &;
  using CodomainRepresentation = typename Codomain::Representation;
  using CodomainRepresentationRef = typename Codomain::Representation &;
  using CodomainRepresentationConstRef =
      const typename Codomain::Representation &;

  // ----------------------------------------------------
  // -------------------  Inheritance -------------------
  // ----------------------------------------------------
  using base_t::base_t;
  using base_t::operator=;

  // ----------------------------------------------------
  // -------------------  Lifecycle -------------------
  // ----------------------------------------------------
  __DEFAULT_LIVE_CYCLE(MixedLinearMap)

  // ----------------------------------------------------
  // -------------------  composition -------------------
  // ----------------------------------------------------

  template <typename OtherDomain>
  MixedLinearMap<Codomain, OtherDomain>
  compose(const MixedLinearMap<OtherDomain, Codomain> &_in) const {
    return MixedLinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }

  virtual DifferentialReprType linearization_buffer() const override {

    if (std::holds_alternative<DenseMatrixType>(this->crepr()))
      return DenseMatrixType();

    SparseMatrixType result;
    return result;
  }
};

template <typename Domain, typename Codomain> class SparseLinearMap;
template <typename Domain, typename Codomain>
class DenseLinearMap
    : public LinearManifoldInheritanceHelper<
          DenseLinearMap<Domain, Codomain>,
          DenseMatrixManifold<Codomain::dimension, Domain::dimension>>,
      public LinearMapInheritanceHelper<DenseLinearMap<Domain, Codomain>,
                                        Map<Domain, Codomain>> {
  using base_t = LinearManifoldInheritanceHelper<
      DenseLinearMap<Domain, Codomain>,
      DenseMatrixManifold<Codomain::dimension, Domain::dimension>>;

public:
  using base_t::base_t;
  using base_t::operator=;

  template <typename OtherDomain>
  DenseLinearMap<Codomain, OtherDomain>
  compose(const DenseLinearMap<OtherDomain, Codomain> &_in) const {
    return DenseLinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }

  template <typename OtherDomain>
  DenseLinearMap<Codomain, OtherDomain>
  compose(const SparseLinearMap<OtherDomain, Codomain> &_in) const;
};

template <typename Domain, typename Codomain>
class SparseLinearMap
    : public LinearManifoldInheritanceHelper<
          SparseLinearMap<Domain, Codomain>,
          SparseMatrixManifold<Codomain::dimension, Domain::dimension>>,
      public LinearMapInheritanceHelper<SparseLinearMap<Domain, Codomain>,
                                        Map<Domain, Codomain>> {

  using base_t = LinearManifoldInheritanceHelper<
      SparseLinearMap<Domain, Codomain>,
      SparseMatrixManifold<Codomain::dimension, Domain::dimension>>;

public:
  using base_t::base_t;
  using base_t::operator=;

  template <typename OtherDomain>
  DenseLinearMap<Codomain, OtherDomain>
  compose(const DenseLinearMap<OtherDomain, Codomain> &_in) const {
    return DenseLinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }

  template <typename OtherDomain>
  DenseLinearMap<Codomain, OtherDomain>
  compose(const SparseLinearMap<OtherDomain, Codomain> &_in) const {
    return SparseLinearMap<Codomain, OtherDomain>(base_t::crepr() *
                                                  _in.crepr());
  }
};

template <typename Domain, typename Codomain>
template <typename OtherDomain>
DenseLinearMap<Codomain, OtherDomain> DenseLinearMap<Domain, Codomain>::compose(
    const SparseLinearMap<OtherDomain, Codomain> &_in) const {

  return DenseLinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
}

using End3 = MixedLinearMap<R3, R3>;

} // namespace manifolds
