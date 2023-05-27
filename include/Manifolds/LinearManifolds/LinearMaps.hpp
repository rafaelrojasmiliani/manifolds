#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <type_traits>
namespace manifolds {

template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class LinearMapInheritanceHelper : public Base {

  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(LinearMapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(Current, Base)

private:
  virtual bool diff_from_repr(typename Base::domain::RepresentationConstRef,
                              DifferentialReprRefType _mat) const override {

    const auto &this_mat = static_cast<const Current *>(this)->crepr();
    std::visit(
        [](auto &res, const auto &in) { matrix_manifold_assing(res, in); },
        _mat, this_mat);
    return true;
  }

  virtual bool value_on_repr(
      typename Base::domain::RepresentationConstRef _in,
      typename Base::codomain::RepresentationRef _result) const override {
    const auto &this_mat = static_cast<const Current *>(this)->crepr();
    std::visit(
        [](const auto &mat, const auto &in, auto &res) {
          auto &result = matrix_manifold_ref_to_type(res);
          const auto &matrix = matrix_manifold_ref_to_type(mat);
          const auto &vec = matrix_manifold_ref_to_type(in);

          if constexpr (std::is_same_v<std::decay_t<decltype(result)>,
                                       Eigen::SparseMatrix<double>> and
                        (std::is_same_v<std::decay_t<decltype(matrix)>,
                                        Eigen::SparseMatrix<double>> xor
                         std::is_same_v<std::decay_t<decltype(vec)>,
                                        Eigen::SparseMatrix<double>>)) {
            result = (matrix * vec).sparseView();
          } else if constexpr (std::is_same_v<std::decay_t<decltype(result)>,
                                              Eigen::SparseMatrix<double>> and
                               (not std::is_same_v<
                                    std::decay_t<decltype(matrix)>,
                                    Eigen::SparseMatrix<double>> and
                                not std::is_same_v<
                                    std::decay_t<decltype(vec)>,
                                    Eigen::SparseMatrix<double>>)) {
            result = (matrix * vec).sparseView();
          } else
            result = matrix * vec;
        },
        this_mat, _in, _result);
    return true;
  }

public:
  virtual MatrixTypeId differential_type() const override {
    return MatrixTypeId::Dense;
  }

  virtual DifferentialReprType linearization_buffer() const override {
    return Eigen::Matrix<double, Current::domain::tangent_repr_dimension,
                         Current::codomain::tangent_repr_dimension>();
  }
};

template <typename Domain, typename Codomain>
class LinearMap : public LinearManifoldInheritanceHelper<
                      LinearMap<Domain, Codomain>,
                      MatrixManifold<Codomain::dimension, Domain::dimension>>,
                  public LinearMapInheritanceHelper<LinearMap<Domain, Codomain>,
                                                    Map<Domain, Codomain>,
                                                    MatrixTypeId::Mixed> {

  // ------------------------
  // --- Static assertions --
  // ------------------------
  static_assert(std::is_base_of_v<LinearManifold<Domain::dimension>, Domain>,
                "Linear map must act between vector spaces");

  static_assert(
      std::is_base_of_v<LinearManifold<Codomain::dimension>, Codomain>,
      "Linear map must act between vector spaces");

  using DenseMatrixType =
      Eigen::Matrix<double, Codomain::tangent_repr_dimension,
                    Domain::tangent_repr_dimension>;
  using DenseMatrixTypeRef = Eigen::Ref<Eigen::MatrixXd>;
  using DenseMatrixTypeConstRef = Eigen::Ref<const Eigen::MatrixXd>;
  using SparseMatrixType = Eigen::SparseMatrix<double>;

public:
  // ----------------------------------------------------
  // -------------------  Taype definitions -------------------
  // ----------------------------------------------------
  using base_t = LinearManifoldInheritanceHelper<
      LinearMap<Domain, Codomain>,
      MatrixManifold<Codomain::dimension, Domain::dimension>>;

  using Representation = typename base_t::Representation;
  using DomainRepresentation = typename Domain::Representation;
  using DomainRepresentationRef = typename Domain::RepresentationRef;
  using DomainRepresentationConstRef = typename Domain::RepresentationConstRef;
  using CodomainRepresentation = typename Codomain::Representation;
  using CodomainRepresentationRef = typename Codomain::RepresentationRef;
  using CodomainRepresentationConstRef =
      typename Codomain::RepresentationConstRef;

  // ----------------------------------------------------
  // -------------------  Inheritance -------------------
  // ----------------------------------------------------
  using base_t::base_t;
  using base_t::operator=;

  // ----------------------------------------------------
  // -------------------  Lifecycle -------------------
  // ----------------------------------------------------
  __DEFAULT_LIVE_CYCLE(LinearMap)

  // ----------------------------------------------------
  // -------------------  composition -------------------
  // ----------------------------------------------------

  template <typename OtherDomain>
  LinearMap<Codomain, OtherDomain>
  compose(const LinearMap<OtherDomain, Codomain> &_in) const {
    return LinearMap<Codomain, OtherDomain>(base_t::crepr() * _in.crepr());
  }

  virtual DifferentialReprType linearization_buffer() const override {

    if (std::holds_alternative<DenseMatrixType>(this->crepr()))
      return DenseMatrixType();

    SparseMatrixType result;
    return result;
  }
};

using End3 = LinearMap<R3, R3>;

} // namespace manifolds
