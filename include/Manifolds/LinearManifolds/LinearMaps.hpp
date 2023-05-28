#pragma once
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <cstddef>
#include <type_traits>
namespace manifolds {

template <typename Current, typename Base,
          MatrixTypeId DT = MatrixTypeId::Mixed>
class LinearMapInheritanceHelper : public Base {

public:
  __INHERIT_LIVE_CYCLE(Base)
  __DEFAULT_LIVE_CYCLE(LinearMapInheritanceHelper)
  __DEFINE_CLONE_FUNCTIONS(LinearMapInheritanceHelper, Current, Base)

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
  virtual MatrixTypeId differential_type() const override { return DT; }
};

template <typename Domain, typename Codomain>
class LinearMap : public LinearManifoldInheritanceHelper<
                      LinearMap<Domain, Codomain>,
                      MatrixManifold<Codomain::dimension, Domain::dimension>>,
                  public LinearMapInheritanceHelper<LinearMap<Domain, Codomain>,
                                                    Map<Domain, Codomain>> {

  // ------------------------
  // --- Static assertions --
  // ------------------------
  static_assert(std::is_base_of_v<LinearManifold<Domain::dimension>, Domain>,
                "Linear map must act between vector spaces");

  static_assert(
      std::is_base_of_v<LinearManifold<Codomain::dimension>, Codomain>,
      "Linear map must act between vector spaces");

  using DenseMatrixType = DenseMatrix<Codomain::tangent_repr_dimension,
                                      Domain::tangent_repr_dimension>;
  using DenseMatrixTypeRef = DenseMatrixRef;
  using DenseMatrixTypeConstRef = DenseMatrixConstRef;
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

template <typename Domain, typename Codomain>
class DenseLinearMap
    : public LinearMapInheritanceHelper<DenseLinearMap<Domain, Codomain>,
                                        LinearMap<Domain, Codomain>,
                                        MatrixTypeId::Dense> {
  using base_t = LinearMapInheritanceHelper<DenseLinearMap<Domain, Codomain>,
                                            LinearMap<Domain, Codomain>,
                                            MatrixTypeId::Dense>;

public:
  using T = DenseMatrix<Codomain::dimension, Domain::dimension>;
  DenseLinearMap() : base_t(T()) {}
  DenseLinearMap(const T &in) : base_t(in) {}
  DenseLinearMap(SparseMatrixConstRef in) : base_t(in.get().toDense()) {}

  DenseLinearMap(const DenseLinearMap &that) : base_t(that) {}
  DenseLinearMap(DenseLinearMap &&that) : base_t(std::move(that)) {}

  DenseLinearMap &operator=(DenseMatrixRef in) {
    this->repr() = in;
    return *this;
  }
  DenseLinearMap &operator=(const DenseLinearMap &in) {
    base_t::operator=(in);
    return *this;
  }
  DenseLinearMap &
  operator=(const MatrixManifold<Codomain::dimension, Domain::dimension> &in) {
    base_t::operator=(std::get<T>(in.crepr()));
    return *this;
  }

  virtual DifferentialReprType linearization_buffer() const override {
    return T();
  }

private:
  using base_t::operator=;
};

template <typename Domain, typename Codomain>
class SparseLinearMap
    : public LinearMapInheritanceHelper<SparseLinearMap<Domain, Codomain>,
                                        Map<Domain, Codomain>,
                                        MatrixTypeId::Dense> {
  using base_t =
      LinearMapInheritanceHelper<SparseLinearMap<Domain, Codomain>,
                                 Map<Domain, Codomain>, MatrixTypeId::Dense>;

public:
  using T = DenseMatrix<Codomain::dimension, Domain::dimension>;
  SparseLinearMap() : base_t(T()) {}
  SparseLinearMap(DenseMatrixConstRef in) : base_t(in) {}
  SparseLinearMap(SparseMatrixConstRef in) : base_t(in.get().toDense()) {}

  SparseLinearMap(const SparseLinearMap &that) : base_t(that) {}
  SparseLinearMap(SparseLinearMap &&that) : base_t(std::move(that)) {}

  SparseLinearMap &operator=(DenseMatrixRef in) {
    this->repr() = in;
    return *this;
  }
  SparseLinearMap &operator=(const SparseLinearMap &in) {
    base_t::operator=(in);
    return *this;
  }
  SparseLinearMap &
  operator=(const MatrixManifold<Codomain::dimension, Domain::dimension> &in) {
    base_t::operator=(std::get<T>(in.crepr()));
    return *this;
  }

  virtual DifferentialReprType linearization_buffer() const override {
    if constexpr (DenseMatrixRef::RowsAtCompileTime == Eigen::Dynamic) {
      // Is the matrix is so large, return an error if this has a dense matrix
      if (std::holds_alternative<DenseMatrix>(this->crepr())) {
        throw std::logic_error("Matrix is too large to be dense");
        return Eigen::MatrixXd();
      }
      return Eigen::SparseMatrix<double>();
    } else {
      if (std::holds_alternative<DenseMatrix>(this->crepr())) {
        return Eigen::MatrixXd();
      }
      return Eigen::SparseMatrix<double>();
    }
  }

private:
  using base_t::operator=;
};

using End3 = LinearMap<R3, R3>;

} // namespace manifolds
