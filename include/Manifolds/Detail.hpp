#pragma once
#include <Eigen/Sparse>
#include <Manifolds/ManifoldBase.hpp>
#include <functional>
#include <memory>
#include <variant>

struct DifferentialRepresentation {
  enum { dense = 0, sparse, mixed };
};

template <long Rows, long Cols>
using DenseMatrix =
    Eigen::Matrix<double, (Rows * Cols < 20000) ? Rows : Eigen::Dynamic,
                  (Rows * Cols < 20000) ? Cols : Eigen::Dynamic>;

using DenseMatrixRef = Eigen::Ref<Eigen::MatrixXd>;

using DenseMatrixConstRef = Eigen::Ref<const Eigen::MatrixXd>;

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

using SparseMatrixRef =
    std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>>;

using SparseMatrixConstRef =
    std::reference_wrapper<const Eigen::SparseMatrix<double, Eigen::RowMajor>>;

template <long Rows, long Cols>
using MixedMatrix = std::variant<Eigen::Matrix<double, Rows, Cols>,
                                 Eigen::SparseMatrix<double, Eigen::RowMajor>>;

using MixedMatrixRef = std::variant<DenseMatrixRef, SparseMatrixRef>;

using MixedMatrixConstRef =
    std::variant<DenseMatrixConstRef, SparseMatrixConstRef>;

using DifferentialReprRefType = std::variant<
    Eigen::Ref<Eigen::MatrixXd>,
    std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>>>;

using DifferentialReprType =
    std::variant<Eigen::MatrixXd, Eigen::SparseMatrix<double, Eigen::RowMajor>>;

enum MatrixTypeId { Dense = 0, Sparse, Mixed };

#define __INHERIT_LIVE_CYCLE(B)                                                \
public:                                                                        \
  using B::B;                                                                  \
  using B::operator=;

#define __DEFINE_CLONE_FUNCTIONS(THISCLASS, C, B)                              \
public:                                                                        \
  std::unique_ptr<C> clone() const {                                           \
    return std::unique_ptr<C>(static_cast<C *>(clone_impl()));                 \
  }                                                                            \
  std::unique_ptr<C> move_clone() {                                            \
    return std::unique_ptr<C>(static_cast<C *>(move_clone_impl()));            \
  }                                                                            \
                                                                               \
protected:                                                                     \
  virtual THISCLASS *clone_impl() const override {                             \
    static_assert(std::is_base_of_v<THISCLASS, C>,                             \
                  "INHERITANCE HELPER ERROR " #THISCLASS                       \
                  " is not a base of " #C);                                    \
    return new C(*static_cast<const C *>(this));                               \
  }                                                                            \
                                                                               \
  virtual THISCLASS *move_clone_impl() override {                              \
    static_assert(std::is_base_of_v<THISCLASS, C>,                             \
                  "INHERITANCE HELPER ERROR " #THISCLASS                       \
                  " is not a base of " #C);                                    \
    return new C(std::move(*static_cast<C *>(this)));                          \
  }

#define __DEFAULT_LIVE_CYCLE(C)                                                \
public:                                                                        \
  C() = default;                                                               \
  C(const C &) = default;                                                      \
  C(C &&) = default;                                                           \
  C &operator=(const C &) = default;                                           \
  C &operator=(C &&) = default;                                                \
  virtual ~C() = default;

#define __DEFAULT_REF(C, B)                                                    \
public:                                                                        \
  static constexpr C Ref(typename B::Representation *in) { return C(in); }     \
  static constexpr C CRef(const typename B::Representation *in) {              \
    return C(in);                                                              \
  }                                                                            \
  static constexpr C Ref(typename B::Representation &in) { return C(&in); }    \
                                                                               \
  static constexpr C CRef(const typename B::Representation &in) {              \
    return C(&in);                                                             \
  }

#define THROW_OUTSIDE_DOMAIN                                                   \
  trow std::logic_error(                                                       \
      "The function is a partial function. You cannot introduce any values");
