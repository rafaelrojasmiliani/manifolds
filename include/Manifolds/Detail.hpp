#pragma once
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Sparse>
#include <Manifolds/ManifoldBase.hpp>
#include <functional>
#include <memory>
#include <variant>
namespace manifolds {
namespace detail {

constexpr int Dynamic = Eigen::Dynamic;

template <typename Base, typename... Args> struct inherits_from;

template <typename Base, typename Arg, typename... Rest>
struct inherits_from<Base, Arg, Rest...> {
  using type = std::conditional_t<std::is_base_of_v<Base, Arg>, Arg,
                                  typename inherits_from<Base, Rest...>::type>;
  static constexpr bool value =
      std::is_base_of_v<Base, Arg> || inherits_from<Base, Rest...>::value;
};

template <typename Base> struct inherits_from<Base> {
  using type = void;
  static constexpr bool value = false;
};

template <typename Base, typename... Args>
using inherits_from_t = typename inherits_from<Base, Args...>::type;

template <typename Base, typename... Args>
inline constexpr bool inherits_from_v = inherits_from<Base, Args...>::value;
struct DifferentialRepresentation {
  enum { dense = 0, sparse };
};

template <long Rows, long Cols>
using DenseMatrix =
    Eigen::Matrix<double, (Rows * Cols < 20000) ? Rows : Eigen::Dynamic,
                  (Rows * Cols < 20000) ? Cols : Eigen::Dynamic>;

using dense_matrix_ref_t = Eigen::Ref<Eigen::MatrixXd>;

template <long Rows = Eigen::Dynamic, long Cols = Eigen::Dynamic>
using dense_matrix_t = Eigen::Matrix<double, Rows, Cols>;

using dense_matrix_const_ref_t = Eigen::Ref<const Eigen::MatrixXd>;

using sparse_matrix_t = Eigen::SparseMatrix<double, Eigen::RowMajor>;

using sparse_matrix_ref_t = std::reference_wrapper<sparse_matrix_t>;

template <long Rows, long Cols>
using MixedMatrix =
    std::variant<Eigen::Matrix<double, Rows, Cols>, sparse_matrix_t>;

using mixed_matrix_ref_t =
    std::variant<dense_matrix_ref_t, sparse_matrix_ref_t>;

using DifferentialReprRefType =
    std::variant<Eigen::Ref<Eigen::MatrixXd>,
                 std::reference_wrapper<sparse_matrix_t>>;

using mixed_matrix_t = std::variant<Eigen::MatrixXd, sparse_matrix_t>;

enum MatrixTypeId { Dense = 0, Sparse };

inline mixed_matrix_ref_t mixed_matrix_to_ref(mixed_matrix_t &mm) {
  if (std::holds_alternative<Eigen::MatrixXd>(mm))
    return Eigen::Ref<Eigen::MatrixXd>(std::get<Eigen::MatrixXd>(mm));
  return sparse_matrix_ref_t(std::get<sparse_matrix_t>(mm));
}

inline Eigen::MatrixXd &mixed_matrix_to_dense(mixed_matrix_t &mm) {
  if (std::holds_alternative<Eigen::MatrixXd>(mm))
    return std::get<Eigen::MatrixXd>(mm);
  throw std::invalid_argument("");
  return std::get<Eigen::MatrixXd>(mm);
  // return sparse_matrix_ref_t(std::get<sparse_matrix_t>(mm)).get().toDense();
}

inline bool mixed_matrix_ref_has_sparse(mixed_matrix_ref_t mr) {
  return std::holds_alternative<sparse_matrix_ref_t>(mr);
}
inline bool mixed_matrix_has_sparse(mixed_matrix_t mr) {
  return std::holds_alternative<sparse_matrix_t>(mr);
}

inline void mixed_matrix_ref_mul(mixed_matrix_ref_t _m1, mixed_matrix_ref_t _m2,
                                 mixed_matrix_ref_t result) {
  std::visit(
      [&](auto &&m1, auto &&m2) {
        using m1_t = std::decay_t<decltype(m1)>;
        using m2_t = std::decay_t<decltype(m2)>;

        if constexpr (std::is_same_v<m1_t, m2_t> and
                      std::is_same_v<m1_t, sparse_matrix_ref_t>) {

          if (not mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a sparse matrix");
          std::get<sparse_matrix_ref_t>(result).get() = m1.get() * m2.get();

        } else if constexpr (std::is_same_v<m1_t, sparse_matrix_ref_t>) {

          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result) = m1.get() * m2;
        } else if constexpr (std::is_same_v<m2_t, sparse_matrix_ref_t>) {

          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result) = m1 * m2.get();
        } else {
          static_assert(std::is_same_v<m1_t, m2_t> &&
                        std::is_same_v<m1_t, Eigen::Ref<Eigen::MatrixXd>>);

          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result) = m1 * m2;
        }
      },
      _m1, _m2);
}

inline void mixed_matrix_mul(mixed_matrix_t &_m1, mixed_matrix_t &_m2,
                             mixed_matrix_t &result) {
  std::visit(
      [&](auto &&m1, auto &&m2) {
        using m1_t = std::decay_t<decltype(m1)>;
        using m2_t = std::decay_t<decltype(m2)>;

        if constexpr (std::is_same_v<m1_t, m2_t> and
                      std::is_same_v<m1_t, sparse_matrix_t>) {

          if (not mixed_matrix_has_sparse(result))
            throw std::invalid_argument("result must contain a sparse matrix");
          std::get<sparse_matrix_t>(result) = m1 * m2;

        } else if constexpr (std::is_same_v<m1_t, sparse_matrix_t>) {

          if (mixed_matrix_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<Eigen::MatrixXd>(result).noalias() = m1 * m2;
        } else if constexpr (std::is_same_v<m2_t, sparse_matrix_t>) {

          if (mixed_matrix_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<Eigen::MatrixXd>(result).noalias() = m1 * m2;
        } else {
          static_assert(std::is_same_v<m1_t, m2_t> &&
                        std::is_same_v<m1_t, Eigen::MatrixXd>);
          if (mixed_matrix_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<Eigen::MatrixXd>(result).noalias() = m1 * m2;
        }
      },
      _m1, _m2);
}

inline void mixed_matrix_mul(mixed_matrix_t &_m1, mixed_matrix_t &_m2,
                             mixed_matrix_ref_t result) {
  std::visit(
      [&](auto &&m1, auto &&m2) {
        using m1_t = std::decay_t<decltype(m1)>;
        using m2_t = std::decay_t<decltype(m2)>;

        if constexpr (std::is_same_v<m1_t, m2_t> and
                      std::is_same_v<m1_t, sparse_matrix_t>) {

          if (not mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a sparse matrix");
          /// See this https://gitlab.com/libeigen/eigen/-/issues/385
          std::get<sparse_matrix_ref_t>(result).get() = m1 * m2;

        } else if constexpr (std::is_same_v<m1_t, sparse_matrix_t>) {

          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result).noalias() = m1 * m2;
        } else if constexpr (std::is_same_v<m2_t, sparse_matrix_t>) {

          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result).noalias() = m1 * m2;
        } else {
          static_assert(std::is_same_v<m1_t, m2_t> &&
                        std::is_same_v<m1_t, Eigen::MatrixXd>);
          if (mixed_matrix_ref_has_sparse(result))
            throw std::invalid_argument("result must contain a dense matrix");
          std::get<dense_matrix_ref_t>(result).noalias() = m1 * m2;
        }
      },
      _m1, _m2);
}

inline mixed_matrix_t get_buffer_of_product(mixed_matrix_t &_m1,
                                            mixed_matrix_t &_m2) {

  return std::visit(
      [&](auto &&m1, auto &&m2) -> mixed_matrix_t {
        using m1_t = std::decay_t<decltype(m1)>;
        using m2_t = std::decay_t<decltype(m2)>;

        if constexpr (std::is_same_v<m1_t, m2_t> and
                      std::is_same_v<m1_t, sparse_matrix_t>) {

          if (m1.cols() != m2.rows())
            throw std::invalid_argument("Matrix must be able to be multiplied");

          return sparse_matrix_t(m1.rows(), m2.cols());

        } else if constexpr (std::is_same_v<m1_t, sparse_matrix_t>) {

          if (m1.cols() != m2.rows())
            throw std::invalid_argument("Matrix must be able to be multiplied");

          return Eigen::MatrixXd(m1.rows(), m2.cols());

        } else if constexpr (std::is_same_v<m2_t, sparse_matrix_t>) {

          if (m1.cols() != m2.rows())
            throw std::invalid_argument("Matrix must be able to be multiplied");

          return Eigen::MatrixXd(m1.rows(), m2.cols());

        } else {

          static_assert(std::is_same_v<m1_t, m2_t> &&
                        std::is_same_v<m1_t, Eigen::MatrixXd>);
          if (m1.cols() != m2.rows())
            throw std::invalid_argument("Matrix must be able to be multiplied");
          return Eigen::MatrixXd(m1.rows(), m2.cols());
        }
      },
      _m1, _m2);
}
// inline void &as(Eigen::MatrixXd& dense_buffer, sparse_matrix_t&
// sparse_buffer, ) {
// }

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

template <typename Current, typename... Bases>
class Clonable : public Bases... {
public:
  using Bases::Bases...;
  using Bases::operator=...;
  __DEFAULT_LIVE_CYCLE(Clonable)
public:
  using manifold_t =
      std::conditional_t<inherits_from_v<ManifoldBase, Bases...>,
                         inherits_from_t<ManifoldBase, Bases...>, void>;

  using map_t = std::conditional_t<inherits_from_v<MapBase, Bases...>,
                                   inherits_from_t<MapBase, Bases...>, void>;

  using base_t = Clonable;

  std::unique_ptr<Current> clone() const & {
    return std::unique_ptr<Current>(static_cast<Current *>(this->clone_impl()));
  }
  std::unique_ptr<Current> clone() && { return move_clone(); }

  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(
        static_cast<Current *>(this->move_clone_impl()));
  }

  template <bool B = inherits_from_v<ManifoldBase, Bases...>,
            typename T = inherits_from_t<ManifoldBase, Bases...>>
  static std::enable_if_t<B, std::unique_ptr<Current>>
  from_repr(const typename T::Representation &_in) {
    return std::unique_ptr<Current>(new Current(_in));
  }
  template <bool B = inherits_from_v<ManifoldBase, Bases...>,
            typename T = inherits_from_t<ManifoldBase, Bases...>>
  static std::enable_if_t<B, std::unique_ptr<Current>>
  from_repr(typename T::Representation &&_in) {
    return std::unique_ptr<Current>(new Current(std::move(_in)));
  }
  /* template <typename T = inherits_from_t<ManifoldBase, Bases...>, */
  /*           typename = std::enable_if<inherits_from_v<ManifoldBase,
   * Bases...>>> */
  template <bool B = inherits_from_v<ManifoldBase, Bases...>,
            typename T = inherits_from_t<ManifoldBase, Bases...>>
  static std::enable_if_t<B, std::unique_ptr<Current>> random() {
    return std::unique_ptr<Current>(new Current(T::atlas::random_projection()));
  }

private:
  virtual Clonable *clone_impl() const override {
    static_assert((std::is_base_of_v<Clonable, Current>),
                  "Clonable inheritance ERROR. Declared base is not a base");
    return new Current(static_cast<const Current &>(*this));
  }
  virtual Clonable *move_clone_impl() override {
    static_assert(std::is_base_of_v<Clonable, Current>,
                  "Clonable inheritance ERROR. Declared base is not a base");
    return new Current(static_cast<Current &&>(std::move(*this)));
  }
};

template <typename Current, typename... Bases>
class ClonableManifold : public Bases... {
public:
  using Bases::Bases...;
  using Bases::operator=...;
  __DEFAULT_LIVE_CYCLE(ClonableManifold)
public:
  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(static_cast<Current *>(this->clone_impl()));
  }
  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(static_cast<Current *>(move_clone_impl()));
  }

private:
  virtual ClonableManifold *clone_impl() const override {
    static_assert(
        (std::is_base_of_v<ClonableManifold, Current>),
        "ClonableManifold inheritance ERROR. Declared base is not a base");
    return new Current(static_cast<const Current &>(*this));
  }
  virtual ClonableManifold *move_clone_impl() override {
    static_assert(
        std::is_base_of_v<ClonableManifold, Current>,
        "ClonableManifold inheritance ERROR. Declared base is not a base");
    return new Current(static_cast<Current &&>(std::move(*this)));
  }
};

template <typename Domain, typename Codomain>
constexpr MatrixTypeId decide_matrix_type() {
  return (Domain::dimension * Codomain::dimension > 1000)
             ? MatrixTypeId::Dense
             : MatrixTypeId::Sparse;
}
} // namespace detail
} // namespace manifolds
