#pragma once
#include <Eigen/Sparse>
#include <Manifolds/ManifoldBase.hpp>
#include <functional>
#include <memory>
#include <variant>
namespace manifolds {
namespace detail {

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

using DifferentialReprRefType = std::variant<
    Eigen::Ref<Eigen::MatrixXd>,
    std::reference_wrapper<Eigen::SparseMatrix<double, Eigen::RowMajor>>>;

using mixed_matrix_t =
    std::variant<Eigen::MatrixXd, Eigen::SparseMatrix<double, Eigen::RowMajor>>;

enum MatrixTypeId { Dense = 0, Sparse };

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
  std::unique_ptr<Current> clone() const {
    return std::unique_ptr<Current>(static_cast<Current *>(this->clone_impl()));
  }
  std::unique_ptr<Current> move_clone() {
    return std::unique_ptr<Current>(static_cast<Current *>(move_clone_impl()));
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
} // namespace detail
} // namespace manifolds
