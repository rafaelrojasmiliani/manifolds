#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <variant>

template <typename IN> auto &&matrix_manifold_ref_to_type(IN &&in) {

  if constexpr (std::is_same_v<
                    std::decay_t<IN>,
                    std::reference_wrapper<Eigen::SparseMatrix<double>>>) {
    return std::forward<Eigen::SparseMatrix<double> &>(in.get());
  } else if constexpr (std::is_same_v<std::decay_t<IN>,
                                      std::reference_wrapper<
                                          const Eigen::SparseMatrix<double>>>) {
    return std::forward<const Eigen::SparseMatrix<double> &>(in.get());
  } else
    return std::forward<IN>(in);
}
template <typename LHS, typename RSH, typename F>
auto &&matrix_manifold_apply(LHS &&_lhs, RSH &&_rhs, F &&_fun) {

  return std::visit(_fun, _lhs, _rhs);
}

template <typename LHS, typename RHS>
void matrix_manifold_assing(LHS &&_lhs, RHS &&_rhs) {

  auto &&lhs = matrix_manifold_ref_to_type(_lhs);
  const auto &rhs = matrix_manifold_ref_to_type(_rhs);

  if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                               Eigen::SparseMatrix<double>> and
                not std::is_same_v<std::decay_t<decltype(rhs)>,
                                   Eigen::SparseMatrix<double>>)
    lhs = rhs.sparseView();
  else {
    lhs = rhs;
  }
}

template <long Rows, long Cols> class LinearManifoldAtlas {
  static constexpr long EffectiveCols =
      (Rows * Cols < 20000) ? Cols : Eigen::Dynamic;
  static constexpr long EffectiveRows =
      (Rows * Cols < 20000) ? Rows : Eigen::Dynamic;

private:
  using DenseMatrix = Eigen::Matrix<double, EffectiveRows, EffectiveCols>;
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using DenseMatrixRef = Eigen::Ref<Eigen::MatrixXd>;
  using DenseMatrixConstRef = Eigen::Ref<const DenseMatrix>;
  using SparseMatrixRef = std::reference_wrapper<Eigen::SparseMatrix<double>>;
  using SparseMatrixConstRef =
      std::reference_wrapper<const Eigen::SparseMatrix<double>>;

public:
  static const constexpr bool is_differential_sparse = false;
  using Representation = std::variant<DenseMatrix, SparseMatrix>;
  using RepresentationRef = std::variant<DenseMatrixRef, SparseMatrixRef>;
  using RepresentationConstRef =
      const std::variant<DenseMatrixConstRef, SparseMatrixConstRef>;

  static DenseMatrixConstRef cref_to_type(RepresentationConstRef &_ref) {
    if (std::holds_alternative<DenseMatrixRef>(_ref))
      return std::get<0>(_ref);
    return std::get<1>(_ref);
  }
  static const Representation *cref_to_type_ptr(RepresentationConstRef &_ref) {
    if (std::holds_alternative<DenseMatrixRef>(_ref))
      return std::get<0>(_ref);
    return std::get<1>(_ref);
  }
  static Representation &ref_to_type(RepresentationRef &_ref) {
    if (std::holds_alternative<DenseMatrixRef>(_ref))
      return (std::get<0>(_ref));
    return (std::get<1>(_ref));
  }
  static RepresentationConstRef ctype_to_ref(const Representation &_ref) {
    if (std::holds_alternative<DenseMatrix>(_ref))
      return std::get<0>(_ref);
    return std::get<1>(_ref);
  }
  static RepresentationRef type_to_ref(Representation &_ref) {
    if (std::holds_alternative<DenseMatrix>(_ref))
      return (std::get<0>(_ref));
    return (std::get<1>(_ref));
  }

  using Coordinates = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  using ChartDifferential = Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                                          EffectiveRows * EffectiveCols>;
  using ChangeOfCoordinatesDiff =
      Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                    EffectiveRows * EffectiveCols>;
  using ParametrizationDifferential =
      Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                    EffectiveRows * EffectiveCols>;

  using Tangent = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  static void chart(const Representation &, const Representation &,
                    const Representation &element, Coordinates &result) {
    result = Eigen::Map<const Coordinates>(element.data());
  }

  static void chart_diff(const Representation &, const Representation &,
                         const Representation &,
                         Eigen::Ref<ChartDifferential> result) {

    result = ChartDifferential::Identity();
  }

  static void param(const Representation &, const Representation &,
                    const Coordinates &coordinates, Representation &result) {
    result = Eigen::Map<const Representation>(coordinates.data());
  }

  static void param_diff(const Representation &, const Representation &,
                         const Coordinates &,
                         Eigen::Ref<ChartDifferential> result) {
    result = ChartDifferential::Identity();
  }

  static Representation random_projection() { return Representation::Random(); }

  static void change_of_coordinates(const Representation &,
                                    const Representation &, const Coordinates &,
                                    const Coordinates &,
                                    const Coordinates &coordinates,
                                    Coordinates &result) {
    result = coordinates;
  }

  static void
  change_of_coordinates_diff(const Representation &, const Representation &,
                             const Representation &, const Representation &,
                             const Coordinates &,
                             Eigen::Ref<ChangeOfCoordinatesDiff> result) {
    result = ChangeOfCoordinatesDiff::Identity();
  }

  static constexpr std::size_t dimension = Rows * Cols;

  static constexpr std::size_t tangent_repr_dimension = Rows * Cols;
};

template <long Rows, long Cols> class DenseLinearManifoldAtlas {

public:
  static const constexpr bool is_differential_sparse = false;
  constexpr static long EffectiveCols = Cols;
  constexpr static long EffectiveRows = Rows;
  using Representation = Eigen::Matrix<double, Rows, Cols>;
  using RepresentationRef = Representation &;

  using Coordinates = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  using ChartDifferential = Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                                          EffectiveRows * EffectiveCols>;
  using ChangeOfCoordinatesDiff =
      Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                    EffectiveRows * EffectiveCols>;
  using ParametrizationDifferential =
      Eigen::Matrix<double, EffectiveRows * EffectiveCols,
                    EffectiveRows * EffectiveCols>;

  using Tangent = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  static void chart(const Representation &, const Representation &,
                    const Representation &element, Coordinates &result) {
    result = Eigen::Map<const Coordinates>(element.data());
  }

  static void chart_diff(const Representation &, const Representation &,
                         const Representation &,
                         Eigen::Ref<ChartDifferential> result) {

    result = ChartDifferential::Identity();
  }

  static void param(const Representation &, const Representation &,
                    const Coordinates &coordinates, Representation &result) {
    result = Eigen::Map<const Representation>(coordinates.data());
  }

  static void param_diff(const Representation &, const Representation &,
                         const Coordinates &,
                         Eigen::Ref<ChartDifferential> result) {
    result = ChartDifferential::Identity();
  }

  static Representation random_projection() { return Representation::Random(); }

  static void change_of_coordinates(const Representation &,
                                    const Representation &, const Coordinates &,
                                    const Coordinates &,
                                    const Coordinates &coordinates,
                                    Coordinates &result) {
    result = coordinates;
  }

  static void
  change_of_coordinates_diff(const Representation &, const Representation &,
                             const Representation &, const Representation &,
                             const Coordinates &,
                             Eigen::Ref<ChangeOfCoordinatesDiff> result) {
    result = ChangeOfCoordinatesDiff::Identity();
  }

  static constexpr std::size_t dimension = Rows * Cols;

  static constexpr std::size_t tangent_repr_dimension = Rows * Cols;
};

template <long EffectiveRows, long EffectiveCols, long Dim>
class SparseLinearManifoldAtlas {
public:
  static const constexpr bool is_differential_sparse = true;
  using Representation = Eigen::SparseMatrix<double>;

  using Coordinates = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  using ChartDifferential = Eigen::SparseMatrix<double>;
  using ChartDifferentialRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using ChangeOfCoordinatesDiff = Eigen::SparseMatrix<double>;
  using ChangeOfCoordinatesDiffRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using ParametrizationDifferential = Eigen::SparseMatrix<double>;
  using ParametrizationDifferentialRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using Tangent = Eigen::Matrix<double, EffectiveRows * EffectiveCols, 1>;

  static void chart(const Representation &, const Representation &,
                    const Representation &element, Coordinates &result) {
    result = Eigen::Map<const Coordinates>(element.data());
  }

  static void chart_diff(const Representation &, const Representation &,
                         const Representation &, ChartDifferentialRef) {}

  static void param(const Representation &, const Representation &,
                    const Coordinates &, Representation &) {}

  static void param_diff(const Representation &, const Representation &,
                         const Coordinates &, ParametrizationDifferentialRef) {}

  static Representation random_projection() {
    return Representation(EffectiveRows, EffectiveCols);
  }

  static void
  change_of_coordinates(const Representation &, const Representation &,
                        const Representation &, const Representation &,
                        const Coordinates &coordinates, Coordinates &result) {
    result = coordinates;
  }

  static void
  change_of_coordinates_diff(const Representation &, const Representation &,
                             const Representation &, const Representation &,
                             const Coordinates &, ChangeOfCoordinatesDiffRef) {}

  static constexpr std::size_t dimension = Dim;

  static constexpr std::size_t tangent_repr_dimension = Dim;
};
