#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <random>
#include <variant>

#define MAX_ELEMENTS_FOR_DENSE 20000

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
      (Rows * Cols < MAX_ELEMENTS_FOR_DENSE) ? Cols : Eigen::Dynamic;
  static constexpr long EffectiveRows =
      (Rows * Cols < MAX_ELEMENTS_FOR_DENSE) ? Rows : Eigen::Dynamic;

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

  static Representation random_projection() {
    if (dimension < MAX_ELEMENTS_FOR_DENSE)
      return DenseMatrix::Random();

    else
      return get_sparse_random();
  }

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

  static bool comparison(const Representation &_lhs,
                         const Representation &_rhs) {

    double _tol = 1.0e-12;
    double err = 0;
    double lhs_max = 0;
    double rhs_max = 0;
    std::visit(
        [&err, &lhs_max, &rhs_max](const auto &lhs, const auto rhs) {
          if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                       DenseMatrix> or
                        std::is_same_v<decltype(rhs), DenseMatrix>) {
            err = DenseMatrix(lhs - rhs).array().abs().maxCoeff();
          } else {
            err = SparseMatrix(lhs - rhs).coeffs().abs().maxCoeff();
          }

          if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                       DenseMatrix>) {
            lhs_max = (lhs).array().abs().maxCoeff();
          } else {
            lhs_max = (lhs).coeffs().abs().maxCoeff();
          }
          if constexpr (std::is_same_v<std::decay_t<decltype(rhs)>,
                                       DenseMatrix>) {
            rhs_max = (rhs).array().abs().maxCoeff();
          } else {
            rhs_max = (rhs).coeffs().abs().maxCoeff();
          }
        },
        _lhs, _rhs);

    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }

  static SparseMatrix get_sparse_random() {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<long> dist_rows(0, Rows - 1);
    std::uniform_int_distribution<long> dist_cols(0, Cols - 1);

    int rows = Rows;
    int cols = Cols;

    std::size_t nzn = rows * cols * 0.05;

    std::vector<Eigen::Triplet<double>> tripletList;
    for (std::size_t i = 0; i < nzn; i++)
      tripletList.push_back(
          Eigen::Triplet<double>(dist_rows(gen), dist_cols(gen), dist(gen)));

    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
  }
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

  static bool comparison(const Representation &_lhs,
                         const Representation &_rhs) {
    double _tol = 1.0e-12;
    double err = 0;
    double lhs_max = 0;
    double rhs_max = 0;
    err = (_lhs - _rhs).array().abs().maxCoeff();
    lhs_max = _lhs.array().abs().maxCoeff();
    rhs_max = _rhs.array().abs().maxCoeff();

    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }
};

template <long Rows, long Cols> class SparseLinearManifoldAtlas {
  static constexpr long EffectiveCols =
      (Rows * Cols < MAX_ELEMENTS_FOR_DENSE) ? Cols : Eigen::Dynamic;
  static constexpr long EffectiveRows =
      (Rows * Cols < MAX_ELEMENTS_FOR_DENSE) ? Rows : Eigen::Dynamic;

private:
  using SparseMatrix = Eigen::SparseMatrix<double>;

public:
  static const constexpr bool is_differential_sparse = false;
  using Representation = SparseMatrix;
  using RepresentationRef = SparseMatrix &;

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

  static Representation random_projection() {
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<long> dist_rows(0, Rows - 1);
    std::uniform_int_distribution<long> dist_cols(0, Cols - 1);

    int rows = Rows;
    int cols = Cols;

    std::size_t nzn = rows * cols * 0.01;

    std::vector<Eigen::Triplet<double>> tripletList;
    for (std::size_t i = 0; i < nzn; i++)
      tripletList.emplace_back(dist_rows(gen), dist_cols(gen), dist(gen));

    SparseMatrix mat(rows, cols);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
  }

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

  static bool comparison(const Representation &_lhs,
                         const Representation &_rhs) {
    double _tol = 1.0e-12;
    double err = 0;
    double lhs_max = 0;
    double rhs_max = 0;
    err = Representation(_lhs - _rhs).coeffs().abs().maxCoeff();
    lhs_max = _lhs.coeffs().abs().maxCoeff();
    rhs_max = _rhs.coeffs().abs().maxCoeff();

    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }
};
