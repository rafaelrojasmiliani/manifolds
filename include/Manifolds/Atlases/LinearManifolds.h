#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>

template <long Rows, long Cols> class LinearManifoldAtlas {
public:
  static const constexpr bool is_differential_sparse = false;
  using Representation = Eigen::Matrix<double, Rows, Cols>;
  using Coordinates = Eigen::Matrix<double, Rows * Cols, 1>;

  using ChartDifferential = Eigen::Matrix<double, Rows * Cols, Rows * Cols>;
  using ChangeOfCoordinatesDiff =
      Eigen::Matrix<double, Rows * Cols, Rows * Cols>;
  using ParametrizationDifferential =
      Eigen::Matrix<double, Rows * Cols, Rows * Cols>;

  using Tangent = Eigen::Matrix<double, Rows * Cols, 1>;

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

  static void
  change_of_coordinates(const Representation &, const Representation &,
                        const Representation &, const Representation &,
                        const Coordinates &coordinates, Coordinates &result) {
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

template <long Rows, long Cols> class SparseLinearManifoldAtlas {
public:
  static const constexpr bool is_differential_sparse = true;
  using Representation = Eigen::SparseMatrix<double>;

  using Coordinates = Eigen::Matrix<double, Rows * Cols, 1>;

  using ChartDifferential = Eigen::SparseMatrix<double>;
  using ChartDifferentialRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using ChangeOfCoordinatesDiff = Eigen::SparseMatrix<double>;
  using ChangeOfCoordinatesDiffRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using ParametrizationDifferential = Eigen::SparseMatrix<double>;
  using ParametrizationDifferentialRef =
      std::reference_wrapper<Eigen::SparseMatrix<double>>;

  using Tangent = Eigen::Matrix<double, Rows * Cols, 1>;

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
    return Representation(Rows, Cols);
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

  static constexpr std::size_t dimension = Rows * Cols;

  static constexpr std::size_t tangent_repr_dimension = Rows * Cols;
};
