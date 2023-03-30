#include <Eigen/Core>
#pragma once

class RealsAtlas {
public:
  static const constexpr bool is_differential_sparse = false;
  using Representation = double;
  using Coordinates = double;
  using ChartDifferential = Eigen::Matrix<double, 1, 1>;

  using ChangeOfCoordinatesDiff = Eigen::Matrix<double, 1, 1>;

  using ParametrizationDifferential = Eigen::Matrix<double, 1, 1>;

  using Tangent = Eigen::Matrix<double, 1, 1>;

  static void chart(const Representation &, const Representation &,
                    const Representation &element, Coordinates &result) {
    result = element;
  }

  static void chart_diff(const Representation &, const Representation &,
                         const Representation &,
                         Eigen::Ref<ChartDifferential> result) {

    result = ChartDifferential::Identity();
  }

  static void param(const Representation &, const Representation &,
                    const Coordinates &coordinates, Representation &result) {
    result = coordinates;
  }

  static void param_diff(const Representation &, const Representation &,
                         const Coordinates &,
                         Eigen::Ref<ChartDifferential> result) {
    result = ChartDifferential::Identity();
  }

  static Representation random_projection() {
    return ChartDifferential::Random()(0, 0);
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
                             const Coordinates &,
                             Eigen::Ref<ChangeOfCoordinatesDiff> result) {
    result = ChangeOfCoordinatesDiff::Identity();
  }

  static constexpr std::size_t dimension = 1;

  static constexpr std::size_t tangent_repr_dimension = 1;
};
