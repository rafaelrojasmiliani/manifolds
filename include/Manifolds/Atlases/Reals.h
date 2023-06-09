#include <Eigen/Core>
#pragma once

class RealsAtlas {
public:
  static const constexpr bool is_differential_sparse = false;
  using Representation = double;
  using RepresentationRef = double &;
  using RepresentationConstRef = const double &;
  using Coordinates = double;
  using ChartDifferential = Eigen::Matrix<double, 1, 1>;

  using ChangeOfCoordinatesDiff = Eigen::Matrix<double, 1, 1>;

  using ParametrizationDifferential = Eigen::Matrix<double, 1, 1>;

  using Tangent = Eigen::Matrix<double, 1, 1>;

  static const Representation &cref_to_type(RepresentationConstRef _ref) {
    return _ref;
  }
  static Representation &ref_to_type(RepresentationRef &_ref) { return _ref; }
  static RepresentationConstRef ctype_to_ref(const Representation &_ref) {
    return _ref;
  }
  static RepresentationRef type_to_ref(Representation &_ref) { return _ref; }

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

  static bool comparison(const Representation &_lhs,
                         const Representation &_rhs) {
    double _tol = 1.0e-12;
    double err = std::abs(_lhs - _rhs);
    double lhs_max = std::fabs(_lhs);
    double rhs_max = std::fabs(_rhs);
    if (lhs_max < _tol or rhs_max < _tol) {
      return err < _tol;
    }

    return err / lhs_max < _tol and err / rhs_max < _tol;
  }
};
