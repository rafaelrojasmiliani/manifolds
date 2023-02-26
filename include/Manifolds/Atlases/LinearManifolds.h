#include <Eigen/Core>
template <long Rows, long Cols> class LinearManifoldAtlas {
public:
  using Representation = Eigen::Matrix<double, Rows, Cols>;
  using Coordinates = Eigen::Matrix<double, Rows * Cols, 1>;

  using ChartDifferential = Eigen::Matrix<double, Rows * Cols, Rows * Cols>;
  using ChangeOfCoordinatesDiff =
      Eigen::Matrix<double, Rows * Cols, Rows * Cols>;
  using ParametrizationDifferential =
      Eigen::Matrix<double, Rows * Cols, Rows * Cols>;

  using Tangent = Eigen::Matrix<double, Rows * Cols, 1>;

  static void chart(const Representation &point1, const Representation &point2,
                    const Representation &element, Coordinates &result) {
    result = Eigen::Map<const Coordinates>(element.data());
  }

  static void chart_diff(const Representation &point1,
                         const Representation &point2,
                         const Representation &element, Coordinates &result) {

    result = ChartDifferential::Identity();
  }

  static void param(const Representation &point1, const Representation &point2,
                    const Coordinates &coordinates, Representation &result) {
    result = Eigen::Map<const Representation>(coordinates.data());
  }

  static void param_diff(const Representation &point1,
                         const Representation &point2,
                         const Eigen::Matrix<double, 2, 1> &coordinates,
                         Coordinates &result) {
    result = ChartDifferential::Identity();
  }

  static Representation random_projection() { return Representation::Random(); }

  static void change_of_coordinates(const Representation &point1A,
                                    const Representation &point2A,
                                    const Representation &point1B,
                                    const Representation &point2B,
                                    const Coordinates &coordinates,
                                    Coordinates &result) {
    result = coordinates;
  }

  static void change_of_coordinates_diff(const Representation &point1A,
                                         const Representation &point2A,
                                         const Representation &point1B,
                                         const Representation &point2B,
                                         const Coordinates &coordinates,
                                         ChangeOfCoordinatesDiff &result) {
    ChangeOfCoordinatesDiff::Identity();
  }

  static const std::size_t dimension = Rows * Cols;

  static const std::size_t tanget_repr_dimension = Rows * Cols;

private:
};
