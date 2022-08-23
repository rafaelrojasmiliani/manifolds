
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm> // std::sort, std::stable_sort
#include <cmath>
#include <gsplines/Collocation/GaussLobattoLagrange.hpp>
#include <gtest/gtest.h>
#include <numeric> // std::iota

/* This example shows how to automatically generate a chart in the sphere from
 * two points.
 *
 * The charts are spherical coordinates with inclination and azimuth  with
 * respect to a frame of coordinates generated from two points.
 * Two points are used to generate charts and respective parametrizations.
 * The chart should cover both points.
 *
 *
 *
 *
 * */

class Parametrization;
class S2 {
private:
  friend Parametrization;
  Eigen::Vector3d repr_;

public:
  typedef Eigen::Vector3d Tanget;

  S2(const Eigen::Vector3d &_in) : repr_(_in.normalized()) {}

  const Eigen::Vector3d get_representation() const { return repr_; }

  bool operator==(const S2 &_other) const {
    Eigen::Vector3d err = _other.get_representation() - get_representation();
    return err.norm() < 1.0e-9;
  }
};

class Chart {
private:
  const Eigen::Vector3d x_;
  const Eigen::Vector3d y_;
  const Eigen::Vector3d z_;

public:
  Chart(const S2 &_p1, const S2 &_p2)
      : x_((0.5 * (_p1.get_representation() + _p2.get_representation()))
               .normalized()),
        y_((_p2.get_representation() - _p2.get_representation().dot(x_) * x_)
               .normalized()),
        z_(x_.cross(y_)) {

    std::cout << x_.transpose() << " <- x \n";
    std::cout << y_.transpose() << " <- y \n";
    std::cout << z_.transpose() << " <- z \n";
  }
  Eigen::Vector2d operator()(const S2 &_p) const {
    const Eigen::Vector3d &p = _p.get_representation();
    double x = p.dot(x_);
    double y = p.dot(y_);
    double z = p.dot(z_);

    double inclination = std::acos(z);
    double azimuth = std::atan2(y, x);
    Eigen::Vector2d result;
    result << inclination, azimuth;
    return result;
  }

  bool in_domain(const S2 &_p) const {
    const Eigen::Vector3d &p = _p.get_representation();
    double x = p.dot(x_);
    double z = p.dot(z_);
    if (x <= 1.0e-5)
      return false;
    if (z < -0.9 or 0.9 < z)
      return false;
    return true;
  }
  Eigen::Matrix<double, 2, 3> diff(const S2 &_p) const {
    const Eigen::Vector3d &p = _p.get_representation();
    double x = p.dot(x_);
    double y = p.dot(y_);
    double z = p.dot(z_);
    double dacos = -1.0 / (std::sqrt(1.0 - z * z));
    Eigen::Matrix<double, 2, 3> result;
    result(0, 0) = dacos * z_(0);
    result(0, 1) = dacos * z_(1);
    result(0, 2) = dacos * z_(2);

    double datan_dx = -y / (x * x + y * y);
    double datan_dy = x / (x * x + y * y);

    result(1, 0) = datan_dx * x_(0) + datan_dy * y_(0);
    result(1, 1) = datan_dx * x_(1) + datan_dy * y_(1);
    result(1, 2) = datan_dx * x_(2) + datan_dy * y_(2);

    return result;
  }
};

class Parametrization {

private:
  Eigen::Vector3d x_;
  Eigen::Vector3d y_;
  Eigen::Vector3d z_;

public:
  Parametrization(const S2 &_p1, const S2 &_p2)
      : x_((0.5 * (_p1.get_representation() + _p2.get_representation()))
               .normalized()),
        y_((_p2.get_representation() - _p2.get_representation().dot(x_) * x_)
               .normalized()),
        z_(x_.cross(y_)) {}

  S2 operator()(const Eigen::Vector2d &_x) const {
    const double &inclination = _x(0);
    const double &azimuth = _x(1);
    Eigen::Vector3d point = std::cos(azimuth) * std::sin(inclination) * x_ +
                            std::sin(azimuth) * std::sin(inclination) * y_ +
                            std::cos(inclination) * z_;

    return S2(point);
  }

  Eigen::Matrix<double, 3, 2> diff(const Eigen::Vector2d &_x) const {
    const double &inclination = _x(0);
    const double &azimuth = _x(1);
    Eigen::Matrix<double, 3, 2> result;

    result.leftCols(1) = std::cos(azimuth) * std::cos(inclination) * x_ +
                         std::sin(azimuth) * std::cos(inclination) * y_ -
                         std::sin(inclination) * z_;

    result.rightCols(1) = -std::sin(azimuth) * std::sin(inclination) * x_ +
                          std::cos(azimuth) * std::sin(inclination) * y_;

    return result;
  }
};

Eigen::MatrixXd C1Continuity(gsplines::basis::Basis &_basis,
                             std::vector<Parametrization> _params,
                             Eigen::VectorXd &_time_intervals) {}

class S2Spline : public gsplines::collocation::GLLSpline {
private:
  Eigen::VectorXd coefficients_;
  Eigen::VectorXd domain_intervals_;
  // std::vector<Parametrization> parametrization_;

public:
};

TEST(Manifolds, Sphere) {

  Eigen::Vector3d v1, v2;
  v1 << 1, 0, 0;
  v2 << 0.7, 0.7, 0;
  S2 p1(v1);
  S2 p2(v2);

  Chart chart(p1, p2);
  Parametrization param(p1, p2);

  Eigen::Vector2d v;
  v << M_PI_2, M_PI_2;

  std::cout << "param diffeo \n " << param.diff(v) << "\n\n";
  std::cout << "chart diffeo \n " << chart.diff(param(v)) << "\n\n";

  std::cout << "d_param * d_chart  \n " << param.diff(v) * chart.diff(param(v))
            << "\n\n";

  std::cout << " d_chart * d_param \n " << chart.diff(param(v)) * param.diff(v)
            << "\n\n";

  ASSERT_TRUE(param(chart(p2)) == p2);
  ASSERT_LT((chart(param(v)) - v).norm(), 1.0e-9);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
