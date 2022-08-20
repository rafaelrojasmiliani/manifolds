
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <gtest/gtest.h>

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
  Eigen::Vector3d x_;
  Eigen::Vector3d y_;
  Eigen::Vector3d z_;

public:
  Chart(const S2 &_p1, const S2 &_p2)
      : x_(_p1.get_representation().normalized()),
        y_((_p2.get_representation() - _p2.get_representation().dot(x_) * x_)
               .normalized()),
        z_(x_.cross(y_)) {}
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

    double datan = x * x / (std::sqrt(x * x + y * y));
    result(1, 0) = datan * z_(0);
    result(1, 1) = datan * z_(1);
    result(1, 2) = 0.0;

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
      : x_(_p1.get_representation()),
        y_(_p2.get_representation() - _p2.get_representation().dot(x_) * x_),
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
    /*
    result(0, 0) = dacos * z_(0);
    result(0, 1) = dacos * z_(1);
    result(0, 2) = dacos * z_(2);

    double datan = x * x / (std::sqrt(x * x + y * y));
    result(1, 0) = datan * z_(0);
    result(1, 1) = datan * z_(1);
    result(1, 2) = 0.0;
    */

    return result;
  }
};

TEST(Manifolds, Sphere) {

  Eigen::Vector3d v1, v2;
  v1 << 1, 0, 0;
  v2 << 0.7, 0.7, 0;
  S2 p1(v1);
  S2 p2(v2);

  Chart chart(p1, p2);
  Parametrization param(p1, p2);

  ASSERT_TRUE(param(chart(p2)) == p2);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
