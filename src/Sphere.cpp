#include <Manifolds/Sphere.hpp>
namespace manifolds {

S2::S2(const Eigen::Vector3d &_vec) : Manifold(_vec.normalized()) {}

S2Chart::S2Chart(const S2 &_p1, const S2 &_p2)
    : x_((0.5 * (_p1.repr() + _p2.repr())).normalized()),
      y_((_p2.repr() - _p2.repr().dot(x_) * x_).normalized()),
      z_(x_.cross(y_)) {}

bool S2::operator==(const S2 &_other) const {
  Eigen::Vector3d err = _other.repr() - repr();
  return err.norm() < 1.0e-9;
}

Eigen::Matrix<double, 2, 1> S2Chart::operator()(const S2 &_p) const {

  const Eigen::Vector3d &p = _p.repr();
  double x = p.dot(x_);
  double y = p.dot(y_);
  double z = p.dot(z_);

  double inclination = std::acos(z);
  double azimuth = std::atan2(y, x);
  Eigen::Vector2d result;
  result << inclination, azimuth;
  return result;
}

Eigen::Matrix<double, 2, 3> S2Chart::diff(const S2 &_p) const {
  const Eigen::Vector3d &p = _p.repr();
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

S2Param::S2Param(const S2 &_p1, const S2 &_p2)
    : x_((0.5 * (_p1.repr() + _p2.repr())).normalized()),
      y_((_p2.repr() - _p2.repr().dot(x_) * x_).normalized()),
      z_(x_.cross(y_)) {}

S2 S2Param::operator()(const Eigen::Vector2d &_x) const {
  const double &inclination = _x(0);
  const double &azimuth = _x(1);
  Eigen::Vector3d point = std::cos(azimuth) * std::sin(inclination) * x_ +
                          std::sin(azimuth) * std::sin(inclination) * y_ +
                          std::cos(inclination) * z_;

  return S2(point);
}

Eigen::Matrix<double, 3, 2> S2Param::diff(const Eigen::Vector2d &_x) const {
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

} // namespace manifolds
