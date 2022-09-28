#include <Manifolds/Sphere.hpp>
namespace manifolds {

S2::S2(const Eigen::Vector3d &_vec)
    : ManifoldInheritanceHelper(_vec.normalized()) {}

S2::S2() : ManifoldInheritanceHelper(Eigen::Vector3d::Random().normalized()) {}

S2Chart::S2Chart(const S2 &_p1, const S2 &_p2)
    : x_((0.5 * (_p1.crepr() + _p2.crepr())).normalized()),
      y_((_p2.crepr() - _p2.crepr().dot(x_) * x_).normalized()),
      z_(x_.cross(y_)) {}

bool S2::operator==(const S2 &_other) const {
  Eigen::Vector3d err = _other.crepr() - crepr();
  return err.norm() < 1.0e-9;
}

bool S2Chart::value_on_repr(const Eigen::Vector3d &_p,
                            Eigen::Vector2d &_result) const {

  const Eigen::Vector3d &p = _p;
  double x = p.dot(x_);
  double y = p.dot(y_);
  double z = p.dot(z_);

  double inclination = std::acos(z);
  double azimuth = std::atan2(y, x);
  _result << inclination, azimuth;
  return true;
} // namespace manifolds

bool S2Chart::diff_from_repr(const Eigen::Vector3d &p,
                             Eigen::MatrixXd &result) const {
  double x = p.dot(x_);
  double y = p.dot(y_);
  double z = p.dot(z_);
  double dacos = -1.0 / (std::sqrt(1.0 - z * z));

  result(0, 0) = dacos * z_(0);
  result(0, 1) = dacos * z_(1);
  result(0, 2) = dacos * z_(2);

  double datan_dx = -y / (x * x + y * y);
  double datan_dy = x / (x * x + y * y);

  result(1, 0) = datan_dx * x_(0) + datan_dy * y_(0);
  result(1, 1) = datan_dx * x_(1) + datan_dy * y_(1);
  result(1, 2) = datan_dx * x_(2) + datan_dy * y_(2);
  return true;
}

S2Param::S2Param(const S2 &_p1, const S2 &_p2)
    : x_((0.5 * (_p1.crepr() + _p2.crepr())).normalized()),
      y_((_p2.crepr() - _p2.crepr().dot(x_) * x_).normalized()),
      z_(x_.cross(y_)) {}

bool S2Param::value_on_repr(const Eigen::Vector2d &_x,
                            Eigen::Vector3d &_p) const {
  const double &inclination = _x(0);
  const double &azimuth = _x(1);
  _p = std::cos(azimuth) * std::sin(inclination) * x_ +
       std::sin(azimuth) * std::sin(inclination) * y_ +
       std::cos(inclination) * z_;

  return true;
}

bool S2Param::diff_from_repr(const Eigen::Vector2d &_x,
                             Eigen::MatrixXd &result) const {
  const double &inclination = _x(0);
  const double &azimuth = _x(1);

  result.leftCols(1) = std::cos(azimuth) * std::cos(inclination) * x_ +
                       std::sin(azimuth) * std::cos(inclination) * y_ -
                       std::sin(inclination) * z_;

  result.rightCols(1) = -std::sin(azimuth) * std::sin(inclination) * x_ +
                        std::cos(azimuth) * std::sin(inclination) * y_;

  return true;
}

bool AntipodalMap::value_on_repr(const Eigen::Vector3d &_x,
                                 Eigen::Vector3d &_p) const {
  _p.noalias() = -_x;
  return true;
}
bool AntipodalMap::diff_from_repr(const Eigen::Vector3d &,
                                  Eigen::MatrixXd &_mat) const {
  _mat = -Eigen::MatrixXd::Identity(3, 3);
  return true;
}

} // namespace manifolds
