#include <Manifolds/Manifold.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

/* Test that we get the correct basis*/
using namespace manifolds;

class SO3 : public Manifold<SO3, 3> {
private:
  Eigen::Isometry3d mat_;

public:
  typedef std::function<Eigen::Matrix<double, 3, 1>(const SO3 &)> chart;

  SO3(const Eigen::Isometry3d &_in) : mat_(_in) {}

  const Eigen::Isometry3d &get_representation() const { return mat_; }
};

Eigen::Matrix<double, 3, 1> euler_rpy_chart(const SO3 &_obj) {
  const Eigen::Isometry3d &_mat = _obj.get_representation();

  Eigen::Matrix<double, 3, 1> result;
  result(0) = std::atan2(_mat(1, 0), _mat(0, 0));
  result(1) = std::atan2(-_mat(2, 0), std::hypot(_mat(2, 1), _mat(2, 2)));
  result(2) = std::atan2(_mat(2, 1), _mat(2, 2));
  return result;
}

SO3 euler_rpy_parametrization(const Eigen::Matrix<double, 3, 1> &_in) {
  Eigen::Isometry3d result =
      Eigen::Translation3d(0, 0, 0) *
      Eigen::Quaterniond(Eigen::AngleAxisd(_in(0), Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(_in(1), Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(_in(2), Eigen::Vector3d::UnitX()));

  return result;
}

Eigen::Matrix<double, 3, 1> euler_zyz_chart(const SO3 &_obj) {
  const Eigen::Isometry3d &_mat = _obj.get_representation();

  Eigen::Matrix<double, 3, 1> result;
  result(0) = std::atan2(_mat(1, 2), _mat(0, 2));
  result(1) = std::atan2(std::hypot(_mat(0, 2), _mat(1, 2)), _mat(2, 2));
  result(2) = std::atan2(_mat(2, 1), -_mat(2, 0));
  return result;
}

SO3 euler_zyz_parametrization(const Eigen::Matrix<double, 3, 1> &_in) {
  Eigen::Isometry3d result =
      Eigen::Translation3d(0, 0, 0) *
      Eigen::Quaterniond(Eigen::AngleAxisd(_in(0), Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(_in(1), Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(_in(2), Eigen::Vector3d::UnitZ()));

  return result;
}

TEST(Basis, Get_Basis) {

  for (int i = 0; i < 100; i++) {

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    Eigen::Vector3d v2 = euler_rpy_chart(euler_rpy_parametrization(v));
    ASSERT_LT((v - v2).norm(), 1.0e-10);

    v(1) += 1.0;
    v2 = euler_zyz_chart(euler_zyz_parametrization(v));
    ASSERT_LT((v - v2).norm(), 1.0e-10) << "ZYZ Euler";
  }
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
