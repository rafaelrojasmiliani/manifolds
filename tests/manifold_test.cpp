#include <Manifolds/LinearManifolds/LinearManifolds.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <unordered_map>

using namespace manifolds;

TEST(Manifolds, FaithfullManifolds) {
  // Test instantiation
  R3 a;
  // Test assigment with representation operator
  Eigen::Vector3d v = {1, 2, 3};
  a = v;
  EXPECT_TRUE(a.crepr().isApprox(v));

  // Test assigment with manifold operator
  R3 d;
  d = a;
  EXPECT_TRUE(a.crepr().isApprox(d.crepr()));

  // Test copy constructor with other manifold
  R3 b = a;
  EXPECT_TRUE(a.crepr().isApprox(b.crepr()));

  // Test copy constructor from represenation
  R3 c = v;
  EXPECT_TRUE(c.crepr().isApprox(v));

  // Test getters
  EXPECT_EQ(a.get_dim(), R3::dimension);
  EXPECT_EQ(a.get_tanget_repr_dim(), R3::tangent_repr_dimension);

  // Test cast to reference
  auto fun = [](Eigen::Vector3d &v) { v(0) = 1.5; };
  fun(a);
  EXPECT_NEAR(a.crepr()(0), 1.5, 1.0e-9);

  // Test cast to const reference
  auto fun2 = [](const Eigen::Vector3d &v) { return v(0); };
  EXPECT_NEAR(fun2(a), 1.5, 1.0e-9);

  // Test cast to const reference again
  Eigen::Vector3d v2 = a;
  EXPECT_TRUE(v2.isApprox(a.crepr()));

  // Test of ManifoldBase
  // Test ManifoldBase constructor
  std::unique_ptr<ManifoldBase> b_ptr = std::make_unique<R3>();
  *static_cast<R3 *>(b_ptr.get()) = v;
  EXPECT_TRUE(static_cast<R3 *>(b_ptr.get())->crepr().isApprox(v));

  // Test ManifoldBase Copy constructor
  std::unique_ptr<ManifoldBase> a_ptr = std::make_unique<R3>(a);
  EXPECT_TRUE(static_cast<R3 *>(a_ptr.get())->crepr().isApprox(a.crepr()));

  // Test ManifoldBase clone
  std::unique_ptr<ManifoldBase> c_ptr = b_ptr->clone();
  EXPECT_TRUE(static_cast<R3 *>(c_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(b_ptr.get())->crepr()));
  // Test methods
  EXPECT_EQ(a_ptr->get_dim(), R3::dimension);
  EXPECT_EQ(a_ptr->get_tanget_repr_dim(), R3::tangent_repr_dimension);
}

/* Test that we get the correct basis*/
/*
using namespace manifolds;

class SO3;

Eigen::Matrix<double, 3, 1> euler_rpy_chart(const SO3 &_obj);

SO3 euler_rpy_parametrization(const Eigen::Matrix<double, 3, 1> &_in);

Eigen::Matrix<double, 3, 1> euler_zyz_chart(const SO3 &_obj);

SO3 euler_zyz_parametrization(const Eigen::Matrix<double, 3, 1> &_in);

class SO3 : public Manifold<SO3, 3> {
private:
  Eigen::Isometry3d mat_;

public:
  typedef std::function<Eigen::Matrix<double, 3, 1>(const SO3 &)> chart;

  SO3(const Eigen::Isometry3d &_in) : mat_(_in) {}

  const Eigen::Isometry3d &get_representation() const { return mat_; }

  typedef std::unordered_map<
      std::string, std::function<Eigen::Matrix<double, 3, 1>(const SO3 &)>>
      ChartDictionary;

  static const ChartDictionary charts_;
};

SO3::ChartDictionary charts_ = {{"euler_rpy", euler_rpy_chart}};

template <typename DomainT, typename CodomainT> class Map {
public:
  virtual CodomainT operator()(const DomainT &) = 0;
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

TEST(Manifolds, Parametrization) {

  topology::CartensianInterval<3> interval({{1, 2}});
  for (int i = 0; i < 100; i++) {

    Eigen::Vector3d v = Eigen::Vector3d::Random();

    Eigen::Vector3d v2 = euler_rpy_chart(euler_rpy_parametrization(v));
    ASSERT_LT((v - v2).norm(), 1.0e-10);

    v(1) += 1.0;
    v2 = euler_zyz_chart(euler_zyz_parametrization(v));
    ASSERT_LT((v - v2).norm(), 1.0e-10) << "ZYZ Euler";
  }
}
*/
int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
