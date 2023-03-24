
#include <Eigen/Core>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

using namespace manifolds;

TEST(LinearMaps, FaithfullManifolds) {

  // Test Empty
  End3 m0;
  // Test assigment with representation operator
  Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
  m0 = m1;
  EXPECT_TRUE(m0.crepr().isApprox(m1));

  // Test assigment with manifold operator
  End3 m2;
  m2 = m0;
  EXPECT_TRUE(m2.crepr().isApprox(m0.crepr()));
}

TEST(LinearMaps, Map) {

  Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();
  Eigen::Vector3d w1 = m1 * v1;

  End3 m2(m1);
  R3 v2(v1);

  R3 w2 = m2(v2);
  EXPECT_TRUE(w2.crepr().isApprox(w1));
}
TEST(LinearMaps, LinearOperations) {

  Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
  Eigen::Matrix3d m2 = Eigen::Matrix3d::Random();
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();
  Eigen::Vector3d w1 = (m1 + m2) * v1;

  End3 M1(m1);
  End3 M2(m2);
  End3 M3;
  M3 = M1 + M2;
  R3 V1(v1);
  R3 W1;
  W1 = M3(V1);
  EXPECT_TRUE(W1.crepr().isApprox(w1));
}
TEST(LinearMaps, AlgebraOperations) {

  Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
  Eigen::Matrix3d m2 = Eigen::Matrix3d::Random();
  Eigen::Vector3d v1 = Eigen::Vector3d::Random();
  Eigen::Vector3d w1 = (m1 * m2) * v1;

  End3 M1(m1);
  End3 M2(m2);
  End3 M3;
  M3 = M1 * M2;
  R3 V1(v1);
  R3 W1;
  W1 = M3(V1);
  EXPECT_TRUE(W1.crepr().isApprox(w1));
}

TEST(LinearMaps, Composition) {

  Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
  End3 M1(m1);
  MapComposition mymap(M1);
  End3 mymap_lin(M1);
  for (int i = 0; i < 20; i++) {

    Eigen::Vector3d v1 = Eigen::Vector3d::Random();
    Eigen::Matrix3d m2 = Eigen::Matrix3d::Random();
    End3 M2(m2);

    mymap.compose(M2);
    mymap_lin.compose(M2);

    Eigen::Vector3d w1 = mymap(v1);
    Eigen::Vector3d w2 = mymap_lin(v1);
    EXPECT_TRUE(w1.isApprox(w2));

    Eigen::Matrix3d d1 = mymap.diff(v1);
    EXPECT_TRUE(d1.isApprox(mymap_lin.crepr()));
  }
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
