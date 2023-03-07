#include <Eigen/Core>
#include <Manifolds/Maps/Charts.hpp>
#include <algorithm>
#include <gtest/gtest.h>

using namespace manifolds;

TEST(Map, Identity) {

  R3 p1(Eigen::Vector3d::Random());
  R3 p2(Eigen::Vector3d::Random());

  R3 p3(Eigen::Vector3d::Random());
  R3 p4(Eigen::Vector3d::Random());

  R3::Chart chart1(p1, p2);
  R3::Parametrization param1(p1, p2);

  R3::Chart chart2(p3, p4);
  R3::Parametrization param2(p3, p4);

  auto id = param1.compose(chart1);

  auto cocA = param1.compose(chart2);
  auto cocB = param2.compose(chart1);

  auto id2 = cocA.compose(cocB);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
