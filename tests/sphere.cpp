
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Manifolds/Sphere.hpp>
#include <algorithm> // std::sort, std::stable_sort
#include <cmath>
#include <gsplines/Collocation/GaussLobattoLagrange.hpp>
#include <gtest/gtest.h>
#include <numeric> // std::iota
#include <random>

// This example shows how to automatically generate a chart in the sphere from
//  two points.
//
//  The charts are spherical coordinates with inclination and azimuth  with
//  respect to a frame of coordinates generated from two points.
//  Two points are used to generate charts and respective parametrizations.
//  The chart should cover both points.
//
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> inclination_dist(-1.5, 1.5);
std::uniform_real_distribution<double> azimuth(-3.0, 3.0);

using namespace manifolds;
TEST(Manifolds, Sphere) {

  Eigen::Vector2d coordinate;
  for (int i = 0; i < 100; i++) {
    S2 p1(Eigen::Vector3d::Random());
    S2 p2(Eigen::Vector3d::Random());

    S2Chart chart(p1, p2);
    S2Param param(p1, p2);

    for (int i = 0; i < 100; i++) {
      coordinate << inclination_dist(mt), azimuth(mt);
      ASSERT_TRUE(param(chart(p2)) == p2);
      ASSERT_LT((chart(param(coordinate)).crepr() - coordinate).norm(), 1.0e-9);
    }
  }
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
