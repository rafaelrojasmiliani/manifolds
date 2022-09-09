
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Manifolds/Sphere.hpp>
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
using namespace manifolds;
TEST(Manifolds, Sphere) {

  Eigen::Vector3d v1, v2;
  v1 << 1, 0, 0;
  v2 << 0.7, 0.7, 0;
  S2 p1(v1);
  S2 p2(v2);

  S2Chart chart(p1, p2);
  S2Param param(p1, p2);

  Eigen::Vector2d v;
  v << M_PI_2, M_PI_2;

  ASSERT_TRUE(param(chart(p2)) == p2);
  ASSERT_LT((chart(param(v)) - v).norm(), 1.0e-9);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
