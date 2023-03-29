#include <Eigen/Core>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <memory>

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

TEST(MapComposition, Base) {

  R3 p1(Eigen::Vector3d::Random());

  Identity<R3> id;
  std::unique_ptr<ManifoldBase> p1_ptr = std::make_unique<R3>(p1);

  MapBaseComposition mc(id);

  auto p2_ptr = mc.domain_buffer();

  mc.value(p1_ptr, p2_ptr);
  EXPECT_TRUE(static_cast<R3 *>(p1_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(p2_ptr.get())->crepr()));

  for (int _ = 0; _ < 100; _++) {
    std::unique_ptr<ManifoldBase> pa_ptr =
        std::make_unique<R3>(Eigen::Vector3d::Random());
    mc.append(id);
    auto pb_ptr = mc.codomain_buffer();
    mc.value(pa_ptr, pb_ptr);
    EXPECT_TRUE(static_cast<R3 *>(pa_ptr.get())
                    ->crepr()
                    .isApprox(static_cast<R3 *>(pb_ptr.get())->crepr()));

    Eigen::MatrixXd d = mc.linearization_buffer();

    mc.diff(pa_ptr, d);
    EXPECT_TRUE(d.isApprox(Eigen::Matrix<double, 3, 3>::Identity()));
  }

  std::unique_ptr<MapBase> mb = mc.clone();
  auto p3_ptr = mc.domain_buffer();
  mb->value(p1_ptr, p3_ptr);
  EXPECT_TRUE(static_cast<R3 *>(p1_ptr.get())
                  ->crepr()
                  .isApprox(static_cast<R3 *>(p3_ptr.get())->crepr()));

  // MapComposition mc = id.compose(id);

  // p2 = mc(p1);
}

TEST(MapComposition, Composition) {

  R3 p1(Eigen::Vector3d::Random());

  Identity<R3> id;
  id.compose(id);

  R3 p2 = id(p1);

  EXPECT_TRUE(p1.crepr().isApprox(p2.crepr()));
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
