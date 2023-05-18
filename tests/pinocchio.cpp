#include <Manifolds/PinocchioFK.hpp>

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <unordered_map>

using namespace manifolds;

TEST(Manifolds, FaithfullManifolds) {
  std::cout << std::filesystem::current_path().string() << '\n';
  auto fk = ForwardKinematics<7>::from_urdf("tests/urdf/panda_arm.urdf",
                                            "panda_link8");
}
int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
