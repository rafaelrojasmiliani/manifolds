
#include "Test.h"
#include <Eigen/Core>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace manifolds;

/// Test properties of LinearMaps
TEST(MixedLinearMaps, AsDenseLinearManifold) {

  using T = MixedLinearMap<MixedLinearManifold<20>, MixedLinearManifold<10>>;
  TestManifoldFaithfull<T>();
  TestMap<T>();
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
