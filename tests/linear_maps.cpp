
#include "Test.h"
#include <Eigen/Core>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <gtest/gtest.h>

using namespace manifolds;
/// Test properties of LinearMaps
TEST(LinearMaps, DenseLinearMaps) {

  using T = DenseLinearMap<DenseLinearManifold<20>, DenseLinearManifold<10>>;
  TestManifoldFaithful<T>();
  TestMap<T>();
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
