#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>

#include <gtest/gtest.h>

#include "Test.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

using namespace manifolds;

TEST(DenseMatrixManifolds, Test) {
  using T = DenseMatrixManifold<3, 1>;

  TestManifoldFaithful<T>();
}

TEST(SparseMatrixManifold, Test) {
  using T = SparseMatrixManifold<50, 100>;
  TestManifoldFaithful<T>();
}

TEST(Reals, Test) {
  using T = Reals;
  TestManifoldFaithful<T>();
}
/*
TEST(DenseMatrixManifold, Lifting) {

  auto value = [](const R3::Representation &_in, double &_out) -> bool {
    _out = _in.norm();
    return true;
  };
  auto diff = [](const R3::Representation &_in,
                 Eigen::Ref<Eigen::MatrixXd> _mat) -> bool {
    _mat = 1.0 / _in.norm() * R3::Representation::Ones().transpose();
    return true;
  };
  auto norm = MapLifting<R3, Reals, MatrixTypeId::Dense>(value, diff);
}
*/
/*
TEST(Lifting, Test) {
  using T1 = R3;
  using T2 = Reals;

  auto fun = [](const typename T1::Representation &in,
                typename T2::Representation &out) {
    out = in.norm();
    return true;
  };
  auto fun_diff = [fun](const typename T1::Representation &in,
                        typename T2::Representation ) {
    double val;
    fun(in, val);
    return true;
  };

  // DenseLinearToDenseLinearLifting<T1, T2, MatrixTypeId::Dense>([](const
  // T1::R));
}
*/
int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
