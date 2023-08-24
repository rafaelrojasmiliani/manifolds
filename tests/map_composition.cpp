

#include <Eigen/Core>
#include <Manifolds/LinearManifolds/LinearMaps.hpp>
#include <gtest/gtest.h>

using namespace manifolds;
/// Test properties of LinearMaps
TEST(MapComposition, DenseLinearMaps) {

  using T1 = DenseLinearManifold<2>;
  using T2 = DenseLinearManifold<4>;
  using T3 = DenseLinearManifold<6>;
  using T4 = DenseLinearManifold<7>;

  using M1 = DenseLinearMap<T1, T2>;
  using M2 = DenseLinearMap<T2, T3>;
  using M3 = DenseLinearMap<T3, T4>;
  using M4 = DenseLinearMap<T4, T4>;

  M1 m1 = *M1::random();

  auto c1 = MapComposition(m1);

  for (int _ = 0; _ < 10; _++) {
    auto v1 = T1::random();
    Eigen::MatrixXd v2_ground_truth = m1.crepr() * v1.crepr();
    auto v2_test = c1(v1);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test.crepr()))
        << "Error on value \n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    auto mat = c1.linearization_buffer();

    auto v2_test_2 = T2::random();
    c1.diff(v1, v2_test_2, mat);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test_2.crepr()))
        << "Error on derivative \n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    EXPECT_TRUE(mat.isApprox(m1.crepr()))
        << "Error on derivative \n ++++++++++++++++++++++++++++++\n"
        << mat << "\n-----------------------------------------\n"
        << m1.crepr() << "\n ++++++++++++++++++++++++++++++++++ \n";
  }

  M2 m2 = *M2::random();

  auto c2 = m1 | m2;

  for (int _ = 0; _ < 10; _++) {
    auto v1 = T1::random();
    Eigen::MatrixXd v2_ground_truth = m2.crepr() * m1.crepr() * v1.crepr();
    auto v2_test = c2(v1);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test.crepr()))
        << "\n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    auto mat = c2.linearization_buffer();
    auto v2_test_2 = T3::random();
    c2.diff(v1, v2_test_2, mat);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test_2.crepr()))
        << "Error on derivative \n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    EXPECT_TRUE(mat.isApprox(m2.crepr() * m1.crepr()))
        << "Error on derivative \n ++++++++++++++++++++++++++++++\n"
        << mat << "\n-----------------------------------------\n"
        << m1.crepr() << "\n ++++++++++++++++++++++++++++++++++ \n";
  }

  M3 m3 = *M3::random();

  auto c3 = m1 | m2 | m3;

  for (int _ = 0; _ < 10; _++) {
    auto v1 = T1::random();
    Eigen::MatrixXd v2_ground_truth =
        m3.crepr() * m2.crepr() * m1.crepr() * v1.crepr();
    auto v2_test = c3(v1);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test.crepr()))
        << "\n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    auto mat = c3.linearization_buffer();
    auto v2_test_2 = T4::random();
    c3.diff(v1, v2_test_2, mat);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test_2.crepr()))
        << "Error on derivative \n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";
    EXPECT_TRUE(mat.isApprox(m3.crepr() * m2.crepr() * m1.crepr()))
        << "Error on derivative \n ++++++++++++++++++++++++++++++\n"
        << mat << "\n-----------------------------------------\n"
        << m1.crepr() << "\n ++++++++++++++++++++++++++++++++++ \n";
  }

  M4 m4 = *M4::random();

  auto c4 = m1 | m2 | m3 | m4 | m4 | m4;

  for (int _ = 0; _ < 10; _++) {
    auto v1 = T1::random();
    Eigen::MatrixXd v2_ground_truth = m4.crepr() * m4.crepr() * m4.crepr() *
                                      m3.crepr() * m2.crepr() * m1.crepr() *
                                      v1.crepr();
    auto v2_test = c3(v1);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test.crepr()))
        << "\n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";

    auto mat = c4.linearization_buffer();
    auto v2_test_2 = T4::random();
    c3.diff(v1, v2_test_2, mat);

    EXPECT_TRUE(v2_ground_truth.isApprox(v2_test_2.crepr()))
        << "Error on derivative \n"
        << v2_ground_truth.transpose() << "\n"
        << v2_test.crepr().transpose() << "\n";
    EXPECT_TRUE(mat.isApprox(m4.crepr() * m4.crepr() * m4.crepr() * m3.crepr() *
                             m2.crepr() * m1.crepr()))
        << "Error on derivative \n ++++++++++++++++++++++++++++++\n"
        << mat << "\n-----------------------------------------\n"
        << m1.crepr() << "\n ++++++++++++++++++++++++++++++++++ \n";
  }
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
