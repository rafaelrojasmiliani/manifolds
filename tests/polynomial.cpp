
#include <gtest/gtest.h>

#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/GLPolynomialOperators.hpp>

using namespace manifolds;

TEST(GLPolynomial, LagrangePolynomial) {

  IntervalPartition<1> interval(-1, 1);
  using Pol = PWGLVPolynomial<12, 1, 1>;

  Pol p(interval);
  (void)p;
  p = Pol::zero(interval);

  for (int k = 0; k < 100; k++) {
    double t = p.get_domain().get_random();
    double val = 0.0;
    for (int i = 0; i < 12; i++) {
      p[i](0) = 1.0;
      val += p(t).crepr()(0);
      p[i](0) = 0.0;
    }
    EXPECT_LT(std::fabs(val - 1.0), 1.0e-12);
  }
}

TEST(GLPolynomial, GetCodomainPoint) {

  constexpr std::size_t intervals = 7;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 7;

  IntervalPartition<intervals> interval(-1, 1);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  Pol p = Pol::from_repr(interval, Pol::Representation::Random());

  Eigen::Matrix<double, intervals * nglp, dim> vec2 =
      p.crepr().reshaped<Eigen::RowMajor>(intervals * nglp, dim);

  for (std::size_t i = 0; i < intervals * nglp; i++) {
    EXPECT_TRUE(vec2.row(i).transpose().isApprox(p[i]));
  }
}

TEST(GLPolynomial, Evaluation) {

  constexpr std::size_t intervals = 7;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;

  IntervalPartition<intervals> interval(-1, 1);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto fun = [](double t) {
    Eigen::Vector<double, dim> result;
    result(0) = 1 + t;
    result(1) = 1 - 0.06 * t * t - 0.1 * t;
    result(2) = -3.0 + 0.06 * t * t - 0.1 * t;
    return result;
  };

  Pol p = Pol::from_function(interval, fun);

  Eigen::Matrix<double, intervals * nglp, dim> test_matrix = p.points();

  Eigen::Vector<double, dim> ground;
  Eigen::Vector<double, dim> test;
  for (std::size_t i = 0; i < intervals * nglp; i++) {
    ground = fun(p.domain_point(i));
    test = test_matrix.row(i).transpose();
    EXPECT_TRUE(ground.isApprox(test));
  }
  for (std::size_t i = 0; i < intervals * nglp; i++) {

    double t = p.get_domain().get_random();
    ground = fun(t);
    test = p(t);

    for (std::size_t k = 0; k < dim; k++) {
      EXPECT_NEAR(ground(k), test(k), 1.0e-9) << "Error in component " << k;
    }
  }
}

TEST(GLPolynomial, DerivativeEvaluation) {

  using Pol = PWGLVPolynomial<12, 20, 9>;

  Pol p(0.0, 100.0);
  (void)p;
  Pol p2(0.0, 100.0);
  Pol::codomain_t val = Pol::codomain_t::random();
  p = Pol::constant({0.0, 100.0}, val);

  for (int k = 0; k < 100; k++) {
    double t = p.get_domain().get_random();

    Pol::codomain_t x = p(t);

    double d = p.diff(t)(0);
    EXPECT_LT(std::fabs(d - 0.0), 1.0e-12);
  }

  using ScalarPol = PWGLVPolynomial<12, 20, 1>;
  ScalarPol q = ScalarPol::identity({0.0, 100.0});
  for (int k = 0; k < 100; k++) {

    double t = q.get_domain().get_random();

    ScalarPol::codomain_t x = q(t);
    EXPECT_LT(std::fabs(x.crepr()(0) - t), 1.0e-10);

    double d = q.diff(t)(0);
    EXPECT_LT(std::fabs(d - 1.0), 1.0e-10);
  }
}

TEST(GLPolynomial, DerivativeOperator) {
  using Pol = PWGLVPolynomial<10, 4, 7>;

  IntervalPartition<4> interval(0.0, 100.0);
  Pol pol = Pol::constant(interval, Eigen::Vector<double, 7>::Random());

  EXPECT_EQ(pol.get_dim(), 10 * 4 * 7);
  Pol::Derivative<1> deriv(interval);

  Pol deriv_pol(interval);
  deriv_pol = deriv(pol);

  EXPECT_TRUE(deriv_pol == Pol::zero(interval));
  Pol random_pol(interval);
  random_pol = Pol::random(interval);
  Pol random_pol_derivative(interval);
  random_pol_derivative = deriv(random_pol);
  for (int k = 0; k < 100; k++) {

    double t = random_pol.get_domain().get_random();

    Eigen::VectorXd x = random_pol.diff(t);
    Eigen::VectorXd y = random_pol_derivative(t).crepr();

    EXPECT_TRUE(y.isApprox(x, 1.0e-8)) << "\n"
                                       << y.transpose() << "\n"
                                       << x.transpose() << "\n";
  }
}

/// Test continuity matrix. Let cont be the continuity matrix, p be the
/// polynomial vector and P(p) be the polynomial of the vector p. Let t_i^+ be
/// the limit to t_i from the right and t_i^- be the limit to t_i from the left.
/// Here we test
///
///   \forall q\in R^n and p=cont*q
///
///   P(p)(t_i^-) = P(p)(t_i^+)
///   P'(p)(t_i^-) = P'(p)(t_i^+)
///   P''(p)(t_i^-) = P''(p)(t_i^+)
TEST(GLPolynomial, ContinuityMatrix) {
  constexpr std::size_t intervals = 5;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 5;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  IntervalPartition<intervals> interval(0.0, 100.0);
  auto mat = Pol::continuity_matrix(2);

  Pol pol(interval);
  Pol pold(interval);
  Pol poldd(interval);
  pol = Pol::random(interval);
  pold = Pol::Derivative<1>(interval)(pol);
  poldd = Pol::Derivative<2>(interval)(pol);

  auto vec = mat * pol.crepr();

  for (std::size_t k = 0; k < intervals - 1; k++)
    EXPECT_TRUE(vec.segment(k * dim, dim)
                    .isApprox(pol[(k + 1) * nglp - 1] - pol[(k + 1) * nglp]));

  for (std::size_t k = 0; k < intervals - 1; k++)
    EXPECT_TRUE((vec.segment((intervals - 1) * dim + k * dim, dim) * 2.0 /
                 interval.subinterval_length(0))
                    .isApprox(pold[(k + 1) * nglp - 1] - pold[(k + 1) * nglp]));

  for (std::size_t k = 0; k < intervals - 1; k++) {
    Eigen::Vector<double, dim> dx =
        vec.segment(2 * (intervals - 1) * dim + k * dim, dim) *
        std::pow(2.0 / interval.subinterval_length(0), 2.0);
    EXPECT_TRUE(dx.isApprox(poldd[(k + 1) * nglp - 1] - poldd[(k + 1) * nglp]));
  }
}

/// Test immersion matrix. Were we test that the canonical immersion matrix is
/// the nullspace of the continuity matrix. In fact, any C^k continuous
/// polynomial multiplied by the continuity matrix fives zero
TEST(GLPolynomial, ImmersionMatrix) {
  constexpr std::size_t intervals = 3;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 2;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto mat = Pol::continuity_matrix(2);
  auto basis = Pol::Continuous<2>::canonical_inclusion();

  Eigen::SparseMatrix<double, Eigen::RowMajor> zero = mat * basis;

  EXPECT_LT(zero.coeffs().abs().maxCoeff(), 1.0e-9);
}

/// Test immersion function. We test that after applying the immersion function
/// to a random continuous polynomial, the result is a continuous polynomial
TEST(GLPolynomial, Inclusion) {
  constexpr std::size_t intervals = 4;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  IntervalPartition<intervals> interval(0.0, 100.0);

  auto pol = Pol(interval);
  auto pold = Pol(interval);
  auto poldd = Pol(interval);
  auto pc = Pol::Continuous<2>(interval);

  pc = Pol::Continuous<2>::random(interval);

  auto im = Pol::Continuous<2>::Inclusion(interval);

  pol = im(pc);
  pold = Pol::Derivative<1>(interval)(pol);
  poldd = Pol::Derivative<2>(interval)(pol);

  for (std::size_t k = 0; k < intervals - 1; k++) {
    EXPECT_TRUE(pol[(k + 1) * nglp - 1].isApprox(pol[(k + 1) * nglp]));
    EXPECT_TRUE(pold[(k + 1) * nglp - 1].isApprox(pold[(k + 1) * nglp]));
    EXPECT_TRUE(poldd[(k + 1) * nglp - 1].isApprox(poldd[(k + 1) * nglp]));
  }
}
/// Test that the projection of pwglpolynomial onto the
/// C^k continuous pwglpolynomial is a cpwglpolynomial
TEST(GLPolynomial, EuclideanProjectionMatrix) {
  constexpr std::size_t intervals = 4;
  constexpr std::size_t nglp = 6;
  constexpr std::size_t dim = 3;
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  IntervalPartition<intervals> interval(0.0, 100.0);
  Pol pol(interval);
  Pol pold(interval);
  Pol poldd(interval);
  pol = Pol::random(interval);

  auto projection = Pol::Continuous<2>::euclidean_projector();
  auto immersion = Pol::Continuous<2>::canonical_inclusion();
  Eigen::MatrixXd P = immersion * projection;
  Eigen::MatrixXd zero = P - P * P;
  EXPECT_LT(zero.array().abs().maxCoeff(), 1.0e-9);
}

TEST(GLPolynomial, Lifting) {
  constexpr std::size_t intervals = 10;
  constexpr std::size_t nglp = 8;
  constexpr std::size_t dim = 3;
  IntervalPartition<intervals> interval(0.0, 100.0);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto norm = Pol::Composition<DenseLinearManifold<1>>(
      [](const Eigen::Vector<double, 3> &in, Eigen::Vector<double, 1> &out) {
        out(0) = in.norm();
        return true;
      },
      [](const Eigen::Vector<double, 3> &in,
         Eigen::Ref<Eigen::MatrixXd> _diff) {
        _diff = Eigen::Vector<double, 3>::Ones() / in.norm();
        return true;
      });
  auto pol = Pol::random(interval);

  auto norm_pol = norm(pol);

  Eigen::Matrix<double, intervals * nglp, 1> norm_pol_test =
      pol.crepr()
          .reshaped<Eigen::RowMajor>(intervals * nglp, dim)
          .rowwise()
          .norm();
  EXPECT_TRUE(norm_pol_test.isApprox(norm_pol.crepr()));
  for (int k = 0; k < 10; k++) {

    double t = pol.get_domain().get_random();

    Eigen::Vector<double, 3> x = pol(t);
    Eigen::Vector<double, 1> y = norm_pol(t);

    EXPECT_NEAR(x.norm(), y(0), 0.6); // "good" approximation
  }
}
TEST(GLPolynomial, Integral) {
  constexpr std::size_t intervals = 10;
  constexpr std::size_t nglp = 8;
  constexpr std::size_t dim = 1;
  IntervalPartition<intervals> interval(0.0, 2.0);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  auto pol = Pol::identity(interval);
  Pol::Integral integral(interval);

  for (int i = 1; i < 3; i++) {
    auto fun = [i](double t) {
      return Eigen::Vector<double, 1>{std::pow(t, i)};
    };

    auto pol = Pol::from_function(interval, fun);

    double ground = std::pow(2.0, i + 1) / (i + 1.0);
    double test = integral(pol);

    EXPECT_NEAR(ground, test, 1.0e-8) << "with i = " << i;
  }

  auto deriv = Pol::Derivative<1>(interval);

  for (int i = 0; i < 10; i++) {
    auto p = Pol::Continuous<2>::Inclusion(interval)(
        Pol::Continuous<2>::random(interval));
    double test = integral(deriv(p));
    double ground = p(2.0).crepr()(0) - p(0.0).crepr()(0);
    EXPECT_NEAR(test, ground, 1.0e-8);
  }
}

TEST(GLPolynomial, Composition) {
  constexpr std::size_t intervals = 10;
  constexpr std::size_t nglp = 8;
  constexpr std::size_t dim = 7;
  IntervalPartition<intervals> interval(0.0, 2.0);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;

  Pol::Continuous<2>::Inclusion inc(interval);

  auto input = Pol::Continuous<2>(interval);
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
