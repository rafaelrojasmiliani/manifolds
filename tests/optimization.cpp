#define EIGEN_RUNTIME_NO_MALLOC

#include <gtest/gtest.h>

#include <Manifolds/LinearManifolds/GLPolynomial.hpp>
#include <Manifolds/LinearManifolds/GLPolynomialOperators.hpp>
#include <Manifolds/Maps/Map.hpp>
#include <Manifolds/Optimizer/Optimizer.hpp>
#ifdef NDEBUG
#include <chrono>
#endif

using namespace manifolds;

// class TestVar : public ifopt::VariableSet {
// public:
//   TestVar(std::size_t m) : ifopt::VariableSet(m, "variable"), var_(m) {

//     ifopt::Bounds default_bound(-ifopt::inf, ifopt::inf);
//     bounds_ = ifopt::Component::VecBound(m, default_bound);
//   }
//   void SetVariables(const Eigen::VectorXd &_vec) override { var_ = _vec; }
//   Eigen::VectorXd GetValues() const override { return var_; }

//   ifopt::Component::VecBound GetBounds() const override { return bounds_; }

//   ~TestVar() override = default;

//   Eigen::VectorXd var_;
//   ifopt::Component::Component::VecBound bounds_;
// };
// TEST(Optimizer, IfOptDetail) {
//   using ComponentVec = std::vector<ifopt::Component::Ptr>;
//   ComponentVec vec;
//   auto var_ptr = std::make_shared<optimizer::ManifoldVariable<100>>();
//   vec.push_back(var_ptr);

//   Eigen::VectorXd g_all = Eigen::VectorXd::Zero(var_ptr->GetRows());

//   int row = 0;
//   for (const auto &c : vec) {
//     int n_rows = c->GetRows();
//     Eigen::VectorXd g = c->GetValues();
//     g_all.middleRows(row, n_rows) += g;

//     row += n_rows;
//   }
//   vec[0]->GetValues();
// }

// TEST(Optimizer, IfOptComposite) {

//   // constexpr std::size_t size = 1;
//   ifopt::Composite cc("variable-set", false);
//   // auto var_ptr = std::make_shared<TestVar>(size);
//   // cc.AddComponent(var_ptr);
//   // EXPECT_EQ(size, cc.GetRows());

//   const auto &vec = cc.GetComponents();

//   Eigen::VectorXd g_all = Eigen::VectorXd::Zero(cc.GetRows());

//   int row = 0;
//   for (const auto &c : vec) {
//     int n_rows = c->GetRows();
//     Eigen::VectorXd g = c->GetValues();
//     g_all.middleRows(row, n_rows) += g;

//     row += n_rows;
//   }
//   Eigen::VectorXd fook = cc.GetValues();

//   // EXPECT_EQ(size, var_ptr->GetRows());

//   // Eigen::VectorXd vec = Eigen::VectorXd::Random(100);
//   // var_ptr->SetVariables(vec);
//   // Eigen::VectorXd x = nlp.GetVariableValues();
//   // EXPECT_TRUE(x.isApprox(vec)) << "\n"
//   //                              << x.transpose() << "\n"
//   //                              << vec.transpose() << "\n";
// }
// TEST(Optimizer, IfOptNpl) {
//   ifopt::Problem nlp;
//   auto var_ptr = std::make_shared<optimizer::ManifoldVariable<100>>();
//   nlp.AddVariableSet(var_ptr);

//   // cc.GetValues();

//   EXPECT_EQ(100, nlp.GetNumberOfOptimizationVariables());
//   EXPECT_EQ(100, var_ptr->GetRows());

//   // Eigen::VectorXd vec = Eigen::VectorXd::Random(100);
//   // var_ptr->SetVariables(vec);
//   // Eigen::VectorXd x = nlp.GetVariableValues();
//   // EXPECT_TRUE(x.isApprox(vec)) << "\n"
//   //                              << x.transpose() << "\n"
//   //                              << vec.transpose() << "\n";
// }
TEST(GLPolynomial, DriftNorm) {

  constexpr std::size_t intervals = 2;
  constexpr std::size_t nglp = 8;
  constexpr std::size_t dim = 2;
  IntervalPartition<intervals> interval(0.0, 2.0);
  using Pol = PWGLVPolynomial<nglp, intervals, dim>;
  using ScalarPol = PWGLVPolynomial<nglp, intervals, 1>;

  auto trj = Pol::straight_p2p(interval, Pol::codomain_t::random(),
                               Pol::codomain_t::random());

  auto foo = Pol::FromVector(interval) | Pol::Minus(trj) |
             Pol::functions::squared_norm | ScalarPol::Integral(interval);

  auto fv = Pol::FromVector(interval);
  auto ce = Pol::Continuous<2>::ContinuityError(interval);
  auto continuity = fv | ce;
  auto pol = Pol::Continuous<2>::Inclusion(interval)(
      Pol::Continuous<2>::random(interval));
  optimizer::Cost c(foo);
  optimizer::Constraint c2(continuity);
  optimizer::Problem p(c);
  p && (c2);

  Eigen::VectorXd res = p.solve_ipopt();
  std::cout << res.transpose() << "\n";
  std::cout << trj.crepr().transpose() << "\n";
}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
