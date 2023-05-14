#pragma once
#include <Eigen/Geometry>
#include <Manifolds/LinearManifolds/LinearManifolds.hpp>
#include <Manifolds/LinearManifolds/Reals.hpp>
#include <Manifolds/ManifoldBase.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace manifolds {

template <std::size_t NJoints>
class ForwardKinematics
    : public MapInheritanceHelper<
          ForwardKinematics<NJoints>,
          Map<LinearManifold<NJoints>, LinearManifold<7>, false>> {

private:
  pinocchio::Model model_;
  mutable pinocchio::Data data_;
  const std::string frame_name_;
  std::size_t frame_index_;

public:
  using Domain = LinearManifold<NJoints>;
  using Codomain = LinearManifold<7>;

  static ForwardKinematics from_urdf(const std::string &_path,
                                     const std::string &_frame) {
    pinocchio::Model model;
    pinocchio::urdf::buildModel(_path, model);

    return ForwardKinematics(model, _frame);
  }

  ForwardKinematics(const pinocchio::Model &model,
                    const std::string &_frame_name)
      : model_(model), data_(model_), frame_name_(_frame_name) {

    frame_index_ = model.getFrameId(_frame_name);

    std::size_t non_fixed_joints = 0;
    for (const auto &j : model_.joints) {
      // std::cout << j << "\n---\n" << j.shortname() << "\n.....\n";
      if (j.idx_q() >= 0)
        non_fixed_joints++;
    }
    if (static_cast<std::size_t>(non_fixed_joints) != NJoints)
      throw std::invalid_argument(std::string("Invalid number of joints") +
                                  std::to_string(model_.njoints) + " and " +
                                  std::to_string(NJoints));
  }

  virtual bool value_on_repr(
      const Eigen::Matrix<double, Domain::dimension, 1> &in,
      Eigen::Matrix<double, Codomain::dimension, 1> &out) const override {

    pinocchio::forwardKinematics(model_, data_, in);
    pinocchio::updateFramePlacements(model_, data_);

    out.head(3) = data_.oMf[model_.getFrameId(frame_name_)].translation();

    // out.tail(4) = Eigen::Quaternion(data_.oMi[0].rotation().matrix());

    return true;
  }

  virtual bool
  diff_from_repr(const Eigen::Matrix<double, Domain::dimension, 1> &in,
                 detail::DifferentialReprRef_t<false> _mat) const override {

    pinocchio::computeJointJacobians(model_, data_, in);
    pinocchio::getFrameJacobian(model_, data_, frame_index_,
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                _mat);
    return true;
  }
};
} // namespace manifolds
