#pragma once
#include <Eigen/Geometry>
#include <Manifolds/Detail.hpp>
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
    : public detail::Clonable<
          ForwardKinematics<NJoints>,
          Map<DenseLinearManifold<NJoints>, DenseLinearManifold<3>>> {

private:
  pinocchio::Model model_;
  mutable pinocchio::Data data_;
  mutable Eigen::Matrix<double, 6, NJoints> jac_buff_;
  const std::string frame_name_;
  std::size_t frame_index_;

public:
  using codomain_t = DenseLinearManifold<3>;
  using base_t = detail::Clonable<
      ForwardKinematics<NJoints>,
      Map<DenseLinearManifold<NJoints>, DenseLinearManifold<3>>>;

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

  // pinocchio::Data does not support copy construction.
  ForwardKinematics(const ForwardKinematics &_that)
      : model_(_that.model_), data_(_that.model_),
        frame_name_(_that.frame_name_), frame_index_(_that.frame_index_) {}

  virtual bool
  value_on_repr(const Eigen::Matrix<double, base_t::domain_t::dimension, 1> &in,
                Eigen::Matrix<double, base_t::codomain_t::dimension, 1> &out)
      const override {

    pinocchio::forwardKinematics(model_, data_, in);
    pinocchio::updateFramePlacements(model_, data_);

    out.head(3) = data_.oMf[model_.getFrameId(frame_name_)].translation();

    // out.tail(4) = Eigen::Quaternion(data_.oMi[0].rotation().matrix());

    return true;
  }

  virtual bool diff_from_repr(
      const Eigen::Matrix<double, base_t::domain_t::dimension, 1> &in,
      typename codomain_t::Representation &_out,
      detail::dense_matrix_ref_t _mat) const override {

    pinocchio::forwardKinematics(model_, data_, in);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, in);
    pinocchio::getFrameJacobian(model_, data_, frame_index_,
                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                jac_buff_);

    _out.template head<3>() =
        data_.oMf[model_.getFrameId(frame_name_)].translation();

    _mat = this->jac_buff_.template topRows<3>();
    return true;
  }
};
} // namespace manifolds
