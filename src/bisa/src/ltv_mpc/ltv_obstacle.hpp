#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <vector>

#include "ltv_types.hpp"

namespace bisa {

struct FrenetObstacle {
  bool detected{false};
  double s_obs{0.0};
  double d_obs{0.0};
  double d_min{0.0};
  double d_max{0.0};
  double s_rear{0.0};
  double s_front{0.0};
  double d_extent{0.0};
  double s_extent{0.0};
};

struct ObstacleAvoidanceConfig {
  double vehicle_length{0.33};
  double vehicle_width{0.15};
  double circle_radius{0.12};

  double obs_body_length{0.46};
  double obs_body_width{0.28};
  double obs_s_margin{0.08};

  double d_safe{0.35};
  double detect_range_s{2.5};

  double activation_sigma_s{0.50};
  double activation_back_s{0.30};
  double activation_front_s{0.95};
  double activation_ramp_s{0.16};
  double extra_push{0.15};

  double road_d_upper{0.50};
  double road_d_lower{-0.50};

  double near_slowdown_s{0.70};
  double near_release_s{1.30};
  double near_speed{0.10};

  double center_deadband{0.12};
  double preferred_sign{1.0};
};

class LTVObstacle {
 public:
  explicit LTVObstacle(const ObstacleAvoidanceConfig& cfg);

  void setConfig(const ObstacleAvoidanceConfig& cfg);

  FrenetObstacle projectToFrenet(
      const geometry_msgs::msg::Pose& ego_pose,
      const geometry_msgs::msg::Pose& obs_pose,
      const std::vector<geometry_msgs::msg::PoseStamped>& path) const;

  void computeTimeVaryingBounds(
      const FrenetObstacle& obs,
      const Eigen::VectorXd& v_bar,
      double Ts,
      int N,
      std::vector<double>& d_upper_seq,
      std::vector<double>& d_lower_seq) const;

  void buildAsymmetricLateralConstraints(
      const Eigen::VectorXd& x0,
      const Eigen::VectorXd& z_bar,
      const Eigen::MatrixXd& A_bar,
      const Eigen::MatrixXd& B_bar,
      const Eigen::MatrixXd& E_bar,
      const Eigen::MatrixXd& C_bar,
      const std::vector<double>& d_upper_seq,
      const std::vector<double>& d_lower_seq,
      double w_slack_lin,
      double w_slack_quad,
      Eigen::SparseMatrix<double>& A_out,
      Eigen::VectorXd& l_out,
      Eigen::VectorXd& u_out,
      int nvar) const;

  bool chooseAvoidRight(
      const FrenetObstacle& obs,
      double d_upper,
      double d_lower) const;

 private:
  ObstacleAvoidanceConfig cfg_;

  static double wrapAngle(double a);
  static double quatToYaw(const geometry_msgs::msg::Quaternion& q);
  static double sigmoid(double x);
};

}  // namespace bisa
