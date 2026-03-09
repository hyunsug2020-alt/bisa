#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geometry_msgs/msg/accel.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>

#include "ltv_mpc/ltv_cost.hpp"
#include "ltv_mpc/ltv_model.hpp"
#include "ltv_mpc/ltv_solver.hpp"
#include "ltv_mpc/ltv_types.hpp"

namespace bisa {

struct ObstacleInfo {
  bool detected{false};
  double d_obs{0.0};
  double s_obs{0.0};
  double d_min{0.0};
  double d_max{0.0};
  double s_rear{0.0};
  double s_front{0.0};
  double d_extent{0.0};
  double s_extent{0.0};
};

class MpcObstacleAvoidanceCpp : public rclcpp::Node {
 public:
  MpcObstacleAvoidanceCpp()
      : Node("mpc_obstacle_avoidance_cpp"), model_(cfg_), cost_(cfg_), solver_(cfg_) {
    declare_parameter("N", 25);
    declare_parameter("Ts", 0.05);
    declare_parameter("horizon", -1);
    declare_parameter("time_step", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("wheelbase", 0.30);
    declare_parameter("max_velocity", 0.72);
    declare_parameter("min_velocity", 0.30);
    declare_parameter("w_d", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("w_theta", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("w_kappa", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("w_u", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("Q_pos", 20.0);
    declare_parameter("Q_heading", 8.0);
    declare_parameter("weight_curvature", 1.0);
    declare_parameter("weight_input", 1.5);
    declare_parameter("lateral_bound", 0.30);
    declare_parameter("w_lateral_slack_lin", 500.0);
    declare_parameter("w_lateral_slack_quad", 5000.0);
    declare_parameter("kappa_min", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("kappa_max", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("kappa_min_delta", -2.0);
    declare_parameter("kappa_max_delta", 2.0);
    declare_parameter("u_min", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("u_max", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("max_accel", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("max_angular_vel", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("max_omega_abs", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("max_omega_rate", std::numeric_limits<double>::quiet_NaN());
    declare_parameter("obstacle_max_omega_abs", 1.70);
    declare_parameter("obstacle_max_omega_rate", 14.0);
    declare_parameter("kappa_limit_ref_velocity", -1.0);
    declare_parameter("curvature_speed_gain", 2.5);
    declare_parameter("kappa_blend_alpha", 0.15);
    declare_parameter("kappa_blend_alpha_obstacle", 0.02);
    declare_parameter("osqp_max_iter", 4000);
    declare_parameter("osqp_eps_abs", 1e-4);
    declare_parameter("osqp_eps_rel", 1e-4);
    declare_parameter("osqp_polish", false);

    declare_parameter("d_safe", 0.40);
    declare_parameter("detect_range_s", 2.2);
    declare_parameter("obstacle_sigma_s", 0.50);
    declare_parameter("obstacle_body_length", 0.46);
    declare_parameter("obstacle_body_width", 0.28);
    declare_parameter("obstacle_body_s_margin", 0.08);
    declare_parameter("avoidance_lateral_bound_min", 0.22);
    declare_parameter("obstacle_center_deadband", 0.12);
    declare_parameter("obstacle_preferred_sign", 1.0);
    declare_parameter("obstacle_near_slowdown_s", 0.70);
    declare_parameter("obstacle_near_release_s", 1.30);
    declare_parameter("obstacle_near_speed", 0.10);
    declare_parameter("obstacle_activation_back_s", 0.30);
    declare_parameter("obstacle_activation_front_s", 0.95);
    declare_parameter("obstacle_activation_ramp_s", 0.16);
    declare_parameter("obstacle_extra_push", 0.18);
    declare_parameter("obstacle_memory_sec", 0.90);
    declare_parameter("max_v_rate_up", 0.45);
    declare_parameter("max_v_rate_down", 0.90);
    declare_parameter("obstacle_clear_back_s", 0.10);
    declare_parameter("obstacle_turn_boost_enable", true);
    declare_parameter("obstacle_turn_boost_front_s", 0.95);
    declare_parameter("obstacle_turn_boost_max_s", 0.25);
    declare_parameter("obstacle_turn_boost_lateral_gate", 0.28);
    declare_parameter("obstacle_turn_boost_omega", 1.25);
    declare_parameter("obstacle_turn_boost_min_omega", 0.50);
    declare_parameter("obstacle_turn_boost_blend", 0.78);
    declare_parameter("no_obstacle_speed_guard_enable", true);
    declare_parameter("no_obstacle_speed_cte_gain", 2.0);
    declare_parameter("no_obstacle_speed_heading_gain", 0.7);
    declare_parameter("no_obstacle_speed_min_ratio", 0.30);
    declare_parameter("curve_speed_enable", true);
    declare_parameter("curve_speed_heading_threshold", 0.16);
    declare_parameter("curve_speed_reduction_gain", 0.04);
    declare_parameter("curve_speed_min_ratio", 0.96);
    declare_parameter("path_hold_enable", true);
    declare_parameter("path_hold_max_omega", 1.40);
    declare_parameter("path_hold_recovery_distance", 0.15);
    declare_parameter("path_hold_heading_error_gain", 0.80);
    declare_parameter("path_hold_cross_track_gain", 0.70);
    declare_parameter("path_hold_recovery_blend", 0.45);
    declare_parameter("overshoot_guard_distance", 0.13);
    declare_parameter("overshoot_reverse_damping", 0.52);
    declare_parameter("oscillation_guard_enable", true);
    declare_parameter("oscillation_guard_cte_deadband", 0.08);
    declare_parameter("oscillation_guard_heading_deadband", 0.20);
    declare_parameter("oscillation_guard_reverse_damping", 0.70);
    declare_parameter("adaptive_corner_mode_enable", true);
    declare_parameter("adaptive_near_cte_thresh", 0.14);
    declare_parameter("adaptive_near_heading_thresh", 0.14);
    declare_parameter("adaptive_near_omega_damping", 0.55);
    declare_parameter("adaptive_near_omega_rate_scale", 0.55);
    declare_parameter("adaptive_near_v_scale", 0.97);
    declare_parameter("sigmoid_tau_accel", 0.40);
    declare_parameter("sigmoid_tau_decel", 0.20);
    declare_parameter("return_recovery_enable", true);
    declare_parameter("return_recovery_enter_cte", 0.12);
    declare_parameter("return_recovery_exit_cte", 0.04);
    declare_parameter("return_recovery_speed", 0.16);
    declare_parameter("return_recovery_heading_gain", 1.2);
    declare_parameter("return_recovery_cte_gain", 2.2);
    declare_parameter("return_recovery_max_omega", 1.6);
    declare_parameter("return_recovery_blend", 0.65);
    declare_parameter("return_recovery_post_obs_sec", 1.4);
    declare_parameter("post_obstacle_recenter_sec", 2.2);
    declare_parameter("post_obstacle_kappa_blend_alpha", 0.40);
    declare_parameter("post_obstacle_max_speed", 0.20);

    loadConfig();

    const auto sensor_qos = rclcpp::SensorDataQoS();
    const auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

    ego_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/CAV_01", sensor_qos,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { ego_pose_ = msg->pose; });
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/cav01/local_path", rclcpp::QoS(10),
        [this](const nav_msgs::msg::Path::SharedPtr msg) { local_path_ = msg->poses; });
    obs_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/CAV_02", sensor_qos,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { obs_pose_ = *msg; });

    cav01_pub_ = create_publisher<geometry_msgs::msg::Accel>("/sim/cav01/accel", cmd_qos);
    zero_pub_[0] = create_publisher<geometry_msgs::msg::Accel>("/sim/cav02/accel", cmd_qos);
    zero_pub_[1] = create_publisher<geometry_msgs::msg::Accel>("/sim/cav03/accel", cmd_qos);
    zero_pub_[2] = create_publisher<geometry_msgs::msg::Accel>("/sim/cav04/accel", cmd_qos);

    prev_cmd_time_ = this->now();
    timer_ = create_wall_timer(std::chrono::milliseconds(50),
                               std::bind(&MpcObstacleAvoidanceCpp::controlLoop, this));
    RCLCPP_INFO(get_logger(),
                "MPC Obstacle Avoidance ready (CAV1 control, CAV2~4 forced zero)");
  }

 private:
  static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  static double wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }
  static double quatToYaw(const geometry_msgs::msg::Quaternion& q) {
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const bool quat_component_range_ok =
        std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w) &&
        std::abs(q.x) <= 1.0 + 1e-3 && std::abs(q.y) <= 1.0 + 1e-3 &&
        std::abs(q.z) <= 1.0 + 1e-3 && std::abs(q.w) <= 1.0 + 1e-3;
    const bool use_quaternion =
        quat_component_range_ok && std::isfinite(n) && std::abs(n - 1.0) <= 0.05;
    if (!use_quaternion) {
      // Simulator often publishes packed yaw in orientation.z (not a valid unit quaternion).
      double yaw = q.z;
      if (!std::isfinite(yaw)) return 0.0;
      if (std::abs(yaw) > 2.0 * M_PI + 0.5) {
        yaw = yaw * M_PI / 180.0;
      } else if (yaw > M_PI && yaw <= 2.0 * M_PI + 0.5) {
        yaw -= 2.0 * M_PI;
      } else if (yaw < -M_PI && yaw >= -2.0 * M_PI - 0.5) {
        yaw += 2.0 * M_PI;
      }
      return wrapAngle(yaw);
    }
    const double x = q.x / n;
    const double y = q.y / n;
    const double z = q.z / n;
    const double w = q.w / n;
    const double siny_cosp = 2.0 * (w * z + x * y);
    const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return wrapAngle(std::atan2(siny_cosp, cosy_cosp));
  }
  bool chooseRightByFreeSpace(const ObstacleInfo& obs, double d_upper, double d_lower,
                              double d_safe_eff) const {
    if (obs.d_min >= 0.0) return true;
    if (obs.d_max <= 0.0) return false;
    const double right_limit = obs.d_min - d_safe_eff;
    const double left_limit = obs.d_max + d_safe_eff;
    const double right_free = right_limit - d_lower;
    const double left_free = d_upper - left_limit;
    return (right_free > left_free + 1e-3) ||
           (std::abs(right_free - left_free) <= 1e-3 && obstacle_preferred_sign_ >= 0.0);
  }

  void loadConfig() {
    const auto read_double = [this](const char* name) { return get_parameter(name).as_double(); };

    cfg_.N = std::max(5, static_cast<int>(get_parameter("N").as_int()));
    const int horizon = static_cast<int>(get_parameter("horizon").as_int());
    if (horizon > 0) cfg_.N = horizon;

    cfg_.Ts = std::max(1e-3, read_double("Ts"));
    const double time_step = read_double("time_step");
    if (std::isfinite(time_step) && time_step > 1e-4) cfg_.Ts = time_step;

    cfg_.wheelbase = std::max(0.1, read_double("wheelbase"));
    cfg_.max_velocity = std::max(0.0, read_double("max_velocity"));
    cfg_.min_velocity = std::max(0.0, read_double("min_velocity"));
    if (cfg_.max_velocity < cfg_.min_velocity) std::swap(cfg_.max_velocity, cfg_.min_velocity);

    cfg_.w_d = read_double("Q_pos");
    cfg_.w_theta = read_double("Q_heading");
    cfg_.w_kappa = read_double("weight_curvature");
    cfg_.w_u = read_double("weight_input");
    const double wd = read_double("w_d");
    const double wtheta = read_double("w_theta");
    const double wkappa = read_double("w_kappa");
    const double wu = read_double("w_u");
    if (std::isfinite(wd)) cfg_.w_d = wd;
    if (std::isfinite(wtheta)) cfg_.w_theta = wtheta;
    if (std::isfinite(wkappa)) cfg_.w_kappa = wkappa;
    if (std::isfinite(wu)) cfg_.w_u = wu;

    cfg_.lateral_bound = std::max(0.05, read_double("lateral_bound"));
    cfg_.lateral_bound = std::max(cfg_.lateral_bound, read_double("avoidance_lateral_bound_min"));
    cfg_.w_lateral_slack_lin = std::max(0.0, read_double("w_lateral_slack_lin"));
    cfg_.w_lateral_slack_quad = std::max(0.0, read_double("w_lateral_slack_quad"));

    cfg_.u_min = -2.0;
    cfg_.u_max = 2.0;
    const double u_min = read_double("u_min");
    const double u_max = read_double("u_max");
    if (std::isfinite(u_min)) cfg_.u_min = u_min;
    if (std::isfinite(u_max)) cfg_.u_max = u_max;
    const double max_accel = read_double("max_accel");
    if (std::isfinite(max_accel) && max_accel > 0.0) {
      cfg_.u_min = -max_accel;
      cfg_.u_max = max_accel;
    }
    if (cfg_.u_min > cfg_.u_max) std::swap(cfg_.u_min, cfg_.u_max);

    cfg_.kappa_min = read_double("kappa_min_delta");
    cfg_.kappa_max = read_double("kappa_max_delta");
    const double kappa_min = read_double("kappa_min");
    const double kappa_max = read_double("kappa_max");
    if (std::isfinite(kappa_min)) cfg_.kappa_min = kappa_min;
    if (std::isfinite(kappa_max)) cfg_.kappa_max = kappa_max;

    const double max_angular_vel = read_double("max_angular_vel");
    if (std::isfinite(max_angular_vel) && max_angular_vel > 0.0) {
      const double v_ref_param = read_double("kappa_limit_ref_velocity");
      const double v_ref =
          (std::isfinite(v_ref_param) && v_ref_param > 0.0)
              ? v_ref_param
              : std::max(cfg_.min_velocity, 0.1);
      const double kappa_limit = max_angular_vel / v_ref;
      cfg_.kappa_min = -kappa_limit;
      cfg_.kappa_max = kappa_limit;
    }
    if (cfg_.kappa_min > cfg_.kappa_max) std::swap(cfg_.kappa_min, cfg_.kappa_max);
    const double max_omega_abs_param = read_double("max_omega_abs");
    max_omega_abs_ = (std::isfinite(max_omega_abs_param) && max_omega_abs_param > 0.05)
                         ? max_omega_abs_param
                         : ((std::isfinite(max_angular_vel) && max_angular_vel > 0.05)
                                ? max_angular_vel
                                : 1.6);
    max_omega_abs_ = std::max(0.1, max_omega_abs_);
    const double max_omega_rate_param = read_double("max_omega_rate");
    max_omega_rate_ =
        (std::isfinite(max_omega_rate_param) && max_omega_rate_param > 0.05)
            ? max_omega_rate_param
            : 12.0;
    obstacle_max_omega_abs_ = std::max(max_omega_abs_, std::max(0.1, read_double("obstacle_max_omega_abs")));
    obstacle_max_omega_rate_ =
        std::max(max_omega_rate_, std::max(0.1, read_double("obstacle_max_omega_rate")));

    cfg_.curvature_speed_gain = std::max(0.0, read_double("curvature_speed_gain"));
    cfg_.kappa_blend_alpha = std::clamp(read_double("kappa_blend_alpha"), 0.0, 1.0);
    kappa_blend_alpha_obstacle_ =
        std::clamp(read_double("kappa_blend_alpha_obstacle"), 0.0, 1.0);
    cfg_.osqp_max_iter = std::max(100, static_cast<int>(get_parameter("osqp_max_iter").as_int()));
    cfg_.osqp_eps_abs = std::max(1e-7, read_double("osqp_eps_abs"));
    cfg_.osqp_eps_rel = std::max(1e-7, read_double("osqp_eps_rel"));
    cfg_.osqp_polish = get_parameter("osqp_polish").as_bool();

    d_safe_ = std::max(0.05, read_double("d_safe"));
    detect_range_s_ = std::max(0.3, read_double("detect_range_s"));
    obstacle_sigma_s_ = std::max(0.2, read_double("obstacle_sigma_s"));
    obstacle_body_length_ = std::max(0.10, read_double("obstacle_body_length"));
    obstacle_body_width_ = std::max(0.10, read_double("obstacle_body_width"));
    obstacle_body_s_margin_ = std::max(0.0, read_double("obstacle_body_s_margin"));
    obstacle_center_deadband_ = std::max(0.01, read_double("obstacle_center_deadband"));
    obstacle_preferred_sign_ = (read_double("obstacle_preferred_sign") >= 0.0) ? 1.0 : -1.0;
    obstacle_near_slowdown_s_ = std::max(0.1, read_double("obstacle_near_slowdown_s"));
    obstacle_near_release_s_ = std::max(obstacle_near_slowdown_s_ + 0.05,
                                        read_double("obstacle_near_release_s"));
    obstacle_near_speed_ = std::max(0.03, read_double("obstacle_near_speed"));
    obstacle_activation_back_s_ = std::max(0.05, read_double("obstacle_activation_back_s"));
    obstacle_activation_front_s_ = std::max(0.10, read_double("obstacle_activation_front_s"));
    obstacle_activation_ramp_s_ = std::max(0.05, read_double("obstacle_activation_ramp_s"));
    obstacle_extra_push_ = std::max(0.0, read_double("obstacle_extra_push"));
    obstacle_memory_sec_ = std::max(0.0, read_double("obstacle_memory_sec"));
    max_v_rate_up_ = std::max(0.05, read_double("max_v_rate_up"));
    max_v_rate_down_ = std::max(0.05, read_double("max_v_rate_down"));
    obstacle_clear_back_s_ = std::max(0.02, read_double("obstacle_clear_back_s"));
    obstacle_turn_boost_enable_ = get_parameter("obstacle_turn_boost_enable").as_bool();
    obstacle_turn_boost_front_s_ = std::max(0.15, read_double("obstacle_turn_boost_front_s"));
    obstacle_turn_boost_max_s_ =
        std::clamp(read_double("obstacle_turn_boost_max_s"), 0.05, obstacle_turn_boost_front_s_);
    obstacle_turn_boost_lateral_gate_ =
        std::max(0.02, read_double("obstacle_turn_boost_lateral_gate"));
    obstacle_turn_boost_omega_ = std::max(0.0, read_double("obstacle_turn_boost_omega"));
    obstacle_turn_boost_min_omega_ = std::max(0.0, read_double("obstacle_turn_boost_min_omega"));
    obstacle_turn_boost_blend_ = std::clamp(read_double("obstacle_turn_boost_blend"), 0.0, 1.0);
    no_obstacle_speed_guard_enable_ = get_parameter("no_obstacle_speed_guard_enable").as_bool();
    no_obstacle_speed_cte_gain_ = std::max(0.0, read_double("no_obstacle_speed_cte_gain"));
    no_obstacle_speed_heading_gain_ =
        std::max(0.0, read_double("no_obstacle_speed_heading_gain"));
    no_obstacle_speed_min_ratio_ =
        std::clamp(read_double("no_obstacle_speed_min_ratio"), 0.05, 1.0);
    curve_speed_enable_ = get_parameter("curve_speed_enable").as_bool();
    curve_speed_heading_threshold_ = std::max(0.01, read_double("curve_speed_heading_threshold"));
    curve_speed_reduction_gain_ = std::max(0.0, read_double("curve_speed_reduction_gain"));
    curve_speed_min_ratio_ = std::clamp(read_double("curve_speed_min_ratio"), 0.05, 1.0);
    path_hold_enable_ = get_parameter("path_hold_enable").as_bool();
    path_hold_max_omega_ = std::max(0.1, read_double("path_hold_max_omega"));
    path_hold_recovery_distance_ = std::max(0.02, read_double("path_hold_recovery_distance"));
    path_hold_heading_error_gain_ = std::max(0.0, read_double("path_hold_heading_error_gain"));
    path_hold_cross_track_gain_ = std::max(0.0, read_double("path_hold_cross_track_gain"));
    path_hold_recovery_blend_ = std::clamp(read_double("path_hold_recovery_blend"), 0.0, 1.0);
    overshoot_guard_distance_ = std::max(0.01, read_double("overshoot_guard_distance"));
    overshoot_reverse_damping_ =
        std::clamp(read_double("overshoot_reverse_damping"), 0.05, 1.0);
    oscillation_guard_enable_ = get_parameter("oscillation_guard_enable").as_bool();
    oscillation_guard_cte_deadband_ =
        std::max(0.0, read_double("oscillation_guard_cte_deadband"));
    oscillation_guard_heading_deadband_ =
        std::max(0.0, read_double("oscillation_guard_heading_deadband"));
    oscillation_guard_reverse_damping_ =
        std::clamp(read_double("oscillation_guard_reverse_damping"), 0.05, 1.0);
    adaptive_corner_mode_enable_ = get_parameter("adaptive_corner_mode_enable").as_bool();
    adaptive_near_cte_thresh_ = std::max(0.0, read_double("adaptive_near_cte_thresh"));
    adaptive_near_heading_thresh_ = std::max(0.0, read_double("adaptive_near_heading_thresh"));
    adaptive_near_omega_damping_ =
        std::clamp(read_double("adaptive_near_omega_damping"), 0.05, 1.0);
    adaptive_near_omega_rate_scale_ =
        std::clamp(read_double("adaptive_near_omega_rate_scale"), 0.05, 1.0);
    adaptive_near_v_scale_ = std::clamp(read_double("adaptive_near_v_scale"), 0.05, 1.0);
    sigmoid_tau_accel_ = std::max(0.01, read_double("sigmoid_tau_accel"));
    sigmoid_tau_decel_ = std::max(0.01, read_double("sigmoid_tau_decel"));
    return_recovery_enable_ = get_parameter("return_recovery_enable").as_bool();
    return_recovery_enter_cte_ = std::max(0.02, read_double("return_recovery_enter_cte"));
    return_recovery_exit_cte_ = std::clamp(read_double("return_recovery_exit_cte"), 0.01,
                                           return_recovery_enter_cte_ - 0.01);
    return_recovery_speed_ = std::max(0.03, read_double("return_recovery_speed"));
    return_recovery_heading_gain_ = std::max(0.0, read_double("return_recovery_heading_gain"));
    return_recovery_cte_gain_ = std::max(0.0, read_double("return_recovery_cte_gain"));
    return_recovery_max_omega_ = std::max(0.2, read_double("return_recovery_max_omega"));
    return_recovery_blend_ = std::clamp(read_double("return_recovery_blend"), 0.0, 1.0);
    return_recovery_post_obs_sec_ = std::max(0.2, read_double("return_recovery_post_obs_sec"));
    post_obstacle_recenter_sec_ = std::max(0.2, read_double("post_obstacle_recenter_sec"));
    post_obstacle_kappa_blend_alpha_ =
        std::clamp(read_double("post_obstacle_kappa_blend_alpha"), 0.0, 1.0);
    post_obstacle_max_speed_ = std::max(0.05, read_double("post_obstacle_max_speed"));

    model_.setConfig(cfg_);
    cost_.setConfig(cfg_);
    solver_.setConfig(cfg_);
  }

  void controlLoop() {
    publishZero();

    const auto now = this->now();
    double dt = (now - prev_cmd_time_).seconds();
    if (!std::isfinite(dt) || dt <= 1e-4) dt = 0.05;
    prev_cmd_time_ = now;

    if (!ego_pose_.has_value() || local_path_.size() < 5) {
      const double v_cmd = applySpeedRateLimit(0.0, dt);
      publishCmd(v_cmd, 0.0);
      return;
    }

    ObstacleInfo obs;
    if (obs_pose_.has_value()) {
      obs = projectObstacle(local_path_, ego_pose_.value(), obs_pose_->pose);
    }

    if (obs.detected) {
      last_obs_ = obs;
      last_obs_time_ = now;
      has_last_obs_ = true;
    } else if (has_last_obs_) {
      const double age = (now - last_obs_time_).seconds();
      if (age <= obstacle_memory_sec_) {
        const double travel = std::max(prev_v_cmd_, 0.05) * age;
        const double s_mem = last_obs_.s_obs - travel;
        const double s_rear_mem = last_obs_.s_rear - travel;
        const double s_front_mem = last_obs_.s_front - travel;
        if (s_front_mem > -obstacle_clear_back_s_) {
          obs.detected = true;
          obs.d_obs = last_obs_.d_obs;
          obs.s_obs = s_mem;
          obs.d_min = last_obs_.d_min;
          obs.d_max = last_obs_.d_max;
          obs.s_rear = s_rear_mem;
          obs.s_front = s_front_mem;
          obs.d_extent = last_obs_.d_extent;
          obs.s_extent = last_obs_.s_extent;
        }
      }
    }

    ObstacleInfo obs_ctrl = obs;
    if (prev_obs_detected_ && !obs_ctrl.detected) {
      post_obstacle_until_ = now + rclcpp::Duration::from_seconds(post_obstacle_recenter_sec_);
    }
    const bool post_obstacle_mode =
        (!obs_ctrl.detected) && ((post_obstacle_until_ - now).seconds() > 0.0);
    prev_obs_detected_ = obs_ctrl.detected;

    const double d_upper = cfg_.lateral_bound;
    const double d_lower = -cfg_.lateral_bound;
    if (obs_ctrl.detected) {
      const bool overlap_center = (obs_ctrl.d_min <= 0.0 && obs_ctrl.d_max >= 0.0);
      if (overlap_center) {
        const bool choose_right = chooseRightByFreeSpace(obs_ctrl, d_upper, d_lower, d_safe_);
        obs_ctrl.d_obs = choose_right ? obstacle_center_deadband_ : -obstacle_center_deadband_;
      } else if (std::abs(obs_ctrl.d_obs) < obstacle_center_deadband_) {
        obs_ctrl.d_obs = obstacle_preferred_sign_ * obstacle_center_deadband_;
      }
    }

    if (obs_ctrl.detected) {
      double d_upper_target = d_upper;
      double d_lower_target = d_lower;
      const bool choose_right = chooseRightByFreeSpace(obs_ctrl, d_upper, d_lower, d_safe_);
      if (obs_ctrl.d_min >= 0.0 || (obs_ctrl.d_min < 0.0 && obs_ctrl.d_max > 0.0 && choose_right)) {
        d_upper_target = std::min(d_upper, obs_ctrl.d_min - d_safe_);
      } else {
        d_lower_target = std::max(d_lower, obs_ctrl.d_max + d_safe_);
      }
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 400,
          "[OBS] s=[%.2f,%.2f] d=[%.3f,%.3f] dc=%.3f base=[%.3f, %.3f] target=[%.3f, %.3f] "
          "act=[-%.2f,+%.2f]",
          obs_ctrl.s_rear, obs_ctrl.s_front, obs_ctrl.d_min, obs_ctrl.d_max, obs_ctrl.d_obs,
          d_lower, d_upper, d_lower_target, d_upper_target, obstacle_activation_back_s_,
          obstacle_activation_front_s_);
    }

    const MPCCommand cmd =
        computeWithObstacle(ego_pose_.value(), local_path_, d_upper, d_lower, obs_ctrl,
                            post_obstacle_mode);

    if (obs_ctrl.detected && obs_ctrl.s_rear < obstacle_near_slowdown_s_) {
      slow_mode_latched_ = true;
    } else if (slow_mode_latched_) {
      if ((!obs_ctrl.detected) ||
          (obs_ctrl.detected &&
           (obs_ctrl.s_rear > obstacle_near_release_s_ ||
            obs_ctrl.s_front < -obstacle_clear_back_s_))) {
        slow_mode_latched_ = false;
      }
    }
    if (!cmd.solved) {
      slow_mode_latched_ = true;
      const double v_cmd = applySpeedRateLimit(0.0, dt);
      publishCmd(v_cmd, 0.0);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "[MPC] solve failed, decelerating safely");
      return;
    }

    double v_target = cmd.v_cmd;
    double omega_target = cmd.omega_cmd;
    double cte_ctrl = last_dr_;
    double heading_ctrl = last_heading_err_;
    double signed_cte_near = 0.0;
    double heading_error_near = 0.0;
    const bool near_metrics_valid = computeNearPathMetrics(
        ego_pose_.value(), local_path_, signed_cte_near, heading_error_near);
    if (near_metrics_valid) {
      cte_ctrl = signed_cte_near;
      heading_ctrl = heading_error_near;
    }
    if (post_obstacle_mode) {
      v_target = std::min(v_target, post_obstacle_max_speed_);
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 500,
          "[POST] recenter active cte=%.3f heading=%.3f v_cap=%.3f", cte_ctrl, heading_ctrl,
          post_obstacle_max_speed_);
    }
    if (slow_mode_latched_) {
      v_target = std::min(v_target, obstacle_near_speed_);
    }
    const bool recent_obstacle =
        has_last_obs_ && ((now - last_obs_time_).seconds() < return_recovery_post_obs_sec_);
    const double boost_range_s =
        std::max(0.05, std::min(obstacle_turn_boost_front_s_, obstacle_turn_boost_max_s_));
    const double s_nearest_front = std::max(0.0, obs_ctrl.s_rear);
    const double d_nearest_centerline =
        (obs_ctrl.d_min > 0.0) ? obs_ctrl.d_min : ((obs_ctrl.d_max < 0.0) ? -obs_ctrl.d_max : 0.0);
    if (obstacle_turn_boost_enable_ && obs_ctrl.detected && s_nearest_front > 0.0 &&
        s_nearest_front < boost_range_s &&
        d_nearest_centerline < obstacle_turn_boost_lateral_gate_) {
      const double closeness =
          std::clamp((boost_range_s - s_nearest_front) / boost_range_s, 0.0, 1.0);
      const bool choose_right = chooseRightByFreeSpace(obs_ctrl, d_upper, d_lower, d_safe_);
      const double avoid_sign = choose_right ? -1.0 : 1.0;
      const double w_boost = avoid_sign * obstacle_turn_boost_omega_ * closeness;
      const double w_min = avoid_sign * obstacle_turn_boost_min_omega_ * closeness;
      omega_target =
          (1.0 - obstacle_turn_boost_blend_) * omega_target + obstacle_turn_boost_blend_ * w_boost;
      if (omega_target * avoid_sign < std::abs(w_min)) {
        omega_target = w_min;
      }
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 350,
                           "[AVOID] s_near=%.2f d_near=%.3f d=[%.3f,%.3f] close=%.2f sign=%.0f "
                           "w_boost=%.3f w_out=%.3f",
                           s_nearest_front, d_nearest_centerline, obs_ctrl.d_min, obs_ctrl.d_max,
                           closeness, avoid_sign, w_boost, omega_target);
    }
    const bool allow_return_recovery = recent_obstacle || post_obstacle_mode;
    if (return_recovery_enable_ && !obs_ctrl.detected && allow_return_recovery) {
      if (!return_recovery_latched_ && std::abs(cte_ctrl) > return_recovery_enter_cte_) {
        return_recovery_latched_ = true;
      } else if (return_recovery_latched_ &&
                 std::abs(cte_ctrl) < return_recovery_exit_cte_) {
        return_recovery_latched_ = false;
      }
    } else {
      return_recovery_latched_ = false;
    }
    if (return_recovery_latched_) {
      const double h_abs = std::abs(heading_ctrl);
      const double h_soft = 0.55;
      const double h_hard = 1.10;
      const double heading_scale =
          (h_abs <= h_soft) ? 1.0
                            : std::clamp((h_hard - h_abs) / std::max(1e-3, h_hard - h_soft), 0.2, 1.0);

      const double heading_term = heading_scale * return_recovery_heading_gain_ * heading_ctrl;
      const double cte_term = -return_recovery_cte_gain_ * cte_ctrl;
      const double w_recover = std::clamp(
          heading_term + cte_term, -return_recovery_max_omega_, return_recovery_max_omega_);

      const double cte_ref = std::max(return_recovery_enter_cte_, 1e-3);
      const double cte_norm = std::clamp(std::abs(cte_ctrl) / cte_ref, 0.0, 1.0);
      const double blend_floor = 0.35;
      const double blend = std::clamp(
          blend_floor + (return_recovery_blend_ - blend_floor) * cte_norm, 0.2,
          return_recovery_blend_);
      omega_target = (1.0 - blend) * omega_target + blend * w_recover;

      // If corrective yaw-rate would increase CTE, damp it to avoid "looking backward".
      if ((omega_target * cte_ctrl) > 0.0) {
        omega_target *= 0.7;
      }
      const double speed_scale = (h_abs > 0.9) ? 0.65 : 1.0;
      v_target = std::min(v_target, speed_scale * return_recovery_speed_);
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                           "[RETURN] cte=%.3f heading=%.3f hs=%.2f blend=%.2f w_out=%.3f",
                           cte_ctrl, heading_ctrl, heading_scale, blend, omega_target);
    }

    const bool clear_path_mode = (!obs_ctrl.detected) && (!post_obstacle_mode);
    if (clear_path_mode && curve_speed_enable_) {
      const double h_abs = std::abs(heading_ctrl);
      if (h_abs > curve_speed_heading_threshold_) {
        const double excess = h_abs - curve_speed_heading_threshold_;
        const double ratio =
            std::max(curve_speed_min_ratio_, 1.0 - curve_speed_reduction_gain_ * excess);
        v_target *= ratio;
      }
    }

    if (clear_path_mode && path_hold_enable_ && !return_recovery_latched_) {
      const bool path_hold_active =
          (std::abs(cte_ctrl) > path_hold_recovery_distance_) ||
          (std::abs(heading_ctrl) > curve_speed_heading_threshold_);
      if (path_hold_active) {
        double w_heading = path_hold_heading_error_gain_ * heading_ctrl;
        double w_cte = -path_hold_cross_track_gain_ * cte_ctrl *
                       std::max(0.3, std::abs(v_target));
        w_heading = std::clamp(w_heading, -path_hold_max_omega_, path_hold_max_omega_);
        w_cte = std::clamp(w_cte, -path_hold_max_omega_, path_hold_max_omega_);
        double w_geo = w_heading + w_cte;
        if (std::abs(omega_target) > 1e-3 && (w_geo * omega_target) < 0.0) {
          w_geo *= 0.35;
        }
        w_geo = std::clamp(w_geo, -path_hold_max_omega_, path_hold_max_omega_);
        omega_target =
            (1.0 - path_hold_recovery_blend_) * omega_target + path_hold_recovery_blend_ * w_geo;
      }
    }

    double omega_abs_limit = max_omega_abs_;
    double omega_rate_limit = max_omega_rate_;
    const bool obstacle_near = obs_ctrl.detected && obs_ctrl.s_rear < obstacle_near_release_s_;
    if (obstacle_near) {
      omega_abs_limit = std::max(omega_abs_limit, obstacle_max_omega_abs_);
      omega_rate_limit = std::max(omega_rate_limit, obstacle_max_omega_rate_);
    }

    const bool clear_mode =
        (!obs_ctrl.detected) && (!post_obstacle_mode) && (!recent_obstacle) &&
        (!return_recovery_latched_);
    if (clear_mode && no_obstacle_speed_guard_enable_) {
      const double err = no_obstacle_speed_cte_gain_ * std::abs(cte_ctrl) +
                         no_obstacle_speed_heading_gain_ * std::abs(heading_ctrl);
      const double ratio = std::clamp(1.0 - err, no_obstacle_speed_min_ratio_, 1.0);
      v_target = std::min(v_target, cfg_.max_velocity * ratio);
    }

    if (clear_mode) {
      if (std::abs(cte_ctrl) > overshoot_guard_distance_ && (omega_target * cte_ctrl) > 0.0) {
        omega_target *= overshoot_reverse_damping_;
      }
      if (oscillation_guard_enable_ && has_prev_errors_) {
        const bool cte_flip =
            (std::abs(cte_ctrl) > oscillation_guard_cte_deadband_) &&
            (std::abs(prev_signed_cte_) > oscillation_guard_cte_deadband_) &&
            ((cte_ctrl * prev_signed_cte_) < 0.0);
        const bool heading_flip =
            (std::abs(heading_ctrl) > oscillation_guard_heading_deadband_) &&
            (std::abs(prev_heading_error_) > oscillation_guard_heading_deadband_) &&
            ((heading_ctrl * prev_heading_error_) < 0.0);
        if (cte_flip || heading_flip) {
          omega_target *= oscillation_guard_reverse_damping_;
        }
      }
      if (adaptive_corner_mode_enable_) {
        const bool near_path = (std::abs(cte_ctrl) < adaptive_near_cte_thresh_) &&
                               (std::abs(heading_ctrl) < adaptive_near_heading_thresh_);
        if (near_path) {
          omega_target *= adaptive_near_omega_damping_;
          omega_rate_limit *= adaptive_near_omega_rate_scale_;
          v_target *= adaptive_near_v_scale_;
        }
      }
      prev_signed_cte_ = cte_ctrl;
      prev_heading_error_ = heading_ctrl;
      has_prev_errors_ = true;
    } else {
      has_prev_errors_ = false;
    }

    if (std::abs(omega_target) > omega_abs_limit && std::abs(omega_target) > 1e-6) {
      const double scale = omega_abs_limit / std::abs(omega_target);
      v_target *= scale;
      omega_target *= scale;
    }

    const double v_cmd = applySpeedRateLimit(v_target, dt);
    const double v_sig = applyVelocitySigmoid(v_cmd, dt);
    omega_target = std::clamp(omega_target, -omega_abs_limit, omega_abs_limit);
    const double w_cmd = applyOmegaRateLimit(omega_target, dt, omega_abs_limit, omega_rate_limit);
    publishCmd(v_sig, w_cmd);

    if (obs_ctrl.detected) {
      last_obs_ = obs_ctrl;
      last_obs_time_ = now;
      has_last_obs_ = true;
    } else if (has_last_obs_) {
      const double age = (now - last_obs_time_).seconds();
      if (age > obstacle_memory_sec_ + 0.5) has_last_obs_ = false;
    }
  }

  double applySpeedRateLimit(double v_target, double dt) {
    if (!speed_initialized_) {
      prev_v_cmd_ = std::max(0.0, v_target);
      speed_initialized_ = true;
      return prev_v_cmd_;
    }
    const double min_v = prev_v_cmd_ - max_v_rate_down_ * dt;
    const double max_v = prev_v_cmd_ + max_v_rate_up_ * dt;
    prev_v_cmd_ = std::clamp(v_target, min_v, max_v);
    if (prev_v_cmd_ < 0.0) prev_v_cmd_ = 0.0;
    return prev_v_cmd_;
  }

  double applyOmegaRateLimit(double w_target, double dt, double omega_abs_limit,
                             double omega_rate_limit) {
    if (!omega_initialized_) {
      prev_omega_cmd_ = std::clamp(w_target, -omega_abs_limit, omega_abs_limit);
      omega_initialized_ = true;
      return prev_omega_cmd_;
    }
    const double max_dw = omega_rate_limit * std::max(1e-3, dt);
    const double min_w = prev_omega_cmd_ - max_dw;
    const double max_w = prev_omega_cmd_ + max_dw;
    prev_omega_cmd_ = std::clamp(w_target, min_w, max_w);
    prev_omega_cmd_ = std::clamp(prev_omega_cmd_, -omega_abs_limit, omega_abs_limit);
    return prev_omega_cmd_;
  }

  double applyVelocitySigmoid(double v_target, double dt) {
    const double v_clamped = std::max(0.0, v_target);
    if (!v_sig_initialized_) {
      v_sig_ = v_clamped;
      v_sig_initialized_ = true;
      return v_sig_;
    }
    const double tau = (v_clamped > v_sig_) ? sigmoid_tau_accel_ : sigmoid_tau_decel_;
    const double alpha = 1.0 - std::exp(-std::max(1e-3, dt) / std::max(1e-3, tau));
    v_sig_ += alpha * (v_clamped - v_sig_);
    if (v_sig_ < 0.0) v_sig_ = 0.0;
    return v_sig_;
  }

  bool computeNearPathMetrics(const geometry_msgs::msg::Pose& ego_pose,
                              const std::vector<geometry_msgs::msg::PoseStamped>& path,
                              double& signed_cte, double& heading_error) const {
    signed_cte = 0.0;
    heading_error = 0.0;
    const int n = static_cast<int>(path.size());
    if (n < 2) return false;

    const double ex = ego_pose.position.x;
    const double ey = ego_pose.position.y;
    int closest_idx = 0;
    double best_d2 = std::numeric_limits<double>::max();
    const int search_max = std::min(n, 80);
    for (int i = 0; i < search_max; ++i) {
      const double dx = path[i].pose.position.x - ex;
      const double dy = path[i].pose.position.y - ey;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        closest_idx = i;
      }
    }

    int i0 = closest_idx;
    int i1 = std::min(closest_idx + 1, n - 1);
    if (i1 == i0 && i0 > 0) i0 = i0 - 1;
    if (i1 == i0) return false;

    const auto& p0 = path[i0].pose.position;
    const auto& p1 = path[i1].pose.position;
    const double tx = p1.x - p0.x;
    const double ty = p1.y - p0.y;
    const double tnorm = std::hypot(tx, ty);
    if (tnorm < 1e-6) return false;

    const double path_yaw = std::atan2(ty, tx);
    const double yaw = quatToYaw(ego_pose.orientation);
    const double rx = ex - p0.x;
    const double ry = ey - p0.y;
    signed_cte = -std::sin(path_yaw) * rx + std::cos(path_yaw) * ry;
    heading_error = wrapAngle(path_yaw - yaw);
    return std::isfinite(signed_cte) && std::isfinite(heading_error);
  }

  ObstacleInfo projectObstacle(const std::vector<geometry_msgs::msg::PoseStamped>& path,
                               const geometry_msgs::msg::Pose& ego_pose,
                               const geometry_msgs::msg::Pose& obs_pose) const {
    ObstacleInfo info;
    if (path.empty()) return info;
    const int n = static_cast<int>(path.size());
    const double ox = obs_pose.position.x;
    const double oy = obs_pose.position.y;
    const double ex = ego_pose.position.x;
    const double ey = ego_pose.position.y;

    int ego_idx = 0;
    double best_ego = std::numeric_limits<double>::max();
    const int ego_search_max = std::min(n, 80);
    for (int i = 0; i < ego_search_max; ++i) {
      const double dx = path[i].pose.position.x - ex;
      const double dy = path[i].pose.position.y - ey;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_ego) {
        best_ego = d2;
        ego_idx = i;
      }
    }

    int obs_idx = ego_idx;
    double best_obs = std::numeric_limits<double>::max();
    double s_scan = 0.0;
    for (int i = ego_idx; i < n; ++i) {
      const double dx = path[i].pose.position.x - ox;
      const double dy = path[i].pose.position.y - oy;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_obs) {
        best_obs = d2;
        obs_idx = i;
      }
      if (i + 1 < n) {
        const double dsx = path[i + 1].pose.position.x - path[i].pose.position.x;
        const double dsy = path[i + 1].pose.position.y - path[i].pose.position.y;
        s_scan += std::hypot(dsx, dsy);
      }
      if (s_scan > detect_range_s_ * 1.5) break;
    }

    if (obs_idx <= ego_idx) return info;

    double s_along = 0.0;
    for (int i = ego_idx; i < obs_idx && i + 1 < n; ++i) {
      const double dx = path[i + 1].pose.position.x - path[i].pose.position.x;
      const double dy = path[i + 1].pose.position.y - path[i].pose.position.y;
      s_along += std::hypot(dx, dy);
    }
    if (s_along > detect_range_s_ + obstacle_body_length_) return info;

    const int i0 = obs_idx;
    const int i1 = (obs_idx + 1 < n) ? obs_idx + 1 : obs_idx - 1;
    if (i1 < 0 || i1 >= n || i1 == i0) return info;

    const double tx = path[i1].pose.position.x - path[i0].pose.position.x;
    const double ty = path[i1].pose.position.y - path[i0].pose.position.y;
    const double tnorm = std::hypot(tx, ty);
    if (tnorm < 1e-6) return info;

    const double tnx = tx / tnorm;
    const double tny = ty / tnorm;
    const double vx = ox - path[i0].pose.position.x;
    const double vy = oy - path[i0].pose.position.y;
    const double d_center = -tny * vx + tnx * vy;

    const double path_yaw = std::atan2(ty, tx);
    const double obs_yaw = quatToYaw(obs_pose.orientation);
    const double delta = wrapAngle(obs_yaw - path_yaw);
    const double half_l = 0.5 * obstacle_body_length_;
    const double half_w = 0.5 * obstacle_body_width_;

    const double s_extent =
        std::abs(std::cos(delta)) * half_l + std::abs(std::sin(delta)) * half_w +
        obstacle_body_s_margin_;
    const double d_extent =
        std::abs(std::sin(delta)) * half_l + std::abs(std::cos(delta)) * half_w;

    const double s_rear = s_along - s_extent;
    const double s_front = s_along + s_extent;
    if (s_front < -obstacle_clear_back_s_) return info;
    if (s_rear > detect_range_s_) return info;

    info.detected = true;
    info.d_obs = d_center;
    info.s_obs = s_along;
    info.d_min = d_center - d_extent;
    info.d_max = d_center + d_extent;
    info.s_rear = s_rear;
    info.s_front = s_front;
    info.d_extent = d_extent;
    info.s_extent = s_extent;
    return info;
  }

  void buildAsymmetricLateralConstraints(
      const Eigen::VectorXd& x0, const Eigen::VectorXd& z_bar, const Eigen::MatrixXd& A_bar,
      const Eigen::MatrixXd& B_bar, const Eigen::MatrixXd& E_bar, const Eigen::MatrixXd& C_bar,
      const std::vector<double>& d_upper_seq, const std::vector<double>& d_lower_seq,
      Eigen::SparseMatrix<double>& A_out, Eigen::VectorXd& l_out, Eigen::VectorXd& u_out,
      int nvar) const {
    const int N = cfg_.N;
    const int n_lat_step = 3;
    const int n_lat = n_lat_step * N;
    const int rows = 2 * n_lat + 2;
    constexpr double kInf = 1e20;

    A_out.resize(rows, nvar);
    l_out = Eigen::VectorXd::Constant(rows, -kInf);
    u_out = Eigen::VectorXd::Constant(rows, kInf);

    Eigen::MatrixXd C_d = Eigen::MatrixXd::Zero(n_lat, kLTVNx * N);
    for (int k = 0; k < N; ++k) {
      for (int i = 0; i < n_lat_step; ++i) {
        C_d.row(k * n_lat_step + i) = C_bar.row(k * kLTVNy + i);
      }
    }

    const Eigen::VectorXd x_free = A_bar * x0 + E_bar * z_bar;
    const Eigen::VectorXd y_free = C_d * x_free;
    const Eigen::MatrixXd Cy = C_d * B_bar;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(2 * n_lat * (N + 1) + 2));

    for (int r = 0; r < n_lat; ++r) {
      for (int c = 0; c < N; ++c) {
        const double v = Cy(r, c);
        if (std::abs(v) > 1e-12) trips.emplace_back(r, c, v);
      }
      trips.emplace_back(r, N, -1.0);
      const int k = std::min(N - 1, r / n_lat_step);
      u_out(r) = d_upper_seq[k] - y_free(r);
    }

    for (int r = 0; r < n_lat; ++r) {
      const int rr = n_lat + r;
      for (int c = 0; c < N; ++c) {
        const double v = -Cy(r, c);
        if (std::abs(v) > 1e-12) trips.emplace_back(rr, c, v);
      }
      trips.emplace_back(rr, N + 1, -1.0);
      const int k = std::min(N - 1, r / n_lat_step);
      u_out(rr) = y_free(r) - d_lower_seq[k];
    }

    const int reu = 2 * n_lat;
    const int rel = reu + 1;
    trips.emplace_back(reu, N, 1.0);
    trips.emplace_back(rel, N + 1, 1.0);
    l_out(reu) = 0.0;
    l_out(rel) = 0.0;

    A_out.setFromTriplets(trips.begin(), trips.end());
  }

  void buildInputConstraints(Eigen::SparseMatrix<double>& A_out, Eigen::VectorXd& l_out,
                             Eigen::VectorXd& u_out, int nvar) const {
    const int N = cfg_.N;
    A_out.resize(N, nvar);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(N);
    for (int i = 0; i < N; ++i) trips.emplace_back(i, i, 1.0);
    A_out.setFromTriplets(trips.begin(), trips.end());
    l_out = Eigen::VectorXd::Constant(N, cfg_.u_min);
    u_out = Eigen::VectorXd::Constant(N, cfg_.u_max);
  }

  void buildKappaConstraints(const Eigen::VectorXd& x0, const Eigen::VectorXd& z_bar,
                             const Eigen::MatrixXd& A_bar, const Eigen::MatrixXd& B_bar,
                             const Eigen::MatrixXd& E_bar, Eigen::SparseMatrix<double>& A_out,
                             Eigen::VectorXd& l_out, Eigen::VectorXd& u_out, int nvar) const {
    const int N = cfg_.N;
    A_out.resize(N, nvar);
    l_out.resize(N);
    u_out.resize(N);
    const Eigen::VectorXd x_free = A_bar * x0 + E_bar * z_bar;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(N * N / 2);
    for (int k = 0; k < N; ++k) {
      const int kr = k * kLTVNx + 2;
      const double kf = x_free(kr);
      for (int j = 0; j <= k; ++j) {
        const double v = B_bar(kr, j);
        if (std::abs(v) > 1e-12) trips.emplace_back(k, j, v);
      }
      l_out(k) = cfg_.kappa_min - kf;
      u_out(k) = cfg_.kappa_max - kf;
    }
    A_out.setFromTriplets(trips.begin(), trips.end());
  }

  static void appendSparse(std::vector<Eigen::Triplet<double>>& trips,
                           const Eigen::SparseMatrix<double>& m, int row_offset) {
    for (int k = 0; k < m.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(m, k); it; ++it) {
        trips.emplace_back(it.row() + row_offset, it.col(), it.value());
      }
    }
  }

  MPCCommand computeWithObstacle(const geometry_msgs::msg::Pose& ego_pose,
                                 const std::vector<geometry_msgs::msg::PoseStamped>& local_path,
                                 double d_upper, double d_lower,
                                 const ObstacleInfo& obs, bool post_obstacle_mode) {
    MPCCommand out;
    if (local_path.size() < 5) return out;

    Eigen::VectorXd x0, z_bar, v_bar;
    std::vector<double> theta_r_seq;
    std::vector<double> kappa_r_seq;
    model_.buildReferenceProfiles(ego_pose, local_path, kappa_state_, x0, z_bar, v_bar,
                                  theta_r_seq, kappa_r_seq);
    if (x0.size() != kLTVNx || z_bar.size() != cfg_.N || v_bar.size() != cfg_.N) return out;
    last_dr_ = x0(0);
    last_heading_err_ = wrapAngle(x0(1) - x0(3));

    if (!kappa_initialized_ && !kappa_r_seq.empty()) {
      kappa_state_ = std::clamp(kappa_r_seq[0], cfg_.kappa_min, cfg_.kappa_max);
      kappa_initialized_ = true;
    }

    Eigen::MatrixXd A_bar, B_bar, E_bar, C_bar;
    model_.buildBatchDynamics(v_bar, A_bar, B_bar, E_bar, C_bar);

    Eigen::MatrixXd Q_bar, R_bar;
    cost_.buildStateInputWeights(Q_bar, R_bar);
    Eigen::SparseMatrix<double> P_base;
    Eigen::VectorXd q_base;
    cost_.buildQPObjective(x0, z_bar, A_bar, B_bar, E_bar, Q_bar, R_bar, P_base, q_base);

    const int nvar = cfg_.N + 2;
    Eigen::SparseMatrix<double> P(nvar, nvar);
    std::vector<Eigen::Triplet<double>> p_trips;
    p_trips.reserve(static_cast<size_t>(P_base.nonZeros() + 2));
    for (int k = 0; k < P_base.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(P_base, k); it; ++it) {
        p_trips.emplace_back(it.row(), it.col(), it.value());
      }
    }
    p_trips.emplace_back(cfg_.N, cfg_.N, 2.0 * cfg_.w_lateral_slack_quad);
    p_trips.emplace_back(cfg_.N + 1, cfg_.N + 1, 2.0 * cfg_.w_lateral_slack_quad);
    P.setFromTriplets(p_trips.begin(), p_trips.end());

    Eigen::VectorXd q = Eigen::VectorXd::Zero(nvar);
    q.head(cfg_.N) = q_base;
    q(cfg_.N) = cfg_.w_lateral_slack_lin;
    q(cfg_.N + 1) = cfg_.w_lateral_slack_lin;

    std::vector<double> d_upper_seq(cfg_.N, d_upper);
    std::vector<double> d_lower_seq(cfg_.N, d_lower);
    if (obs.detected) {
      const double sigma = std::max(0.2, obstacle_sigma_s_);
      const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
      double s_pred = 0.0;
      for (int k = 0; k < cfg_.N; ++k) {
        if (k > 0) {
          const double vk = (k - 1 < v_bar.size()) ? v_bar(k - 1) : v_bar(v_bar.size() - 1);
          s_pred += std::max(0.05, vk) * cfg_.Ts;
        }

        const double ds_back = s_pred - obs.s_rear;
        const double ds_front = obs.s_front - s_pred;
        const double in_from_back =
            sigmoid((ds_back + obstacle_activation_back_s_) / obstacle_activation_ramp_s_);
        const double in_to_front =
            sigmoid((ds_front + obstacle_activation_front_s_) / obstacle_activation_ramp_s_);
        const double gate = in_from_back * in_to_front;

        double ds_interval = 0.0;
        if (s_pred < obs.s_rear) {
          ds_interval = obs.s_rear - s_pred;
        } else if (s_pred > obs.s_front) {
          ds_interval = s_pred - obs.s_front;
        }

        const double w = gate * std::exp(-ds_interval * ds_interval * inv_2sigma2);
        const double d_safe_eff = d_safe_ + obstacle_extra_push_ * gate;

        if (obs.d_min >= 0.0) {
          const double target_upper = std::min(d_upper, obs.d_min - d_safe_eff);
          d_upper_seq[k] = (1.0 - w) * d_upper + w * target_upper;
        } else if (obs.d_max <= 0.0) {
          const double target_lower = std::max(d_lower, obs.d_max + d_safe_eff);
          d_lower_seq[k] = (1.0 - w) * d_lower + w * target_lower;
        } else {
          const double right_limit = obs.d_min - d_safe_eff;
          const double left_limit = obs.d_max + d_safe_eff;
          const bool choose_right = chooseRightByFreeSpace(obs, d_upper, d_lower, d_safe_eff);
          if (choose_right) {
            const double target_upper = std::min(d_upper, right_limit);
            d_upper_seq[k] = (1.0 - w) * d_upper + w * target_upper;
          } else {
            const double target_lower = std::max(d_lower, left_limit);
            d_lower_seq[k] = (1.0 - w) * d_lower + w * target_lower;
          }
        }

        if (d_upper_seq[k] < d_lower_seq[k] + 0.03) {
          const double mid = 0.5 * (d_upper_seq[k] + d_lower_seq[k]);
          d_upper_seq[k] = mid + 0.015;
          d_lower_seq[k] = mid - 0.015;
        }
      }
    }

    Eigen::SparseMatrix<double> A_in, A_kap, A_lat;
    Eigen::VectorXd l_in, l_kap, l_lat;
    Eigen::VectorXd u_in, u_kap, u_lat;
    buildInputConstraints(A_in, l_in, u_in, nvar);
    buildKappaConstraints(x0, z_bar, A_bar, B_bar, E_bar, A_kap, l_kap, u_kap, nvar);
    buildAsymmetricLateralConstraints(x0, z_bar, A_bar, B_bar, E_bar, C_bar, d_upper_seq,
                                      d_lower_seq, A_lat, l_lat, u_lat, nvar);

    const int r_in = A_in.rows();
    const int r_kap = A_kap.rows();
    const int r_lat = A_lat.rows();

    Eigen::SparseMatrix<double> A_cons(r_in + r_kap + r_lat, nvar);
    std::vector<Eigen::Triplet<double>> a_trips;
    a_trips.reserve(static_cast<size_t>(A_in.nonZeros() + A_kap.nonZeros() + A_lat.nonZeros()));
    appendSparse(a_trips, A_in, 0);
    appendSparse(a_trips, A_kap, r_in);
    appendSparse(a_trips, A_lat, r_in + r_kap);
    A_cons.setFromTriplets(a_trips.begin(), a_trips.end());

    Eigen::VectorXd l_cons(r_in + r_kap + r_lat);
    Eigen::VectorXd u_cons(r_in + r_kap + r_lat);
    l_cons << l_in, l_kap, l_lat;
    u_cons << u_in, u_kap, u_lat;

    Eigen::VectorXd u_opt;
    if (!solver_.solve(P, q, A_cons, l_cons, u_cons, u_opt) || u_opt.size() < cfg_.N) return out;

    const double u0 = u_opt(0);
    kappa_state_ += cfg_.Ts * u0;
    if (!kappa_r_seq.empty()) {
      const double kappa_ref = std::clamp(kappa_r_seq[0], cfg_.kappa_min, cfg_.kappa_max);
      double alpha_ref = obs.detected ? kappa_blend_alpha_obstacle_ : cfg_.kappa_blend_alpha;
      if (post_obstacle_mode) {
        alpha_ref = std::max(alpha_ref, post_obstacle_kappa_blend_alpha_);
      }
      const double alpha = std::clamp(alpha_ref, 0.0, 1.0);
      kappa_state_ = (1.0 - alpha) * kappa_state_ + alpha * kappa_ref;
    }
    kappa_state_ = std::clamp(kappa_state_, cfg_.kappa_min, cfg_.kappa_max);

    const double v_cmd = std::clamp(v_bar(0), cfg_.min_velocity, cfg_.max_velocity);
    out.v_cmd = v_cmd;
    out.kappa_cmd = kappa_state_;
    out.omega_cmd = v_cmd * kappa_state_;
    out.solved = true;
    out.u_seq.resize(cfg_.N);
    for (int i = 0; i < cfg_.N; ++i) out.u_seq[i] = u_opt(i);
    return out;
  }

  void publishCmd(double v, double w) const {
    geometry_msgs::msg::Accel msg;
    msg.linear.x = std::max(0.0, v);
    msg.angular.z = w;
    cav01_pub_->publish(msg);
  }

  void publishZero() const {
    geometry_msgs::msg::Accel msg;
    msg.linear.x = 0.0;
    msg.angular.z = 0.0;
    for (const auto& pub : zero_pub_) pub->publish(msg);
  }

  LTVMPCConfig cfg_{};
  LTVModel model_;
  LTVCost cost_;
  LTVSolver solver_;

  double kappa_state_{0.0};
  bool kappa_initialized_{false};
  double kappa_blend_alpha_obstacle_{0.02};
  double max_omega_abs_{1.6};
  double max_omega_rate_{12.0};
  double obstacle_max_omega_abs_{1.7};
  double obstacle_max_omega_rate_{14.0};

  double d_safe_{0.40};
  double detect_range_s_{2.2};
  double obstacle_sigma_s_{0.50};
  double obstacle_body_length_{0.46};
  double obstacle_body_width_{0.28};
  double obstacle_body_s_margin_{0.08};
  double obstacle_center_deadband_{0.12};
  double obstacle_preferred_sign_{1.0};
  double obstacle_near_slowdown_s_{0.70};
  double obstacle_near_release_s_{1.30};
  double obstacle_near_speed_{0.10};
  double obstacle_activation_back_s_{0.30};
  double obstacle_activation_front_s_{0.95};
  double obstacle_activation_ramp_s_{0.16};
  double obstacle_extra_push_{0.18};
  double obstacle_memory_sec_{0.90};
  double max_v_rate_up_{0.45};
  double max_v_rate_down_{0.90};
  double obstacle_clear_back_s_{0.10};
  bool obstacle_turn_boost_enable_{true};
  double obstacle_turn_boost_front_s_{0.95};
  double obstacle_turn_boost_max_s_{0.25};
  double obstacle_turn_boost_lateral_gate_{0.28};
  double obstacle_turn_boost_omega_{1.25};
  double obstacle_turn_boost_min_omega_{0.50};
  double obstacle_turn_boost_blend_{0.78};
  bool no_obstacle_speed_guard_enable_{true};
  double no_obstacle_speed_cte_gain_{2.0};
  double no_obstacle_speed_heading_gain_{0.7};
  double no_obstacle_speed_min_ratio_{0.30};
  bool curve_speed_enable_{true};
  double curve_speed_heading_threshold_{0.16};
  double curve_speed_reduction_gain_{0.04};
  double curve_speed_min_ratio_{0.96};
  bool path_hold_enable_{true};
  double path_hold_max_omega_{1.40};
  double path_hold_recovery_distance_{0.15};
  double path_hold_heading_error_gain_{0.80};
  double path_hold_cross_track_gain_{0.70};
  double path_hold_recovery_blend_{0.45};
  double overshoot_guard_distance_{0.13};
  double overshoot_reverse_damping_{0.52};
  bool oscillation_guard_enable_{true};
  double oscillation_guard_cte_deadband_{0.08};
  double oscillation_guard_heading_deadband_{0.20};
  double oscillation_guard_reverse_damping_{0.70};
  bool adaptive_corner_mode_enable_{true};
  double adaptive_near_cte_thresh_{0.14};
  double adaptive_near_heading_thresh_{0.14};
  double adaptive_near_omega_damping_{0.55};
  double adaptive_near_omega_rate_scale_{0.55};
  double adaptive_near_v_scale_{0.97};
  double sigmoid_tau_accel_{0.40};
  double sigmoid_tau_decel_{0.20};
  bool return_recovery_enable_{true};
  double return_recovery_enter_cte_{0.12};
  double return_recovery_exit_cte_{0.04};
  double return_recovery_speed_{0.16};
  double return_recovery_heading_gain_{1.2};
  double return_recovery_cte_gain_{2.2};
  double return_recovery_max_omega_{1.6};
  double return_recovery_blend_{0.65};
  double return_recovery_post_obs_sec_{1.4};
  double post_obstacle_recenter_sec_{2.2};
  double post_obstacle_kappa_blend_alpha_{0.40};
  double post_obstacle_max_speed_{0.20};

  bool slow_mode_latched_{false};
  bool return_recovery_latched_{false};
  bool has_last_obs_{false};
  bool prev_obs_detected_{false};
  ObstacleInfo last_obs_{};
  rclcpp::Time last_obs_time_;
  rclcpp::Time post_obstacle_until_;
  double last_dr_{0.0};
  double last_heading_err_{0.0};
  bool has_prev_errors_{false};
  double prev_signed_cte_{0.0};
  double prev_heading_error_{0.0};

  bool speed_initialized_{false};
  double prev_v_cmd_{0.0};
  bool v_sig_initialized_{false};
  double v_sig_{0.0};
  bool omega_initialized_{false};
  double prev_omega_cmd_{0.0};
  rclcpp::Time prev_cmd_time_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ego_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr obs_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr cav01_pub_;
  std::array<rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr, 3> zero_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::optional<geometry_msgs::msg::Pose> ego_pose_;
  std::vector<geometry_msgs::msg::PoseStamped> local_path_;
  std::optional<geometry_msgs::msg::PoseStamped> obs_pose_;
};

}  // namespace bisa

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bisa::MpcObstacleAvoidanceCpp>());
  rclcpp::shutdown();
  return 0;
}
