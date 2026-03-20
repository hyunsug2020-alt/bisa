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
#include <std_msgs/msg/bool.hpp>

#include "ltv_mpc/ltv_cost.hpp"
#include "ltv_mpc/ltv_model.hpp"
#include "ltv_mpc/ltv_obstacle.hpp"
#include "ltv_mpc/ltv_solver.hpp"
#include "ltv_mpc/ltv_types.hpp"

namespace bisa {

class LtvObstacleAvoidanceNode : public rclcpp::Node {
 public:
  LtvObstacleAvoidanceNode()
      : Node("ltv_obstacle_avoidance_node"),
        model_(mpc_cfg_),
        cost_(mpc_cfg_),
        solver_(mpc_cfg_),
        obstacle_(obs_cfg_) {
    declare_parameter("horizon", 25);
    declare_parameter("time_step", 0.05);
    declare_parameter("wheelbase", 0.30);
    declare_parameter("max_velocity", 0.72);
    declare_parameter("min_velocity", 0.30);
    declare_parameter("Q_pos", 20.0);
    declare_parameter("Q_heading", 8.0);
    declare_parameter("weight_curvature", 1.0);
    declare_parameter("weight_input", 1.5);
    declare_parameter("kappa_min", -2.0);
    declare_parameter("kappa_max", 2.0);
    declare_parameter("u_min", -2.0);
    declare_parameter("u_max", 2.0);
    declare_parameter("max_omega_abs", 1.70);
    declare_parameter("max_omega_rate", 14.0);
    declare_parameter("kappa_blend_alpha", 0.15);
    declare_parameter("kappa_blend_alpha_obs", 0.02);
    declare_parameter("ref_preview_steps", 0);
    declare_parameter("lateral_bound", 0.30);
    declare_parameter("w_lateral_slack_lin", 500.0);
    declare_parameter("w_lateral_slack_quad", 5000.0);
    declare_parameter("osqp_max_iter", 4000);
    declare_parameter("osqp_eps_abs", 1e-4);
    declare_parameter("osqp_eps_rel", 1e-4);
    declare_parameter("osqp_polish", false);

    declare_parameter("d_safe", 0.35);
    declare_parameter("detect_range_s", 2.5);
    declare_parameter("obs_body_length", 0.46);
    declare_parameter("obs_body_width", 0.28);
    declare_parameter("obs_s_margin", 0.08);
    declare_parameter("activation_sigma_s", 0.50);
    declare_parameter("activation_back_s", 0.30);
    declare_parameter("activation_front_s", 0.95);
    declare_parameter("activation_ramp_s", 0.16);
    declare_parameter("extra_push", 0.15);
    declare_parameter("road_d_upper", 0.50);
    declare_parameter("road_d_lower", -0.50);
    declare_parameter("near_slowdown_s", 0.70);
    declare_parameter("near_release_s", 1.30);
    declare_parameter("near_speed", 0.10);

    declare_parameter("sigmoid_tau_accel", 0.40);
    declare_parameter("sigmoid_tau_decel", 0.20);
    declare_parameter("max_v_rate_up", 0.45);
    declare_parameter("max_v_rate_down", 0.90);

    declare_parameter("post_obs_recenter_sec", 2.2);
    declare_parameter("post_obs_kappa_blend", 0.40);
    declare_parameter("post_obs_max_speed", 0.20);
    declare_parameter("return_enter_cte", 0.10);
    declare_parameter("return_exit_cte", 0.03);
    declare_parameter("return_speed", 0.16);
    declare_parameter("return_heading_gain", 1.2);
    declare_parameter("return_cte_gain", 2.2);
    declare_parameter("return_max_omega", 1.6);
    declare_parameter("return_blend", 0.65);

    declare_parameter("oscillation_guard_enable", true);
    declare_parameter("oscillation_cte_deadband", 0.06);
    declare_parameter("oscillation_heading_deadband", 0.15);
    declare_parameter("oscillation_damping", 0.70);
    declare_parameter("overshoot_guard_distance", 0.10);
    declare_parameter("overshoot_damping", 0.50);

    loadConfig();

    const auto sensor_qos = rclcpp::SensorDataQoS();
    const auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    const auto status_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    ego_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/CAV_01", sensor_qos,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          ego_pose_ = msg->pose;
        });

    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/cav01/local_path", rclcpp::QoS(10),
        [this](const nav_msgs::msg::Path::SharedPtr msg) {
          local_path_ = msg->poses;
        });

    obs_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/CAV_02", sensor_qos,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          obs_pose_ = msg->pose;
        });

    cav01_pub_ =
        create_publisher<geometry_msgs::msg::Accel>("/sim/cav01/accel", cmd_qos);
    zero_pub_[0] =
        create_publisher<geometry_msgs::msg::Accel>("/sim/cav02/accel", cmd_qos);
    zero_pub_[1] =
        create_publisher<geometry_msgs::msg::Accel>("/sim/cav03/accel", cmd_qos);
    zero_pub_[2] =
        create_publisher<geometry_msgs::msg::Accel>("/sim/cav04/accel", cmd_qos);
    avoid_active_pub_ = create_publisher<std_msgs::msg::Bool>(
        "/cav01/obstacle_avoidance_active", status_qos);

    prev_cmd_time_ = this->now();
    post_obs_until_ = this->now();
    timer_ = create_wall_timer(
        std::chrono::milliseconds(50),
        std::bind(&LtvObstacleAvoidanceNode::controlLoop, this));

    RCLCPP_INFO(get_logger(),
                "LTV Obstacle Avoidance ready: CAV1 control, avoiding CAV2");
    RCLCPP_INFO(get_logger(), "  N=%d, Ts=%.3f, d_safe=%.2f, detect=%.1f m",
                mpc_cfg_.N, mpc_cfg_.Ts, obs_cfg_.d_safe,
                obs_cfg_.detect_range_s);
  }

 private:
  void loadConfig() {
    const auto rd = [this](const char* n) { return get_parameter(n).as_double(); };

    mpc_cfg_.N = std::max(5, static_cast<int>(get_parameter("horizon").as_int()));
    mpc_cfg_.Ts = std::max(1e-3, rd("time_step"));
    mpc_cfg_.wheelbase = std::max(0.1, rd("wheelbase"));
    mpc_cfg_.max_velocity = std::max(0.0, rd("max_velocity"));
    mpc_cfg_.min_velocity = std::max(0.0, rd("min_velocity"));
    mpc_cfg_.w_d = rd("Q_pos");
    mpc_cfg_.w_theta = rd("Q_heading");
    mpc_cfg_.w_kappa = rd("weight_curvature");
    mpc_cfg_.w_u = rd("weight_input");
    mpc_cfg_.kappa_min = rd("kappa_min");
    mpc_cfg_.kappa_max = rd("kappa_max");
    mpc_cfg_.u_min = rd("u_min");
    mpc_cfg_.u_max = rd("u_max");
    mpc_cfg_.ref_preview_steps =
        std::max(0, static_cast<int>(get_parameter("ref_preview_steps").as_int()));
    mpc_cfg_.lateral_bound = std::max(0.05, rd("lateral_bound"));
    mpc_cfg_.w_lateral_slack_lin = std::max(0.0, rd("w_lateral_slack_lin"));
    mpc_cfg_.w_lateral_slack_quad = std::max(0.0, rd("w_lateral_slack_quad"));
    mpc_cfg_.osqp_max_iter =
        std::max(100, static_cast<int>(get_parameter("osqp_max_iter").as_int()));
    mpc_cfg_.osqp_eps_abs = std::max(1e-7, rd("osqp_eps_abs"));
    mpc_cfg_.osqp_eps_rel = std::max(1e-7, rd("osqp_eps_rel"));
    mpc_cfg_.osqp_polish = get_parameter("osqp_polish").as_bool();

    max_omega_abs_ = std::max(0.1, rd("max_omega_abs"));
    max_omega_rate_ = std::max(0.1, rd("max_omega_rate"));
    kappa_blend_alpha_ = std::clamp(rd("kappa_blend_alpha"), 0.0, 1.0);
    kappa_blend_alpha_obs_ = std::clamp(rd("kappa_blend_alpha_obs"), 0.0, 1.0);
    sigmoid_tau_accel_ = std::max(0.01, rd("sigmoid_tau_accel"));
    sigmoid_tau_decel_ = std::max(0.01, rd("sigmoid_tau_decel"));
    max_v_rate_up_ = std::max(0.05, rd("max_v_rate_up"));
    max_v_rate_down_ = std::max(0.05, rd("max_v_rate_down"));

    obs_cfg_.d_safe = std::max(0.05, rd("d_safe"));
    obs_cfg_.detect_range_s = std::max(0.3, rd("detect_range_s"));
    obs_cfg_.obs_body_length = std::max(0.10, rd("obs_body_length"));
    obs_cfg_.obs_body_width = std::max(0.10, rd("obs_body_width"));
    obs_cfg_.obs_s_margin = std::max(0.0, rd("obs_s_margin"));
    obs_cfg_.activation_sigma_s = std::max(0.2, rd("activation_sigma_s"));
    obs_cfg_.activation_back_s = std::max(0.05, rd("activation_back_s"));
    obs_cfg_.activation_front_s = std::max(0.10, rd("activation_front_s"));
    obs_cfg_.activation_ramp_s = std::max(0.05, rd("activation_ramp_s"));
    obs_cfg_.extra_push = std::max(0.0, rd("extra_push"));
    obs_cfg_.road_d_upper = rd("road_d_upper");
    obs_cfg_.road_d_lower = rd("road_d_lower");
    obs_cfg_.near_slowdown_s = std::max(0.1, rd("near_slowdown_s"));
    obs_cfg_.near_release_s =
        std::max(obs_cfg_.near_slowdown_s + 0.05, rd("near_release_s"));
    obs_cfg_.near_speed = std::max(0.03, rd("near_speed"));

    post_obs_recenter_sec_ = std::max(0.2, rd("post_obs_recenter_sec"));
    post_obs_kappa_blend_ = std::clamp(rd("post_obs_kappa_blend"), 0.0, 1.0);
    post_obs_max_speed_ = std::max(0.05, rd("post_obs_max_speed"));
    return_enter_cte_ = std::max(0.02, rd("return_enter_cte"));
    return_exit_cte_ =
        std::clamp(rd("return_exit_cte"), 0.01, return_enter_cte_ - 0.01);
    return_speed_ = std::max(0.03, rd("return_speed"));
    return_heading_gain_ = std::max(0.0, rd("return_heading_gain"));
    return_cte_gain_ = std::max(0.0, rd("return_cte_gain"));
    return_max_omega_ = std::max(0.2, rd("return_max_omega"));
    return_blend_ = std::clamp(rd("return_blend"), 0.0, 1.0);

    oscillation_guard_enable_ =
        get_parameter("oscillation_guard_enable").as_bool();
    oscillation_cte_deadband_ = std::max(0.0, rd("oscillation_cte_deadband"));
    oscillation_heading_deadband_ =
        std::max(0.0, rd("oscillation_heading_deadband"));
    oscillation_damping_ = std::clamp(rd("oscillation_damping"), 0.1, 1.0);
    overshoot_guard_distance_ = std::max(0.01, rd("overshoot_guard_distance"));
    overshoot_damping_ = std::clamp(rd("overshoot_damping"), 0.1, 1.0);

    model_.setConfig(mpc_cfg_);
    cost_.setConfig(mpc_cfg_);
    solver_.setConfig(mpc_cfg_);
    obstacle_.setConfig(obs_cfg_);
  }

  bool computeNearPathMetrics(double& signed_cte, double& heading_error) const {
    signed_cte = 0.0;
    heading_error = 0.0;
    if (!ego_pose_.has_value() || local_path_.size() < 2) return false;

    const double ex = ego_pose_->position.x;
    const double ey = ego_pose_->position.y;
    int closest_idx = 0;
    double best_d2 = std::numeric_limits<double>::max();
    const int search_max = std::min(static_cast<int>(local_path_.size()), 80);
    for (int i = 0; i < search_max; ++i) {
      const double dx = local_path_[i].pose.position.x - ex;
      const double dy = local_path_[i].pose.position.y - ey;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        closest_idx = i;
      }
    }

    int i0 = closest_idx;
    int i1 = std::min(closest_idx + 1, static_cast<int>(local_path_.size()) - 1);
    if (i1 == i0 && i0 > 0) i0 -= 1;
    if (i1 == i0) return false;

    const auto& p0 = local_path_[i0].pose.position;
    const auto& p1 = local_path_[i1].pose.position;
    const double tnorm = std::hypot(p1.x - p0.x, p1.y - p0.y);
    if (tnorm < 1e-6) return false;

    const double path_yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);
    const double ego_yaw = quatToYaw(ego_pose_->orientation);
    const double rx = ex - p0.x;
    const double ry = ey - p0.y;
    signed_cte = -std::sin(path_yaw) * rx + std::cos(path_yaw) * ry;
    heading_error = wrapAngle(path_yaw - ego_yaw);
    return std::isfinite(signed_cte) && std::isfinite(heading_error);
  }

  static double quatToYaw(const geometry_msgs::msg::Quaternion& q) {
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const bool valid = std::isfinite(n) && std::abs(n - 1.0) <= 0.15 &&
                       std::abs(q.x) <= 1.0 + 1e-3 &&
                       std::abs(q.y) <= 1.0 + 1e-3 &&
                       std::abs(q.z) <= 1.0 + 1e-3 &&
                       std::abs(q.w) <= 1.0 + 1e-3;
    if (!valid) return wrapAngle(q.z);
    const double x = q.x / n;
    const double y = q.y / n;
    const double z = q.z / n;
    const double w = q.w / n;
    return wrapAngle(std::atan2(2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z)));
  }

  static double wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  void publishZero() const {
    geometry_msgs::msg::Accel z;
    z.linear.x = 0.0;
    z.angular.z = 0.0;
    for (const auto& pub : zero_pub_) pub->publish(z);
  }

  void publishAvoidanceStatus(bool active) const {
    std_msgs::msg::Bool msg;
    msg.data = active;
    avoid_active_pub_->publish(msg);
  }

  void publishCmd(double v, double w) const {
    geometry_msgs::msg::Accel msg;
    msg.linear.x = std::max(0.0, v);
    msg.angular.z = w;
    cav01_pub_->publish(msg);
  }

  double applySpeedRateLimit(double v_target, double dt) {
    if (!speed_init_) {
      prev_v_ = std::max(0.0, v_target);
      speed_init_ = true;
      return prev_v_;
    }
    prev_v_ = std::clamp(v_target, prev_v_ - max_v_rate_down_ * dt,
                         prev_v_ + max_v_rate_up_ * dt);
    if (prev_v_ < 0.0) prev_v_ = 0.0;
    return prev_v_;
  }

  double applyVelocitySigmoid(double v_target, double dt) {
    const double v_c = std::max(0.0, v_target);
    if (!v_sig_init_) {
      v_sig_ = v_c;
      v_sig_init_ = true;
      return v_sig_;
    }
    const double tau = (v_c > v_sig_) ? sigmoid_tau_accel_ : sigmoid_tau_decel_;
    const double alpha =
        1.0 - std::exp(-std::max(1e-3, dt) / std::max(1e-3, tau));
    v_sig_ += alpha * (v_c - v_sig_);
    if (v_sig_ < 0.0) v_sig_ = 0.0;
    return v_sig_;
  }

  double applyOmegaRateLimit(double w_target, double dt) {
    if (!omega_init_) {
      prev_w_ = std::clamp(w_target, -max_omega_abs_, max_omega_abs_);
      omega_init_ = true;
      return prev_w_;
    }
    const double max_dw = max_omega_rate_ * std::max(1e-3, dt);
    prev_w_ = std::clamp(w_target, prev_w_ - max_dw, prev_w_ + max_dw);
    prev_w_ = std::clamp(prev_w_, -max_omega_abs_, max_omega_abs_);
    return prev_w_;
  }

  void buildInputConstraints(Eigen::SparseMatrix<double>& A,
                             Eigen::VectorXd& l,
                             Eigen::VectorXd& u,
                             int nvar) const {
    const int N = mpc_cfg_.N;
    A.resize(N, nvar);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(N);
    for (int i = 0; i < N; ++i) trips.emplace_back(i, i, 1.0);
    A.setFromTriplets(trips.begin(), trips.end());
    l = Eigen::VectorXd::Constant(N, mpc_cfg_.u_min);
    u = Eigen::VectorXd::Constant(N, mpc_cfg_.u_max);
  }

  void buildKappaConstraints(const Eigen::VectorXd& x0,
                             const Eigen::VectorXd& z_bar,
                             const Eigen::MatrixXd& A_bar,
                             const Eigen::MatrixXd& B_bar,
                             const Eigen::MatrixXd& E_bar,
                             Eigen::SparseMatrix<double>& A,
                             Eigen::VectorXd& l,
                             Eigen::VectorXd& u,
                             int nvar) const {
    const int N = mpc_cfg_.N;
    A.resize(N, nvar);
    l.resize(N);
    u.resize(N);
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
      l(k) = mpc_cfg_.kappa_min - kf;
      u(k) = mpc_cfg_.kappa_max - kf;
    }
    A.setFromTriplets(trips.begin(), trips.end());
  }

  static void appendSparse(std::vector<Eigen::Triplet<double>>& trips,
                           const Eigen::SparseMatrix<double>& m,
                           int row_offset) {
    for (int k = 0; k < m.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(m, k); it; ++it) {
        trips.emplace_back(it.row() + row_offset, it.col(), it.value());
      }
    }
  }

  void controlLoop() {
    publishZero();

    const auto now = this->now();
    double dt = (now - prev_cmd_time_).seconds();
    if (!std::isfinite(dt) || dt <= 1e-4) dt = 0.05;
    prev_cmd_time_ = now;

    if (!ego_pose_.has_value() || local_path_.size() < 5) {
      publishAvoidanceStatus(false);
      publishCmd(applySpeedRateLimit(0.0, dt), 0.0);
      return;
    }

    const int N = mpc_cfg_.N;
    Eigen::VectorXd x0, z_bar, v_bar;
    std::vector<double> theta_r_seq, kappa_r_seq;
    model_.buildReferenceProfiles(ego_pose_.value(), local_path_, kappa_state_, x0,
                                  z_bar, v_bar, theta_r_seq, kappa_r_seq);

    if (x0.size() != kLTVNx || z_bar.size() != N || v_bar.size() != N) {
      publishAvoidanceStatus(false);
      publishCmd(applySpeedRateLimit(0.0, dt), 0.0);
      return;
    }

    if (!kappa_init_ && !kappa_r_seq.empty()) {
      kappa_state_ =
          std::clamp(kappa_r_seq[0], mpc_cfg_.kappa_min, mpc_cfg_.kappa_max);
      kappa_init_ = true;
    }

    FrenetObstacle obs;
    if (obs_pose_.has_value()) {
      obs =
          obstacle_.projectToFrenet(ego_pose_.value(), obs_pose_.value(), local_path_);
    }

    if (obs.detected != last_obs_logged_) {
      if (obs.detected) {
        RCLCPP_WARN(get_logger(),
                    "[DETECT] ON: s=%.2f d=%.3f d_range=[%.3f,%.3f]",
                    obs.s_obs, obs.d_obs, obs.d_min, obs.d_max);
      } else {
        RCLCPP_INFO(get_logger(), "[DETECT] OFF: obstacle cleared");
      }
      last_obs_logged_ = obs.detected;
    }

    std::vector<double> d_upper_seq, d_lower_seq;
    obstacle_.computeTimeVaryingBounds(obs, v_bar, mpc_cfg_.Ts, N, d_upper_seq,
                                       d_lower_seq);

    Eigen::MatrixXd A_bar, B_bar, E_bar, C_bar;
    model_.buildBatchDynamics(v_bar, A_bar, B_bar, E_bar, C_bar);

    Eigen::MatrixXd Q_bar, R_bar;
    cost_.buildStateInputWeights(Q_bar, R_bar);

    Eigen::SparseMatrix<double> P_base;
    Eigen::VectorXd q_base;
    cost_.buildQPObjective(x0, z_bar, A_bar, B_bar, E_bar, Q_bar, R_bar, P_base,
                           q_base);

    const int nvar = N + 2;
    Eigen::SparseMatrix<double> P(nvar, nvar);
    {
      std::vector<Eigen::Triplet<double>> p_trips;
      p_trips.reserve(static_cast<size_t>(P_base.nonZeros() + 2));
      for (int k = 0; k < P_base.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(P_base, k); it; ++it) {
          p_trips.emplace_back(it.row(), it.col(), it.value());
        }
      }
      p_trips.emplace_back(N, N, 2.0 * mpc_cfg_.w_lateral_slack_quad);
      p_trips.emplace_back(N + 1, N + 1, 2.0 * mpc_cfg_.w_lateral_slack_quad);
      P.setFromTriplets(p_trips.begin(), p_trips.end());
    }

    Eigen::VectorXd q = Eigen::VectorXd::Zero(nvar);
    q.head(N) = q_base;
    q(N) = mpc_cfg_.w_lateral_slack_lin;
    q(N + 1) = mpc_cfg_.w_lateral_slack_lin;

    Eigen::SparseMatrix<double> A_in, A_kap, A_lat;
    Eigen::VectorXd l_in, u_in, l_kap, u_kap, l_lat, u_lat;

    buildInputConstraints(A_in, l_in, u_in, nvar);
    buildKappaConstraints(x0, z_bar, A_bar, B_bar, E_bar, A_kap, l_kap, u_kap,
                          nvar);
    obstacle_.buildAsymmetricLateralConstraints(
        x0, z_bar, A_bar, B_bar, E_bar, C_bar, d_upper_seq, d_lower_seq,
        mpc_cfg_.w_lateral_slack_lin, mpc_cfg_.w_lateral_slack_quad, A_lat,
        l_lat, u_lat, nvar);

    const int r_in = A_in.rows();
    const int r_kap = A_kap.rows();
    const int r_lat = A_lat.rows();
    const int total_rows = r_in + r_kap + r_lat;
    Eigen::SparseMatrix<double> A_cons(total_rows, nvar);
    {
      std::vector<Eigen::Triplet<double>> trips;
      trips.reserve(static_cast<size_t>(
          A_in.nonZeros() + A_kap.nonZeros() + A_lat.nonZeros()));
      appendSparse(trips, A_in, 0);
      appendSparse(trips, A_kap, r_in);
      appendSparse(trips, A_lat, r_in + r_kap);
      A_cons.setFromTriplets(trips.begin(), trips.end());
    }

    Eigen::VectorXd l_cons(total_rows), u_cons(total_rows);
    l_cons << l_in, l_kap, l_lat;
    u_cons << u_in, u_kap, u_lat;

    Eigen::VectorXd u_opt;
    const bool solved = solver_.solve(P, q, A_cons, l_cons, u_cons, u_opt);
    if (!solved || u_opt.size() < N) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "[MPC] solve failed, decelerating safely");
      slow_latched_ = true;
      publishAvoidanceStatus(obs.detected || slow_latched_ || prev_obs_detected_ ||
                             return_latched_);
      publishCmd(applySpeedRateLimit(0.0, dt), 0.0);
      return;
    }

    const double u0 = u_opt(0);
    kappa_state_ += mpc_cfg_.Ts * u0;

    if (prev_obs_detected_ && !obs.detected) {
      post_obs_until_ =
          now + rclcpp::Duration::from_seconds(post_obs_recenter_sec_);
    }
    const bool post_obs_mode =
        (!obs.detected) && ((post_obs_until_ - now).seconds() > 0.0);
    prev_obs_detected_ = obs.detected;

    kappa_state_ =
        std::clamp(kappa_state_, mpc_cfg_.kappa_min, mpc_cfg_.kappa_max);

    double v_cmd =
        std::clamp(v_bar(0), mpc_cfg_.min_velocity, mpc_cfg_.max_velocity);
    double omega_cmd = v_cmd * kappa_state_;

    double signed_cte = 0.0;
    double heading_error = 0.0;
    const bool metrics_valid = computeNearPathMetrics(signed_cte, heading_error);

    if (!obs.detected) {
      slow_latched_ = false;
    } else if (obs.s_rear < obs_cfg_.near_slowdown_s) {
      slow_latched_ = true;
    } else if (slow_latched_ && obs.s_rear > obs_cfg_.near_release_s) {
      slow_latched_ = false;
    }
    if (slow_latched_) {
      v_cmd = std::min(v_cmd, obs_cfg_.near_speed);
    }

    if (post_obs_mode) {
      v_cmd = std::min(v_cmd, post_obs_max_speed_);
    }

    if (metrics_valid && !obs.detected && post_obs_mode) {
      if (!return_latched_ && std::abs(signed_cte) > return_enter_cte_) {
        return_latched_ = true;
      } else if (return_latched_ && std::abs(signed_cte) < return_exit_cte_) {
        return_latched_ = false;
      }
    } else {
      return_latched_ = false;
    }

    if (return_latched_ && metrics_valid) {
      const double w_heading = return_heading_gain_ * heading_error;
      const double w_cte = -return_cte_gain_ * signed_cte;
      double w_recover = std::clamp(w_heading + w_cte, -return_max_omega_,
                                    return_max_omega_);

      if ((w_recover * signed_cte) > 0.0) {
        w_recover *= 0.7;
      }

      const double cte_norm = std::clamp(
          std::abs(signed_cte) / std::max(return_enter_cte_, 1e-3), 0.0, 1.0);
      const double blend = std::clamp(
          0.35 + (return_blend_ - 0.35) * cte_norm, 0.2, return_blend_);
      omega_cmd = (1.0 - blend) * omega_cmd + blend * w_recover;
      v_cmd = std::min(v_cmd, return_speed_);

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                           "[RETURN] cte=%.3f heading=%.3f blend=%.2f w_out=%.3f",
                           signed_cte, heading_error, blend, omega_cmd);
    }

    if (metrics_valid && !obs.detected && !return_latched_) {
      if (std::abs(signed_cte) > overshoot_guard_distance_ &&
          (omega_cmd * signed_cte) > 0.0) {
        omega_cmd *= overshoot_damping_;
      }
    }

    if (oscillation_guard_enable_ && metrics_valid && !obs.detected) {
      if (has_prev_errors_) {
        const bool cte_flip =
            (std::abs(signed_cte) > oscillation_cte_deadband_) &&
            (std::abs(prev_cte_) > oscillation_cte_deadband_) &&
            ((signed_cte * prev_cte_) < 0.0);
        const bool heading_flip =
            (std::abs(heading_error) > oscillation_heading_deadband_) &&
            (std::abs(prev_heading_) > oscillation_heading_deadband_) &&
            ((heading_error * prev_heading_) < 0.0);
        if (cte_flip || heading_flip) {
          omega_cmd *= oscillation_damping_;
        }
      }
      prev_cte_ = signed_cte;
      prev_heading_ = heading_error;
      has_prev_errors_ = true;
    }

    const bool avoid_active =
        obs.detected || slow_latched_ || post_obs_mode || return_latched_;
    v_cmd = applySpeedRateLimit(v_cmd, dt);
    v_cmd = applyVelocitySigmoid(v_cmd, dt);
    omega_cmd = std::clamp(omega_cmd, -max_omega_abs_, max_omega_abs_);
    omega_cmd = applyOmegaRateLimit(omega_cmd, dt);

    publishAvoidanceStatus(avoid_active);
    publishCmd(v_cmd, omega_cmd);

    const char* phase = obs.detected ? "AVOID" : (post_obs_mode ? "RETURN" : "NORMAL");
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 500,
        "[%s] obs=%d s=[%.2f,%.2f] d=[%.3f,%.3f] cte=%.3f | v=%.3f w=%.3f kappa=%.4f",
        phase, obs.detected ? 1 : 0, obs.s_rear, obs.s_front, obs.d_min,
        obs.d_max, signed_cte, v_cmd, omega_cmd, kappa_state_);
  }

  LTVMPCConfig mpc_cfg_{};
  ObstacleAvoidanceConfig obs_cfg_{};
  double max_omega_abs_{1.70};
  double max_omega_rate_{14.0};
  double kappa_blend_alpha_{0.15};
  double kappa_blend_alpha_obs_{0.02};
  double sigmoid_tau_accel_{0.40};
  double sigmoid_tau_decel_{0.20};
  double max_v_rate_up_{0.45};
  double max_v_rate_down_{0.90};

  double post_obs_recenter_sec_{2.2};
  double post_obs_kappa_blend_{0.40};
  double post_obs_max_speed_{0.20};
  double return_enter_cte_{0.10};
  double return_exit_cte_{0.03};
  double return_speed_{0.16};
  double return_heading_gain_{1.2};
  double return_cte_gain_{2.2};
  double return_max_omega_{1.6};
  double return_blend_{0.65};

  bool oscillation_guard_enable_{true};
  double oscillation_cte_deadband_{0.06};
  double oscillation_heading_deadband_{0.15};
  double oscillation_damping_{0.70};
  double overshoot_guard_distance_{0.10};
  double overshoot_damping_{0.50};

  LTVModel model_;
  LTVCost cost_;
  LTVSolver solver_;
  LTVObstacle obstacle_;

  double kappa_state_{0.0};
  bool kappa_init_{false};
  bool slow_latched_{false};
  bool prev_obs_detected_{false};
  bool last_obs_logged_{false};
  bool return_latched_{false};
  rclcpp::Time post_obs_until_;

  bool has_prev_errors_{false};
  double prev_cte_{0.0};
  double prev_heading_{0.0};

  bool speed_init_{false};
  bool v_sig_init_{false};
  bool omega_init_{false};
  double prev_v_{0.0};
  double v_sig_{0.0};
  double prev_w_{0.0};
  rclcpp::Time prev_cmd_time_;

  std::optional<geometry_msgs::msg::Pose> ego_pose_;
  std::optional<geometry_msgs::msg::Pose> obs_pose_;
  std::vector<geometry_msgs::msg::PoseStamped> local_path_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ego_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr obs_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr cav01_pub_;
  std::array<rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr, 3>
      zero_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr avoid_active_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace bisa

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bisa::LtvObstacleAvoidanceNode>());
  rclcpp::shutdown();
  return 0;
}
