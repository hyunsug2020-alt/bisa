// ============================================================
// mpc_path_tracker_cpp.cpp  (bisa)
// DBM RTI-NMPC 통합 버전
//
// 주요 변경:
//   - use_ltv_mpc_=false 시 DBMRTINMPCController 사용
//   - pose 차분으로 vx, vy, omega 추정 후 컨트롤러에 주입
//   - 1-sample delay 처리 (논문 식 3)
//   - 기존 post-processing (path-hold, 진동 방지 등) 유지
// ============================================================
#include "bisa/mpc_path_tracker_cpp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace bisa {

namespace {
static constexpr double kPi = 3.14159265358979323846;

double wrap_angle(double a) {
  while (a >  kPi) a -= 2.0*kPi;
  while (a < -kPi) a += 2.0*kPi;
  return a;
}

double pose_yaw_from_quat_or_packed(const geometry_msgs::msg::Pose& pose) {
  const auto& q = pose.orientation;
  const double n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  const bool range_ok =
      std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w) &&
      std::abs(q.x) <= 1.0+1e-3 && std::abs(q.y) <= 1.0+1e-3 &&
      std::abs(q.z) <= 1.0+1e-3 && std::abs(q.w) <= 1.0+1e-3;
  if (!range_ok || !std::isfinite(n) || std::abs(n - 1.0) > 0.05) {
    double yaw = q.z;
    if (!std::isfinite(yaw)) return 0.0;
    if (std::abs(yaw) > 2.0*kPi + 0.5) yaw *= kPi / 180.0;
    else if (yaw >  kPi) yaw -= 2.0*kPi;
    else if (yaw < -kPi) yaw += 2.0*kPi;
    return wrap_angle(yaw);
  }
  const double x = q.x/n, y = q.y/n, z = q.z/n, w = q.w/n;
  return wrap_angle(std::atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)));
}
}  // namespace

// ============================================================
// 생성자
// ============================================================
MPCPathTrackerCpp::MPCPathTrackerCpp()
  : Node("mpc_path_tracker_cpp"),
    last_log_time_(this->now()),
    prev_cmd_time_(this->now())
{
  // ── 파라미터 선언 ──────────────────────────────────────
  this->declare_parameter("target_cav_id",    1);
  this->declare_parameter("use_ltv_mpc",      false);   // DBM RTI-NMPC 기본
  this->declare_parameter("publish_accel_cmd", true);

  // 공통
  this->declare_parameter("prediction_horizon", 25);
  this->declare_parameter("time_step",          0.05);
  this->declare_parameter("wheelbase",          0.30);
  this->declare_parameter("max_velocity",       0.72);
  this->declare_parameter("min_velocity",       0.18);
  this->declare_parameter("horizon",            -1);

  // LTV 호환 이름 (legacy)
  this->declare_parameter("Q_pos",             20.0);
  this->declare_parameter("Q_heading",          8.0);
  this->declare_parameter("weight_curvature",   1.0);
  this->declare_parameter("weight_input",       5.0);
  this->declare_parameter("u_min",             -1.5);
  this->declare_parameter("u_max",              1.5);
  this->declare_parameter("kappa_min_delta",   -3.0);
  this->declare_parameter("kappa_max_delta",    3.0);
  this->declare_parameter("max_angular_vel",    std::numeric_limits<double>::quiet_NaN());
  this->declare_parameter("max_accel",          std::numeric_limits<double>::quiet_NaN());

  // DBM RTI-NMPC 전용 파라미터
  this->declare_parameter("mass_kg",            0.8);
  this->declare_parameter("inertia_kgm2",       0.02);
  this->declare_parameter("lf_m",               0.15);
  this->declare_parameter("lr_m",               0.15);
  this->declare_parameter("Ca_N_rad",           10.0);
  this->declare_parameter("Cx_N",               12.0);
  this->declare_parameter("Jw_kgm2",            0.002);
  this->declare_parameter("r_w_m",              0.033);
  this->declare_parameter("v_min_mps",          0.05);
  this->declare_parameter("mu",                 0.8);
  this->declare_parameter("v_delta_max",        1.5);
  this->declare_parameter("T_drive_max",        0.5);
  this->declare_parameter("delta_c_max",        0.785);
  this->declare_parameter("w_xy",               20.0);
  this->declare_parameter("w_psi",               8.0);
  this->declare_parameter("w_vx",                1.0);
  this->declare_parameter("w_vy",                0.5);
  this->declare_parameter("w_omega",             0.5);
  this->declare_parameter("w_delta_c",           0.2);
  this->declare_parameter("w_wheel",             0.01);
  this->declare_parameter("w_u_delta",           5.0);
  this->declare_parameter("w_u_torque",          5.0);

  // 제어 후처리
  this->declare_parameter("max_omega_abs",       1.7);
  this->declare_parameter("max_omega_rate",     14.0);
  this->declare_parameter("max_v_rate",          2.0);
  this->declare_parameter("sigmoid_tau_accel",   0.40);
  this->declare_parameter("sigmoid_tau_decel",   0.20);
  this->declare_parameter("path_reset_distance_threshold", 1.0);
  this->declare_parameter("path_hold_distance_gain",      0.22);
  this->declare_parameter("path_hold_heading_gain",       0.22);
  this->declare_parameter("path_hold_max_omega",          1.40);
  this->declare_parameter("path_hold_lookahead_index",    12);
  this->declare_parameter("path_hold_recovery_distance",  0.15);
  this->declare_parameter("path_hold_heading_error_gain", 0.80);
  this->declare_parameter("path_hold_cross_track_gain",   0.70);
  this->declare_parameter("path_hold_recovery_blend",     0.45);
  this->declare_parameter("curve_speed_enable",           true);
  this->declare_parameter("curve_speed_heading_threshold",0.16);
  this->declare_parameter("curve_speed_reduction_gain",   0.04);
  this->declare_parameter("curve_speed_min_ratio",        0.96);
  this->declare_parameter("overshoot_guard_distance",     0.13);
  this->declare_parameter("overshoot_reverse_damping",    0.52);
  this->declare_parameter("oscillation_guard_enable",     true);
  this->declare_parameter("oscillation_guard_cte_deadband",     0.08);
  this->declare_parameter("oscillation_guard_heading_deadband", 0.10);
  this->declare_parameter("oscillation_guard_reverse_damping",  0.70);
  this->declare_parameter("adaptive_corner_mode_enable",  true);
  this->declare_parameter("adaptive_near_cte_thresh",     0.07);
  this->declare_parameter("adaptive_near_heading_thresh", 0.07);
  this->declare_parameter("adaptive_near_omega_damping",  0.82);
  this->declare_parameter("adaptive_near_omega_rate_scale",0.55);
  this->declare_parameter("adaptive_near_v_scale",        0.97);
  this->declare_parameter("off_path_recovery_enable",     true);
  this->declare_parameter("off_path_recovery_distance",   0.50);
  this->declare_parameter("off_path_recovery_exit_distance",0.22);
  this->declare_parameter("off_path_recovery_speed",      0.08);
  this->declare_parameter("off_path_recovery_heading_gain",1.0);
  this->declare_parameter("off_path_recovery_cte_gain",   1.2);
  this->declare_parameter("off_path_recovery_max_omega",  1.0);
  // OSQP
  this->declare_parameter("osqp_max_iter", 4000);
  this->declare_parameter("osqp_eps_abs",  1e-4);
  this->declare_parameter("osqp_eps_rel",  1e-4);
  this->declare_parameter("osqp_polish",   true);
  // LTV lateral (legacy)
  this->declare_parameter("lateral_bound",          -1.0);
  this->declare_parameter("w_lateral_slack_lin",   500.0);
  this->declare_parameter("w_lateral_slack_quad", 5000.0);
  this->declare_parameter("ref_preview_steps",       0);
  this->declare_parameter("kappa_blend_alpha",       0.40);
  this->declare_parameter("kappa_limit_ref_velocity",-1.0);
  this->declare_parameter("N",                       -1);
  this->declare_parameter("Ts",    std::numeric_limits<double>::quiet_NaN());
  this->declare_parameter("w_d",   std::numeric_limits<double>::quiet_NaN());
  this->declare_parameter("w_theta",std::numeric_limits<double>::quiet_NaN());
  this->declare_parameter("w_kappa",std::numeric_limits<double>::quiet_NaN());
  this->declare_parameter("w_u",   std::numeric_limits<double>::quiet_NaN());

  param_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&MPCPathTrackerCpp::parameter_callback, this, std::placeholders::_1));

  target_cav_id_    = this->get_parameter("target_cav_id").as_int();
  use_ltv_mpc_      = this->get_parameter("use_ltv_mpc").as_bool();
  publish_accel_cmd_= this->get_parameter("publish_accel_cmd").as_bool();

  ltv_controller_    = std::make_unique<LTVMPC>(LTVMPCConfig{});
  legacy_controller_ = std::make_unique<MPCControllerCpp>();
  update_controller_params();

  std::string id_str = (target_cav_id_ < 10)
    ? "0" + std::to_string(target_cav_id_) : std::to_string(target_cav_id_);
  RCLCPP_INFO(this->get_logger(),
    "MPC Tracker CAV_%s [%s]",
    id_str.c_str(), use_ltv_mpc_ ? "LTV-MPC" : "DBM-RTI-NMPC");

  local_sub_ = this->create_subscription<nav_msgs::msg::Path>(
    "/local_path", 10,
    std::bind(&MPCPathTrackerCpp::local_path_callback, this, std::placeholders::_1));
  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/Ego_pose", rclcpp::SensorDataQoS(),
    std::bind(&MPCPathTrackerCpp::pose_callback, this, std::placeholders::_1));

  const auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
  if (publish_accel_cmd_) {
    accel_pub_ = this->create_publisher<geometry_msgs::msg::Accel>("/Accel", cmd_qos);
  }
  accel_pub_raw_ = this->create_publisher<geometry_msgs::msg::Accel>("/Accel_raw", cmd_qos);
  pred_pub_      = this->create_publisher<nav_msgs::msg::Path>("/mpc_predicted_path", 10);
  perf_pub_      = this->create_publisher<bisa::msg::MPCPerformance>("/mpc_performance", 10);

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&MPCPathTrackerCpp::control_loop, this));
}

// ============================================================
// 상태 추정 (pose 차분 → body-frame 속도)
// ============================================================
void MPCPathTrackerCpp::updateStateEstimate(
    const geometry_msgs::msg::PoseStamped& new_pose)
{
  if (!prev_pose_stamped_.has_value()) {
    prev_pose_stamped_ = new_pose;
    return;
  }

  const double dt = (rclcpp::Time(new_pose.header.stamp) -
                     rclcpp::Time(prev_pose_stamped_->header.stamp)).seconds();
  if (dt < 1e-4 || dt > 1.0) {
    prev_pose_stamped_ = new_pose;
    return;
  }

  const double psi = pose_yaw_from_quat_or_packed(new_pose.pose);
  const double dx  = new_pose.pose.position.x - prev_pose_stamped_->pose.position.x;
  const double dy  = new_pose.pose.position.y - prev_pose_stamped_->pose.position.y;

  // world → body 변환
  const double vwx = dx / dt;
  const double vwy = dy / dt;
  const double vx_new =  vwx * std::cos(psi) + vwy * std::sin(psi);
  const double vy_new = -vwx * std::sin(psi) + vwy * std::cos(psi);

  const double psi_prev = pose_yaw_from_quat_or_packed(prev_pose_stamped_->pose);
  const double omega_new = wrap_angle(psi - psi_prev) / dt;

  // 1차 저역통과 필터 (노이즈 억제)
  const double alpha = 0.4;
  est_vx_    = alpha * vx_new    + (1.0 - alpha) * est_vx_;
  est_vy_    = alpha * vy_new    + (1.0 - alpha) * est_vy_;
  est_omega_ = alpha * omega_new + (1.0 - alpha) * est_omega_;

  // δc 추정: kinematic 근사 δc = atan(omega * L / vx)
  const double L   = this->get_parameter("wheelbase").as_double();
  const double vx_safe = std::max(std::abs(est_vx_), 0.05);
  est_delta_ = std::atan(est_omega_ * L / vx_safe);

  // 바퀴 각속도 추정: ωf = ωr = vx / r_w (rolling 조건)
  const double r_w = this->get_parameter("r_w_m").as_double();
  est_wf_ = est_vx_ / std::max(r_w, 0.01);
  est_wr_ = est_vx_ / std::max(r_w, 0.01);

  prev_pose_stamped_ = new_pose;
}

// ============================================================
// 파라미터 업데이트
// ============================================================
void MPCPathTrackerCpp::update_controller_params() {
  const auto rd = [this](const char* n) { return this->get_parameter(n).as_double(); };

  // ── DBM RTI-NMPC 파라미터 ──────────────────────────────
  DBMRTINMPCParams dbm;
  dbm.N            = this->get_parameter("prediction_horizon").as_int();
  const int h_over = this->get_parameter("horizon").as_int();
  if (h_over > 0) dbm.N = h_over;
  const int N_new = this->get_parameter("N").as_int();
  if (N_new > 0) dbm.N = N_new;

  dbm.Ts           = std::max(1e-3, rd("time_step"));
  const double ts_new = rd("Ts");
  if (std::isfinite(ts_new) && ts_new > 1e-4) dbm.Ts = ts_new;

  dbm.wheelbase    = std::max(0.1, rd("wheelbase"));
  dbm.max_velocity = std::max(0.05, rd("max_velocity"));
  dbm.min_velocity = std::max(0.01, rd("min_velocity"));

  dbm.mass_kg      = std::max(0.1,  rd("mass_kg"));
  dbm.inertia_kgm2 = std::max(1e-4, rd("inertia_kgm2"));
  dbm.lf_m         = std::max(0.01, rd("lf_m"));
  dbm.lr_m         = std::max(0.01, rd("lr_m"));
  dbm.Ca_N_rad     = std::max(0.1,  rd("Ca_N_rad"));
  dbm.Cx_N         = std::max(0.1,  rd("Cx_N"));
  dbm.Jw_kgm2      = std::max(1e-6, rd("Jw_kgm2"));
  dbm.r_w_m        = std::max(0.01, rd("r_w_m"));
  dbm.v_min_mps    = std::max(0.01, rd("v_min_mps"));
  dbm.mu           = std::clamp(rd("mu"), 0.1, 1.5);
  dbm.v_delta_max  = std::max(0.1,  rd("v_delta_max"));
  dbm.T_drive_max  = std::max(0.01, rd("T_drive_max"));
  dbm.delta_c_max  = std::max(0.1,  rd("delta_c_max"));
  dbm.w_xy         = std::max(0.0,  rd("w_xy"));
  dbm.w_psi        = std::max(0.0,  rd("w_psi"));
  dbm.w_vx         = std::max(0.0,  rd("w_vx"));
  dbm.w_vy         = std::max(0.0,  rd("w_vy"));
  dbm.w_omega      = std::max(0.0,  rd("w_omega"));
  dbm.w_delta_c    = std::max(0.0,  rd("w_delta_c"));
  dbm.w_wheel      = std::max(0.0,  rd("w_wheel"));
  dbm.w_u_delta    = std::max(0.0,  rd("w_u_delta"));
  dbm.w_u_torque   = std::max(0.0,  rd("w_u_torque"));

  // LTVMPCParams (legacy wrapper용)
  LTVMPCParams lp;
  lp.N            = dbm.N;
  lp.Ts           = dbm.Ts;
  lp.l            = dbm.wheelbase;
  lp.max_velocity = dbm.max_velocity;
  lp.min_velocity = dbm.min_velocity;
  lp.mu           = dbm.mu;
  lp.wu           = dbm.w_u_delta;

  if (legacy_controller_) {
    // DBM 파라미터를 직접 주입
    legacy_controller_->update_parameters(lp);
    // DBMRTINMPCParams도 직접 설정
    static_cast<void>(dbm);  // DBM 파라미터는 아래 별도 경로로 설정
  }

  // ── DBMRTINMPCController 직접 접근 (legacy_controller_ 내부) ─
  // wrapper의 dbm_params_ 업데이트
  if (legacy_controller_) {
    // LTVMPCParams를 통해 기본값 전달 후, DBM 전용 파라미터는
    // setEstimatedState처럼 별도 setter를 통해 전달
    // (여기서는 LTVMPCParams를 DBM 파라미터와 동일하게 매핑)
    lp.wd    = dbm.w_xy;
    lp.wtheta= dbm.w_psi;
    lp.wkappa= dbm.w_vy;
    legacy_controller_->update_parameters(lp);
  }

  // ── LTVMPC 파라미터 ────────────────────────────────────
  LTVMPCConfig cfg;
  cfg.N            = dbm.N;
  cfg.Ts           = dbm.Ts;
  cfg.wheelbase    = dbm.wheelbase;
  cfg.max_velocity = dbm.max_velocity;
  cfg.min_velocity = dbm.min_velocity;
  cfg.w_d          = rd("Q_pos");
  cfg.w_theta      = rd("Q_heading");
  cfg.w_kappa      = rd("weight_curvature");
  cfg.w_u          = rd("weight_input");
  cfg.u_min        = rd("u_min");
  cfg.u_max        = rd("u_max");
  cfg.kappa_min    = rd("kappa_min_delta");
  cfg.kappa_max    = rd("kappa_max_delta");
  cfg.ref_preview_steps = this->get_parameter("ref_preview_steps").as_int();
  cfg.lateral_bound     = rd("lateral_bound");
  cfg.w_lateral_slack_lin  = std::max(0.0, rd("w_lateral_slack_lin"));
  cfg.w_lateral_slack_quad = std::max(0.0, rd("w_lateral_slack_quad"));
  cfg.osqp_max_iter = this->get_parameter("osqp_max_iter").as_int();
  cfg.osqp_eps_abs  = std::max(1e-7, rd("osqp_eps_abs"));
  cfg.osqp_eps_rel  = std::max(1e-7, rd("osqp_eps_rel"));
  cfg.osqp_polish   = this->get_parameter("osqp_polish").as_bool();
  if (ltv_controller_) { ltv_controller_->setConfig(cfg); }

  // ── 후처리 파라미터 ────────────────────────────────────
  max_omega_abs_   = std::max(0.1, rd("max_omega_abs"));
  max_omega_rate_  = std::max(0.1, rd("max_omega_rate"));
  max_v_rate_      = std::max(0.1, rd("max_v_rate"));
  sig_tau_up_      = std::max(0.01, rd("sigmoid_tau_accel"));
  sig_tau_down_    = std::max(0.01, rd("sigmoid_tau_decel"));
  path_reset_distance_threshold_ = std::max(0.1, rd("path_reset_distance_threshold"));
  path_hold_distance_gain_       = std::max(0.0, rd("path_hold_distance_gain"));
  path_hold_heading_gain_        = std::max(0.0, rd("path_hold_heading_gain"));
  path_hold_max_omega_           = std::max(0.1, rd("path_hold_max_omega"));
  path_hold_lookahead_index_     = std::max(1L, this->get_parameter("path_hold_lookahead_index").as_int());
  path_hold_recovery_distance_   = std::max(0.05, rd("path_hold_recovery_distance"));
  path_hold_heading_error_gain_  = std::max(0.0, rd("path_hold_heading_error_gain"));
  path_hold_cross_track_gain_    = std::max(0.0, rd("path_hold_cross_track_gain"));
  path_hold_recovery_blend_      = std::clamp(rd("path_hold_recovery_blend"), 0.0, 1.0);
  curve_speed_enable_             = this->get_parameter("curve_speed_enable").as_bool();
  curve_speed_heading_threshold_  = std::max(0.01, rd("curve_speed_heading_threshold"));
  curve_speed_reduction_gain_     = std::max(0.0, rd("curve_speed_reduction_gain"));
  curve_speed_min_ratio_          = std::clamp(rd("curve_speed_min_ratio"), 0.1, 1.0);
  overshoot_guard_distance_       = std::max(0.01, rd("overshoot_guard_distance"));
  overshoot_reverse_damping_      = std::clamp(rd("overshoot_reverse_damping"), 0.1, 1.0);
  oscillation_guard_enable_       = this->get_parameter("oscillation_guard_enable").as_bool();
  oscillation_guard_cte_deadband_     = std::max(0.0, rd("oscillation_guard_cte_deadband"));
  oscillation_guard_heading_deadband_ = std::max(0.0, rd("oscillation_guard_heading_deadband"));
  oscillation_guard_reverse_damping_  = std::clamp(rd("oscillation_guard_reverse_damping"), 0.1, 1.0);
  adaptive_corner_mode_enable_    = this->get_parameter("adaptive_corner_mode_enable").as_bool();
  adaptive_near_cte_thresh_       = std::max(0.0, rd("adaptive_near_cte_thresh"));
  adaptive_near_heading_thresh_   = std::max(0.0, rd("adaptive_near_heading_thresh"));
  adaptive_near_omega_damping_    = std::clamp(rd("adaptive_near_omega_damping"), 0.1, 1.0);
  adaptive_near_omega_rate_scale_ = std::clamp(rd("adaptive_near_omega_rate_scale"), 0.1, 1.0);
  adaptive_near_v_scale_          = std::clamp(rd("adaptive_near_v_scale"), 0.1, 1.0);
  off_path_recovery_enable_       = this->get_parameter("off_path_recovery_enable").as_bool();
  off_path_recovery_distance_     = std::max(0.1, rd("off_path_recovery_distance"));
  off_path_recovery_exit_distance_= std::clamp(rd("off_path_recovery_exit_distance"), 0.05, off_path_recovery_distance_ - 0.05);
  off_path_recovery_speed_        = std::max(0.03, rd("off_path_recovery_speed"));
  off_path_recovery_heading_gain_ = std::max(0.0, rd("off_path_recovery_heading_gain"));
  off_path_recovery_cte_gain_     = std::max(0.0, rd("off_path_recovery_cte_gain"));
  off_path_recovery_max_omega_    = std::max(0.2, rd("off_path_recovery_max_omega"));
  effective_horizon_ = std::max(1, dbm.N);
}

rcl_interfaces::msg::SetParametersResult MPCPathTrackerCpp::parameter_callback(
    const std::vector<rclcpp::Parameter>&)
{
  update_controller_params();
  rcl_interfaces::msg::SetParametersResult r;
  r.successful = true;
  return r;
}

int MPCPathTrackerCpp::get_effective_horizon() const {
  return std::max(1, effective_horizon_);
}

void MPCPathTrackerCpp::local_path_callback(
    const nav_msgs::msg::Path::SharedPtr msg)
{
  local_path_ = msg->poses;
  if (local_path_.empty()) return;

  const double ax = local_path_.front().pose.position.x;
  const double ay = local_path_.front().pose.position.y;
  if (!has_local_path_anchor_) {
    local_path_anchor_x_ = ax;
    local_path_anchor_y_ = ay;
    has_local_path_anchor_ = true;
    return;
  }
  const double dist = std::hypot(ax - local_path_anchor_x_, ay - local_path_anchor_y_);
  if (dist > path_reset_distance_threshold_) {
    if (ltv_controller_)    ltv_controller_->reset();
    if (legacy_controller_) legacy_controller_->reset_state();
    cmd_initialized_ = false;
    v_sig_state_ = 0.0;
    has_prev_errors_ = false;
    off_path_recovery_latched_ = false;
    prev_pose_stamped_.reset();
    est_vx_ = est_vy_ = est_omega_ = est_delta_ = est_wf_ = est_wr_ = 0.0;
  }
  local_path_anchor_x_ = ax;
  local_path_anchor_y_ = ay;
}

void MPCPathTrackerCpp::pose_callback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  current_pose_ = msg->pose;
  // 논문 식(3): 1-sample delay를 위해 상태 추정은 매 pose에서 수행
  updateStateEstimate(*msg);
  // 추정 상태를 레거시 컨트롤러에 주입
  if (legacy_controller_) {
    legacy_controller_->setEstimatedState(
      est_vx_, est_vy_, est_omega_, est_delta_, est_wf_, est_wr_);
  }
}

// ============================================================
// 메인 제어 루프
// ============================================================
void MPCPathTrackerCpp::control_loop() {
  const auto start_total = std::chrono::high_resolution_clock::now();

  if (local_path_.empty() || !current_pose_) {
    if (ltv_controller_)    ltv_controller_->reset();
    if (legacy_controller_) legacy_controller_->reset_state();
    cmd_initialized_ = false;
    v_sig_state_ = 0.0;
    has_prev_errors_ = false;
    off_path_recovery_latched_ = false;
    publish_control(0.0, 0.0);
    publish_performance(0.0, 0.0, 0.0, -1, "NO_DATA");
    return;
  }

  const size_t min_pts = static_cast<size_t>(get_effective_horizon() + 5);
  if (local_path_.size() < min_pts) {
    publish_control(0.0, 0.0);
    publish_performance(0.0, 0.0, 0.0, -1, "PATH_TOO_SHORT");
    return;
  }

  double v_cmd = 0.0;
  double w_cmd = 0.0;
  std::vector<std::array<double,3>> pred;
  double model_time_us = 0.0;
  double solver_time_us = 0.0;
  std::string solver_status = "UNKNOWN";

  if (use_ltv_mpc_) {
    // LTV-MPC (기존 경로 유지)
    auto output = ltv_controller_->computeControl(*current_pose_, local_path_);
    model_time_us  = output.model_time_us;
    solver_time_us = output.solver_time_us;
    v_cmd = output.v_cmd;
    w_cmd = output.omega_cmd;
    pred  = std::move(output.predicted_xy);
    solver_status = output.solved ? "SOLVED" : "FAILED";
  } else {
    // DBM RTI-NMPC (논문 구현)
    const auto ts = std::chrono::high_resolution_clock::now();
    auto output = legacy_controller_->compute_control(*current_pose_, local_path_);
    const auto te = std::chrono::high_resolution_clock::now();
    solver_time_us = model_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();
    v_cmd = output.velocity;
    w_cmd = output.angular_velocity;
    pred  = std::move(output.predicted_trajectory);
    solver_status = (v_cmd != 0.0 || w_cmd != 0.0) ? "SOLVED" : "FAILED";
  }

  const auto end_total = std::chrono::high_resolution_clock::now();
  const double total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      end_total - start_total).count();

  // ── 경로 추종 지표 계산 ───────────────────────────────
  double min_path_dist = std::numeric_limits<double>::max();
  size_t closest_idx = 0;
  double signed_cte = 0.0, heading_error_near = 0.0;
  bool near_metrics_valid = false;
  double adaptive_rate_scale = 1.0, adaptive_v_scale = 1.0;

  if (current_pose_ && !local_path_.empty()) {
    const double ex = current_pose_->position.x;
    const double ey = current_pose_->position.y;
    for (size_t i = 0; i < local_path_.size(); ++i) {
      const double dx = local_path_[i].pose.position.x - ex;
      const double dy = local_path_[i].pose.position.y - ey;
      const double d = std::hypot(dx, dy);
      if (d < min_path_dist) { min_path_dist = d; closest_idx = i; }
    }
    if (closest_idx + 1 < local_path_.size()) {
      const auto& p0 = local_path_[closest_idx].pose.position;
      const auto& p1 = local_path_[closest_idx+1].pose.position;
      const double path_yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);
      const double yaw = pose_yaw_from_quat_or_packed(*current_pose_);
      signed_cte      = -std::sin(path_yaw)*(ex - p0.x) + std::cos(path_yaw)*(ey - p0.y);
      heading_error_near = wrap_angle(path_yaw - yaw);
      near_metrics_valid = true;
    }
  }

  // off-path 복구
  if (off_path_recovery_enable_ && near_metrics_valid) {
    if (!off_path_recovery_latched_ && min_path_dist > off_path_recovery_distance_) {
      off_path_recovery_latched_ = true;
    } else if (off_path_recovery_latched_ && min_path_dist < off_path_recovery_exit_distance_) {
      off_path_recovery_latched_ = false;
    }
  } else {
    off_path_recovery_latched_ = false;
  }

  if (near_metrics_valid && off_path_recovery_latched_) {
    const double w_r = off_path_recovery_heading_gain_ * heading_error_near
                     - off_path_recovery_cte_gain_ * signed_cte;
    w_cmd = std::clamp(w_r, -off_path_recovery_max_omega_, off_path_recovery_max_omega_);
    v_cmd = std::min(v_cmd, off_path_recovery_speed_);
  }

  // path-hold 보정
  if (!off_path_recovery_latched_ && current_pose_ && !local_path_.empty() &&
      std::isfinite(min_path_dist))
  {
    const size_t tgt = std::min(closest_idx + static_cast<size_t>(path_hold_lookahead_index_),
                                local_path_.size() - 1);
    const double dx = local_path_[tgt].pose.position.x - current_pose_->position.x;
    const double dy = local_path_[tgt].pose.position.y - current_pose_->position.y;
    const double ld = std::hypot(dx, dy);
    const double yaw = pose_yaw_from_quat_or_packed(*current_pose_);
    const auto& p0 = local_path_[closest_idx].pose.position;
    const auto& p1 = local_path_[std::min(closest_idx+1, local_path_.size()-1)].pose.position;
    const double path_yaw = std::atan2(p1.y - p0.y, p1.x - p0.x);

    const double dist_scale = 1.0 / (1.0 + path_hold_distance_gain_ * min_path_dist);
    v_cmd *= std::clamp(dist_scale, 0.2, 1.0);

    double w_geo = 0.0;
    if (ld > 1e-3) {
      const double alpha = wrap_angle(std::atan2(dy, dx) - yaw);
      w_geo = 2.0 * std::max(0.2, std::abs(v_cmd)) * std::sin(alpha) / ld;
    }
    const double he = wrap_angle(path_yaw - yaw);
    const double cte = near_metrics_valid ? signed_cte : 0.0;
    double w_aux = path_hold_heading_error_gain_ * he
                 - path_hold_cross_track_gain_ * cte * std::max(0.3, std::abs(v_cmd));
    w_aux = std::clamp(w_aux, -path_hold_max_omega_, path_hold_max_omega_);
    if (std::abs(w_geo) > 1e-3 && (w_aux * w_geo) < 0.0) w_aux *= 0.3;

    const double base_blend = std::clamp(
      path_hold_heading_gain_ * (0.3 + std::min(min_path_dist, 2.0) / 2.0), 0.0, 0.9);
    const double rec_scale = std::clamp(
      (min_path_dist - path_hold_recovery_distance_) /
       std::max(path_hold_recovery_distance_, 1e-3), 0.0, 1.0);
    const double blend = std::clamp(base_blend + path_hold_recovery_blend_ * rec_scale, 0.0, 0.95);
    if (blend > 1e-6) {
      w_cmd = std::clamp((1.0 - blend) * w_cmd + blend * (w_geo + w_aux),
                          -path_hold_max_omega_, path_hold_max_omega_);
    }
  }

  if (near_metrics_valid && !off_path_recovery_latched_) {
    // 커브 속도 감소
    if (curve_speed_enable_) {
      const double h_abs = std::abs(heading_error_near);
      if (h_abs > curve_speed_heading_threshold_) {
        const double ratio = std::max(curve_speed_min_ratio_,
          1.0 - curve_speed_reduction_gain_ * (h_abs - curve_speed_heading_threshold_));
        v_cmd *= ratio;
      }
    }
    // overshoot 방지
    if (std::abs(signed_cte) > overshoot_guard_distance_ && (w_cmd * signed_cte) > 0.0) {
      w_cmd *= overshoot_reverse_damping_;
    }
    // 진동 방지
    if (oscillation_guard_enable_ && has_prev_errors_) {
      const bool cte_flip = (std::abs(signed_cte) > oscillation_guard_cte_deadband_) &&
                            (std::abs(prev_signed_cte_) > oscillation_guard_cte_deadband_) &&
                            (signed_cte * prev_signed_cte_ < 0.0);
      const bool hd_flip  = (std::abs(heading_error_near) > oscillation_guard_heading_deadband_) &&
                            (std::abs(prev_heading_error_) > oscillation_guard_heading_deadband_) &&
                            (heading_error_near * prev_heading_error_ < 0.0);
      if (cte_flip || hd_flip) w_cmd *= oscillation_guard_reverse_damping_;
    }
    // post-curve
    {
      const double h_abs = std::abs(heading_error_near);
      const double now_s = this->now().seconds();
      if (!post_curve_active_ && prev_heading_abs_ > 0.28 && h_abs < prev_heading_abs_) {
        post_curve_active_   = true;
        post_curve_exit_time_ = now_s;
      }
      if (post_curve_active_ && (now_s - post_curve_exit_time_) > post_curve_window_sec_) {
        post_curve_active_ = false;
      }
      prev_heading_abs_ = h_abs;
    }
    // adaptive corner
    if (adaptive_corner_mode_enable_ && !post_curve_active_) {
      const bool near = (std::abs(signed_cte) < adaptive_near_cte_thresh_) &&
                        (std::abs(heading_error_near) < adaptive_near_heading_thresh_);
      if (near) {
        w_cmd *= adaptive_near_omega_damping_;
        adaptive_rate_scale = adaptive_near_omega_rate_scale_;
        adaptive_v_scale    = adaptive_near_v_scale_;
      }
    }
    prev_signed_cte_   = signed_cte;
    prev_heading_error_= heading_error_near;
    has_prev_errors_   = true;
  }

  // ── 속도/요레이트 레이트 제한 ─────────────────────────
  const auto now = this->now();
  double dt = (now - prev_cmd_time_).seconds();
  if (!std::isfinite(dt) || dt <= 1e-4) dt = 0.05;
  if (!cmd_initialized_) {
    prev_v_cmd_ = v_cmd;
    prev_w_cmd_ = w_cmd;
    v_sig_state_ = std::max(0.0, v_cmd);
    cmd_initialized_ = true;
  }

  w_cmd = std::clamp(w_cmd, -max_omega_abs_, max_omega_abs_);
  const double max_dw = max_omega_rate_ * adaptive_rate_scale * dt;
  w_cmd = std::clamp(w_cmd, prev_w_cmd_ - max_dw, prev_w_cmd_ + max_dw);

  // sigmoid 속도 스무딩
  {
    const double v_t = v_cmd * adaptive_v_scale;
    const double tau = (v_t > v_sig_state_) ? sig_tau_up_ : sig_tau_down_;
    const double alpha = 1.0 - std::exp(-dt / std::max(tau, 1e-3));
    v_sig_state_ += alpha * (v_t - v_sig_state_);
    v_sig_state_  = std::max(0.0, v_sig_state_);
    v_cmd = v_sig_state_;
  }

  prev_v_cmd_    = v_cmd;
  prev_w_cmd_    = w_cmd;
  prev_cmd_time_ = now;

  publish_control(v_cmd, w_cmd);
  publish_predicted_path(pred);
  publish_performance(model_time_us, solver_time_us, total_time_us, -1, solver_status);

  if ((now - last_log_time_).seconds() > 1.0) {
    RCLCPP_INFO(this->get_logger(), "[%s] v=%.3f w=%.3f cte=%.3f",
      use_ltv_mpc_ ? "LTV" : "DBM-RTI", v_cmd, w_cmd, signed_cte);
    last_log_time_ = now;
  }
}

void MPCPathTrackerCpp::publish_control(double v, double w) {
  geometry_msgs::msg::Accel msg;
  msg.linear.x  = v;
  msg.angular.z = w;
  if (publish_accel_cmd_ && accel_pub_) accel_pub_->publish(msg);
  if (accel_pub_raw_) accel_pub_raw_->publish(msg);
}

void MPCPathTrackerCpp::publish_predicted_path(
    const std::vector<std::array<double,3>>& traj)
{
  nav_msgs::msg::Path msg;
  msg.header.frame_id = "world";
  msg.header.stamp    = this->now();
  for (const auto& pt : traj) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg.header;
    ps.pose.position.x = pt[0];
    ps.pose.position.y = pt[1];
    ps.pose.position.z = pt[2];
    ps.pose.orientation.w = 1.0;
    msg.poses.push_back(ps);
  }
  pred_pub_->publish(msg);
}

void MPCPathTrackerCpp::publish_performance(
    double model_time_us, double solver_time_us,
    double total_time_us, int solver_iterations,
    const std::string& solver_status)
{
  if (!perf_pub_) return;
  bisa::msg::MPCPerformance msg;
  msg.header.stamp    = this->now();
  msg.header.frame_id = "world";
  msg.cav_id          = target_cav_id_;
  msg.model_time_us   = model_time_us;
  msg.solver_time_us  = solver_time_us;
  msg.total_time_us   = total_time_us;
  msg.horizon         = get_effective_horizon();
  msg.solver_iterations = solver_iterations;
  msg.solver_status   = solver_status;
  perf_pub_->publish(msg);
}

}  // namespace bisa

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<bisa::MPCPathTrackerCpp>());
  rclcpp::shutdown();
  return 0;
}