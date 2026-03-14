#pragma once
// ============================================================
// mpc_path_tracker_cpp.hpp  (bisa)
// DBM RTI-NMPC 통합 버전
// ============================================================
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/accel.hpp>
#include "bisa/msg/mpc_performance.hpp"
#include "bisa/mpc_controller_cpp.hpp"
#include "ltv_mpc/ltv_mpc.hpp"
#include <memory>
#include <optional>

namespace bisa {

class MPCPathTrackerCpp : public rclcpp::Node {
public:
  MPCPathTrackerCpp();

private:
  int get_effective_horizon() const;
  void local_path_callback(const nav_msgs::msg::Path::SharedPtr msg);
  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void control_loop();
  void publish_control(double v, double w);
  void publish_predicted_path(const std::vector<std::array<double, 3>>& traj);
  void publish_performance(double model_time_us, double solver_time_us,
                           double total_time_us, int solver_iterations,
                           const std::string& solver_status);
  void update_controller_params();
  rcl_interfaces::msg::SetParametersResult parameter_callback(
      const std::vector<rclcpp::Parameter>& params);

  // ── 상태 추정 (DBM RTI-NMPC용) ───────────────────────────
  // pose 차분으로 body-frame 속도 추정
  void updateStateEstimate(const geometry_msgs::msg::PoseStamped& new_pose);

  // ── 컨트롤러 ─────────────────────────────────────────────
  std::unique_ptr<LTVMPC>          ltv_controller_;
  std::unique_ptr<MPCControllerCpp> legacy_controller_;  // DBM RTI-NMPC wrapper

  std::vector<geometry_msgs::msg::PoseStamped> local_path_;
  std::optional<geometry_msgs::msg::Pose>      current_pose_;

  // ── 상태 추정 변수 ────────────────────────────────────────
  std::optional<geometry_msgs::msg::PoseStamped> prev_pose_stamped_;
  double est_vx_    {0.0};
  double est_vy_    {0.0};
  double est_omega_ {0.0};
  double est_delta_ {0.0};  // 조향각 추정 (omega * L / v)
  double est_wf_    {0.0};  // 앞바퀴 각속도 추정
  double est_wr_    {0.0};  // 뒷바퀴 각속도 추정

  // ── 구독/발행 ─────────────────────────────────────────────
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr         local_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr      accel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Accel>::SharedPtr      accel_pub_raw_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr            pred_pub_;
  rclcpp::Publisher<bisa::msg::MPCPerformance>::SharedPtr      perf_pub_;
  rclcpp::TimerBase::SharedPtr                                 timer_;

  rclcpp::Time last_log_time_;
  rclcpp::Time prev_cmd_time_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  // ── 제어 변수 ─────────────────────────────────────────────
  double max_omega_abs_   {1.7};
  double max_omega_rate_  {14.0};
  double max_v_rate_      {2.0};
  double prev_v_cmd_      {0.0};
  double prev_w_cmd_      {0.0};
  bool   cmd_initialized_ {false};

  double v_sig_state_     {0.0};
  double sig_tau_up_      {0.40};
  double sig_tau_down_    {0.20};

  int    effective_horizon_{25};
  int    target_cav_id_   {1};
  bool   use_ltv_mpc_     {false};  // false = DBM RTI-NMPC 사용
  bool   publish_accel_cmd_{true};

  // ── path-hold / 보정 ─────────────────────────────────────
  bool   has_local_path_anchor_      {false};
  double local_path_anchor_x_        {0.0};
  double local_path_anchor_y_        {0.0};
  double path_reset_distance_threshold_{1.0};

  double path_hold_distance_gain_    {0.22};
  double path_hold_heading_gain_     {0.22};
  double path_hold_max_omega_        {1.40};
  int    path_hold_lookahead_index_  {12};
  double path_hold_recovery_distance_{0.15};
  double path_hold_heading_error_gain_{0.80};
  double path_hold_cross_track_gain_ {0.70};
  double path_hold_recovery_blend_   {0.45};

  bool   curve_speed_enable_          {true};
  double curve_speed_heading_threshold_{0.16};
  double curve_speed_reduction_gain_  {0.04};
  double curve_speed_min_ratio_       {0.96};

  double overshoot_guard_distance_    {0.13};
  double overshoot_reverse_damping_   {0.52};

  bool   oscillation_guard_enable_              {true};
  double oscillation_guard_cte_deadband_        {0.08};
  double oscillation_guard_heading_deadband_    {0.10};
  double oscillation_guard_reverse_damping_     {0.70};

  bool   adaptive_corner_mode_enable_           {true};
  double adaptive_near_cte_thresh_              {0.07};
  double adaptive_near_heading_thresh_          {0.07};
  double adaptive_near_omega_damping_           {0.82};
  double adaptive_near_omega_rate_scale_        {0.55};
  double adaptive_near_v_scale_                 {0.97};

  bool   off_path_recovery_enable_              {true};
  double off_path_recovery_distance_            {0.50};
  double off_path_recovery_exit_distance_       {0.22};
  double off_path_recovery_speed_               {0.08};
  double off_path_recovery_heading_gain_        {1.0};
  double off_path_recovery_cte_gain_            {1.2};
  double off_path_recovery_max_omega_           {1.0};
  bool   off_path_recovery_latched_             {false};

  bool   has_prev_errors_    {false};
  double prev_signed_cte_    {0.0};
  double prev_heading_error_ {0.0};
  double prev_heading_abs_   {0.0};
  bool   post_curve_active_  {false};
  double post_curve_exit_time_{0.0};
  double post_curve_window_sec_{1.8};
};

}  // namespace bisa
