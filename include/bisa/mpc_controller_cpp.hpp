#pragma once
// ============================================================
// mpc_controller_cpp.hpp  (bisa)
//
// Wagner & Normey-Rico (2024) arXiv:2410.12170 기반
// 9-state Dynamic Bicycle Model RTI-NMPC
//
// 상태 x(9) = [px, py, ψ, vx, vy, ω, δc, ωf, ωr]
// 입력 u(2) = [v_delta (조향 속도 rad/s), T_drive (구동 토크 N*m)]
// ============================================================
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <osqp/osqp.h>

#include <array>
#include <optional>
#include <vector>

namespace bisa {

// ── 9-state DBM RTI-NMPC 파라미터 ──────────────────────────
struct DBMRTINMPCParams {
  // 공통
  int    N              {25};       // 예측 호라이즌
  double Ts             {0.05};     // 샘플링 주기 [s]
  double wheelbase      {0.30};     // 축간 거리 [m]
  double max_velocity   {0.72};     // 최대 속도 [m/s]
  double min_velocity   {0.18};     // 최소 속도 [m/s]

  // 물리 파라미터
  double mass_kg        {0.8};      // 차량 질량 [kg]
  double inertia_kgm2   {0.02};     // 요 관성모멘트 [kg*m^2]
  double lf_m           {0.15};     // 무게중심~앞축 [m]
  double lr_m           {0.15};     // 무게중심~뒷축 [m]
  double Ca_N_rad       {10.0};     // 코너링 강성 [N/rad]
  double Cx_N           {12.0};     // 종방향 타이어 강성 [N]
  double Jw_kgm2        {0.002};    // 바퀴 관성모멘트 [kg*m^2]
  double r_w_m          {0.033};    // 타이어 반경 [m]
  double v_min_mps      {0.05};     // 분모 0 방지 최솟값
  double mu             {0.8};      // 마찰계수

  // 입력 제약
  double v_delta_max    {1.5};      // 조향 속도 최대 [rad/s]  (논문: ±1.5)
  double T_drive_max    {0.5};      // 구동 토크 최대 [N*m]    (논문: ±300N*m → 소형 스케일)
  double delta_c_max    {0.785};    // 조향각 최대 [rad] ≈ 45°

  // 비용 가중치
  double w_xy           {20.0};
  double w_psi          {8.0};
  double w_vx           {1.0};
  double w_vy           {0.5};
  double w_omega        {0.5};
  double w_delta_c      {0.2};
  double w_wheel        {0.01};
  double w_u_delta      {5.0};      // 논문: weight 5 (normalized)
  double w_u_torque     {5.0};      // 논문: weight 5 (normalized)
};

// ── 9-state DBM 현재 상태 ───────────────────────────────────
struct DBMState {
  double px      {0.0};
  double py      {0.0};
  double psi     {0.0};
  double vx      {0.0};
  double vy      {0.0};
  double omega   {0.0};
  double delta_c {0.0};
  double omega_f {0.0};
  double omega_r {0.0};
};

// ── 제어 출력 ───────────────────────────────────────────────
struct DBMCommand {
  double v_cmd      {0.0};   // 속도 명령 [m/s]
  double omega_cmd  {0.0};   // 요레이트 명령 [rad/s]
  double kappa_cmd  {0.0};   // 곡률 [1/m]
  double delta_c_opt{0.0};   // 최적 조향각 [rad]
  double T_drive_opt{0.0};   // 최적 토크 [N*m]
  double model_time_us{0.0};
  double solver_time_us{0.0};
  bool   solved     {false};
  std::vector<std::array<double,3>> predicted_xy;
};

// ── 타이어 힘 구조체 ─────────────────────────────────────────
struct TireForces {
  double alpha_f{0.0}, alpha_r{0.0};
  double sigma_f{0.0}, sigma_r{0.0};
  double Fy_f{0.0},    Fy_r{0.0};
  double Fx_f{0.0},    Fx_r{0.0};
  double v_safe{0.0};
  double denom_f{1.0}, denom_r{1.0};
  // Jacobian용 편미분
  double daf_dvy{0.0}, daf_dom{0.0};
  double dar_dvy{0.0}, dar_dom{0.0};
};

// ============================================================
// DBMRTINMPCController
// ============================================================
class DBMRTINMPCController {
public:
  explicit DBMRTINMPCController(const DBMRTINMPCParams& params = DBMRTINMPCParams{});
  void setConfig(const DBMRTINMPCParams& params);
  void reset();

  // 메인 제어 계산
  // state: 현재 9-state (외부에서 추정)
  // path:  로컬 경로
  // speed_mps: 목표 속도 (부호 포함, 음수=후진)
  DBMCommand computeControl(
    const DBMState& state,
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    double speed_mps);

  static double wrapAngle(double a);

private:
  DBMRTINMPCParams params_;

  // warm start (스텝 간 유지)
  std::vector<Eigen::Matrix<double,2,1>> warm_u_;
  std::vector<Eigen::Matrix<double,9,1>> warm_x_;
  bool warm_valid_{false};

  // 타이어 힘 계산
  TireForces computeTireForces(
    const Eigen::Matrix<double,9,1>& x) const;

  // 논문 식(26-27): ẋ = f(x, u)
  Eigen::Matrix<double,9,1> dbmF(
    const Eigen::Matrix<double,9,1>& x,
    const Eigen::Matrix<double,2,1>& u) const;

  // 논문 식(12-14): Jacobian Ac(9×9), Bc(9×2)
  void dbmJacobian(
    const Eigen::Matrix<double,9,1>& x,
    const Eigen::Matrix<double,2,1>& u,
    Eigen::Matrix<double,9,9>& Ac,
    Eigen::Matrix<double,9,2>& Bc) const;

  // 논문 식(2)(8): Implicit Euler (fixed-point 1회)
  Eigen::Matrix<double,9,1> dbmImplicitEuler(
    const Eigen::Matrix<double,9,1>& xk,
    const Eigen::Matrix<double,2,1>& uk) const;

  // 논문 식(16-20): G(x,u) 마찰원 선형화 Jacobians D, E
  void frictionCircleJacobian(
    const Eigen::Matrix<double,9,1>& x,
    const TireForces& t,
    Eigen::Matrix<double,2,9>& D,
    Eigen::Matrix<double,2,2>& E) const;
};

// ============================================================
// 레거시 호환 wrapper (mpc_path_tracker_cpp.cpp가 참조)
// MPCControllerCpp → DBMRTINMPCController 위임
// ============================================================
struct LTVMPCParams {
  int    N            {25};
  double Ts           {0.05};
  double l            {0.30};
  double wd           {20.0};
  double wtheta       {8.0};
  double wkappa       {1.0};
  double wu           {5.0};
  double max_velocity {0.72};
  double min_velocity {0.18};
  double u_min        {-1.5};
  double u_max        {1.5};
  double kappa_min_delta{-3.0};
  double kappa_max_delta{ 3.0};
  double mu           {0.8};
};

struct ControlOutput {
  double velocity         {0.0};
  double angular_velocity {0.0};
  std::vector<std::array<double,3>> predicted_trajectory;
};

struct CollisionConstraints {};

class MPCControllerCpp {
public:
  MPCControllerCpp();
  void update_parameters(const LTVMPCParams& params);
  void reset_state();

  // 레거시 인터페이스 유지
  ControlOutput compute_control(
    const geometry_msgs::msg::Pose& current_pose,
    const std::vector<geometry_msgs::msg::PoseStamped>& local_path,
    const CollisionConstraints& constraints = CollisionConstraints());

  // 상태 주입 (tracker에서 추정값 전달용)
  void setEstimatedState(double vx, double vy, double omega,
                         double delta_c, double omega_f, double omega_r);

private:
  LTVMPCParams params_;
  DBMRTINMPCParams dbm_params_;
  DBMRTINMPCController dbm_ctrl_;

  // 추정 상태
  double est_vx_    {0.0};
  double est_vy_    {0.0};
  double est_omega_ {0.0};
  double est_delta_ {0.0};
  double est_omegaf_{0.0};
  double est_omegar_{0.0};

  static double quatToYaw(const geometry_msgs::msg::Quaternion& q);
};

}  // namespace bisa