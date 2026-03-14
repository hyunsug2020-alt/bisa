// ============================================================
// mpc_controller_cpp.cpp  (bisa)
//
// Wagner & Normey-Rico (2024) arXiv:2410.12170
// 9-state Dynamic Bicycle Model RTI-NMPC
//
// 논문 구현 대응:
//   식(1)      ẋ = f(x, u)           → dbmF()
//   식(2)(8)   Implicit Euler         → dbmImplicitEuler()
//   식(10)     이산 Jacobian Ad, Bd   → 솔버 내부
//              Ad = (I - Ac·dt)^{-1}  (Implicit Euler)
//              Bd = Ad · Bc · dt
//   식(11)     선형화 전개 (A0,A1,B)  → A_batch, B_batch, F_batch
//   식(12-14)  Ac = ∂f/∂x, Bc = ∂f/∂u → dbmJacobian()
//   식(15)     δx 동역학              → QP 내부
//   식(16-19)  G(x,u) 제약 선형화    → frictionCircleJacobian()
//   식(20)     최종 QP               → OSQP
//   식(25-27)  DBM + 바퀴 동역학     → dbmF()
//   식(28-29)  선형 타이어 모델       → computeTireForces()
//   식(30)     마찰원 제약            → A_fc 블록
// ============================================================
#include "bisa/mpc_controller_cpp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <chrono>

namespace bisa {

static constexpr double kPi = 3.14159265358979323846;

// ============================================================
// DBMRTINMPCController
// ============================================================
DBMRTINMPCController::DBMRTINMPCController(const DBMRTINMPCParams& p)
  : params_(p) {}

void DBMRTINMPCController::setConfig(const DBMRTINMPCParams& p) {
  params_ = p;
  warm_valid_ = false;
  warm_u_.clear();
  warm_x_.clear();
}

void DBMRTINMPCController::reset() {
  warm_valid_ = false;
  warm_u_.clear();
  warm_x_.clear();
}

double DBMRTINMPCController::wrapAngle(double a) {
  while (a >  kPi) { a -= 2.0 * kPi; }
  while (a < -kPi) { a += 2.0 * kPi; }
  return a;
}

// ============================================================
// 타이어 힘 계산 (논문 식 28-29)
// Flon = Cs * σ,  Flat = Ca * α
// ============================================================
TireForces DBMRTINMPCController::computeTireForces(
    const Eigen::Matrix<double,9,1>& x) const
{
  const double vx    = x(3);
  const double vy    = x(4);
  const double omega = x(5);
  const double dc    = x(6);
  const double wf    = x(7);
  const double wr    = x(8);

  TireForces t;
  t.v_safe  = std::max(std::abs(vx), params_.v_min_mps);
  t.denom_f = t.v_safe * t.v_safe + (vy + params_.lf_m * omega) * (vy + params_.lf_m * omega);
  t.denom_r = t.v_safe * t.v_safe + (vy - params_.lr_m * omega) * (vy - params_.lr_m * omega);

  // 슬립각 (논문 식 22)
  t.alpha_f = dc - std::atan2(vy + params_.lf_m * omega, t.v_safe);
  t.alpha_r =    - std::atan2(vy - params_.lr_m * omega, t.v_safe);

  // 횡방향 힘 (논문 식 29)
  t.Fy_f = params_.Ca_N_rad * t.alpha_f;
  t.Fy_r = params_.Ca_N_rad * t.alpha_r;

  // 종방향 슬립 (논문 식 23)
  t.sigma_f = (wf * params_.r_w_m - vx) / t.v_safe;
  t.sigma_r = (wr * params_.r_w_m - vx) / t.v_safe;

  // 종방향 힘 (논문 식 28)
  t.Fx_f = params_.Cx_N * t.sigma_f;
  t.Fx_r = params_.Cx_N * t.sigma_r;

  // Jacobian용 편미분 (식 12-14에서 사용)
  t.daf_dvy = -t.v_safe / t.denom_f;
  t.daf_dom = -params_.lf_m * t.v_safe / t.denom_f;
  t.dar_dvy = -t.v_safe / t.denom_r;
  t.dar_dom =  params_.lr_m * t.v_safe / t.denom_r;

  return t;
}

// ============================================================
// 논문 식(26)(27): ẋ = f(x, u)
//
// x = [px, py, ψ, vx, vy, ω, δc, ωf, ωr]
// u = [v_delta, T_drive]
//
// 전진/후진: vx 부호 그대로 유지
// 앞바퀴: 토크 없음 (구동력 없음)
// 뒷바퀴: T_drive 전달 (RWD)
// ============================================================
Eigen::Matrix<double,9,1> DBMRTINMPCController::dbmF(
    const Eigen::Matrix<double,9,1>& x,
    const Eigen::Matrix<double,2,1>& u) const
{
  const double psi    = x(2);
  const double vx     = x(3);
  const double vy     = x(4);
  const double omega  = x(5);
  const double dc     = x(6);
  const double v_delta = u(0);
  const double T_drive = u(1);

  const double cos_d = std::cos(dc);
  const double sin_d = std::sin(dc);

  const auto t = computeTireForces(x);

  Eigen::Matrix<double,9,1> xdot;
  // 논문 식(26): DBM 동역학
  xdot(0) =  vx * std::cos(psi) - vy * std::sin(psi);
  xdot(1) =  vx * std::sin(psi) + vy * std::cos(psi);
  xdot(2) =  omega;
  xdot(3) = (2.0 * t.Fx_f * cos_d - 2.0 * t.Fy_f * sin_d + 2.0 * t.Fx_r)
             / params_.mass_kg + vy * omega;
  xdot(4) = (2.0 * t.Fx_f * sin_d + 2.0 * t.Fy_f * cos_d + 2.0 * t.Fy_r)
             / params_.mass_kg - vx * omega;
  xdot(5) = (2.0 * params_.lf_m * (t.Fx_f * sin_d + t.Fy_f * cos_d)
             - 2.0 * params_.lr_m * t.Fy_r)
             / params_.inertia_kgm2;
  // 논문 식(25): δ̇c = v_delta  (steering-rate 입력)
  xdot(6) = v_delta;
  // 논문 식(27): 바퀴 동역학 (앞: 저항만, 뒤: 토크)
  xdot(7) = (-params_.r_w_m * t.Fx_f) / params_.Jw_kgm2;
  xdot(8) = (T_drive - params_.r_w_m * t.Fx_r) / params_.Jw_kgm2;

  return xdot;
}

// ============================================================
// 논문 식(12-14): Jacobian
// Ac = ∂f/∂x (9×9),  Bc = ∂f/∂u (9×2)
// ============================================================
void DBMRTINMPCController::dbmJacobian(
    const Eigen::Matrix<double,9,1>& x,
    const Eigen::Matrix<double,2,1>& /*u*/,
    Eigen::Matrix<double,9,9>& Ac,
    Eigen::Matrix<double,9,2>& Bc) const
{
  const double psi   = x(2);
  const double vx    = x(3);
  const double vy    = x(4);
  const double omega = x(5);
  const double dc    = x(6);
  const double cos_d = std::cos(dc);
  const double sin_d = std::sin(dc);

  const auto t = computeTireForces(x);
  const double m   = params_.mass_kg;
  const double Iz  = params_.inertia_kgm2;
  const double lf  = params_.lf_m;
  const double lr  = params_.lr_m;
  const double Ca  = params_.Ca_N_rad;
  const double Cx  = params_.Cx_N;
  const double rw  = params_.r_w_m;
  const double Jw  = params_.Jw_kgm2;
  const double vs  = t.v_safe;

  // Fy 편미분
  const double dFyf_dvy = Ca * t.daf_dvy;
  const double dFyf_dom = Ca * t.daf_dom;
  const double dFyf_ddc = Ca;               // ∂α_f/∂δc = 1
  const double dFyr_dvy = Ca * t.dar_dvy;
  const double dFyr_dom = Ca * t.dar_dom;

  // Fx 편미분 (v_safe 고정 근사)
  const double dFxf_dvx = Cx * (-1.0 / vs);
  const double dFxf_dwf = Cx * (rw / vs);
  const double dFxr_dvx = Cx * (-1.0 / vs);
  const double dFxr_dwr = Cx * (rw / vs);

  Ac = Eigen::Matrix<double,9,9>::Zero();

  // 행 0: ṗx = vx*cos(ψ) - vy*sin(ψ)
  Ac(0,2) = -vx * std::sin(psi) - vy * std::cos(psi);
  Ac(0,3) =  std::cos(psi);
  Ac(0,4) = -std::sin(psi);

  // 행 1: ṗy = vx*sin(ψ) + vy*cos(ψ)
  Ac(1,2) =  vx * std::cos(psi) - vy * std::sin(psi);
  Ac(1,3) =  std::sin(psi);
  Ac(1,4) =  std::cos(psi);

  // 행 2: ψ̇ = ω
  Ac(2,5) = 1.0;

  // 행 3: v̇x = (2Fxf*cos-2Fyf*sin+2Fxr)/m + vy*ω
  Ac(3,3) = (2.0 * dFxf_dvx * cos_d + 2.0 * dFxr_dvx) / m;
  Ac(3,4) = -2.0 * dFyf_dvy * sin_d / m + omega;
  Ac(3,5) = -2.0 * dFyf_dom * sin_d / m + vy;
  Ac(3,6) = 2.0 * (-t.Fx_f * sin_d - dFyf_ddc * sin_d - t.Fy_f * cos_d) / m;
  Ac(3,7) =  2.0 * dFxf_dwf * cos_d / m;
  Ac(3,8) =  2.0 * dFxr_dwr / m;

  // 행 4: v̇y = (2Fxf*sin+2Fyf*cos+2Fyr)/m - vx*ω
  Ac(4,3) = 2.0 * dFxf_dvx * sin_d / m - omega;
  Ac(4,4) = 2.0 * (dFyf_dvy * cos_d + dFyr_dvy) / m;
  Ac(4,5) = 2.0 * (dFyf_dom * cos_d + dFyr_dom) / m - vx;
  Ac(4,6) = 2.0 * (t.Fx_f * cos_d + dFyf_ddc * cos_d - t.Fy_f * sin_d) / m;
  Ac(4,7) =  2.0 * dFxf_dwf * sin_d / m;

  // 행 5: ω̇ = (2lf*(Fxf*sin+Fyf*cos) - 2lr*Fyr)/Iz
  Ac(5,3) = 2.0 * lf * dFxf_dvx * sin_d / Iz;
  Ac(5,4) = 2.0 * (lf * dFyf_dvy * cos_d - lr * dFyr_dvy) / Iz;
  Ac(5,5) = 2.0 * (lf * dFyf_dom * cos_d - lr * dFyr_dom) / Iz;
  Ac(5,6) = 2.0 * lf * (t.Fx_f * cos_d + dFyf_ddc * cos_d - t.Fy_f * sin_d) / Iz;
  Ac(5,7) =  2.0 * lf * dFxf_dwf * sin_d / Iz;

  // 행 6: δ̇c = v_delta
  // (모두 0 - v_delta는 입력이고 δc는 상태)

  // 행 7: ω̇f = -r*Fxf/Jw
  Ac(7,3) = -rw * dFxf_dvx / Jw;
  Ac(7,7) = -rw * dFxf_dwf / Jw;

  // 행 8: ω̇r = (T_drive - r*Fxr)/Jw
  Ac(8,3) = -rw * dFxr_dvx / Jw;
  Ac(8,8) = -rw * dFxr_dwr / Jw;

  // Bc = ∂f/∂u (9×2)
  Bc = Eigen::Matrix<double,9,2>::Zero();
  Bc(6,0) = 1.0;                      // ∂δ̇c/∂v_delta = 1
  Bc(8,1) = 1.0 / Jw;                 // ∂ω̇r/∂T_drive = 1/Jw
}

// ============================================================
// 논문 식(2)(8): Implicit Euler  (fixed-point 1회)
// x_{k+1} ← x_k + f(x^{(0)}_{k+1}, u_k)·dt
// x^{(0)} = x_k + f(x_k, u_k)·dt  (Explicit 예측)
// ============================================================
Eigen::Matrix<double,9,1> DBMRTINMPCController::dbmImplicitEuler(
    const Eigen::Matrix<double,9,1>& xk,
    const Eigen::Matrix<double,2,1>& uk) const
{
  const double dt = params_.Ts;
  Eigen::Matrix<double,9,1> x0 = xk + dbmF(xk, uk) * dt;
  x0(2) = wrapAngle(x0(2));
  Eigen::Matrix<double,9,1> x1 = xk + dbmF(x0, uk) * dt;
  x1(2) = wrapAngle(x1(2));
  return x1;
}

// ============================================================
// 논문 식(16-19): 마찰원 제약 G(x,u) 선형화
// G = Flat^2 + Flon^2 - (μmg)^2 ≤ 0
// ∂G/∂x = D (2×9),  ∂G/∂u = E (2×2)
// (앞바퀴/뒷바퀴 각각 1행씩)
// ============================================================
void DBMRTINMPCController::frictionCircleJacobian(
    const Eigen::Matrix<double,9,1>& x,
    const TireForces& t,
    Eigen::Matrix<double,2,9>& D,
    Eigen::Matrix<double,2,2>& E) const
{
  D = Eigen::Matrix<double,2,9>::Zero();
  E = Eigen::Matrix<double,2,2>::Zero();

  const double Ca  = params_.Ca_N_rad;
  const double Cx  = params_.Cx_N;
  const double rw  = params_.r_w_m;
  const double vs  = t.v_safe;
  const double dc  = x(6);
  const double cos_d = std::cos(dc);
  const double sin_d = std::sin(dc);

  // ── 앞바퀴 (행 0) ─────────────────────────────────────────
  // G_f = Fy_f^2 + Fx_f^2 - (μmg)^2
  // ∂G_f/∂x_j = 2*Fy_f*∂Fy_f/∂x_j + 2*Fx_f*∂Fx_f/∂x_j
  const double dFyf_dvy = Ca * t.daf_dvy;
  const double dFyf_dom = Ca * t.daf_dom;
  const double dFyf_ddc = Ca;
  const double dFxf_dvx = Cx * (-1.0 / vs);
  const double dFxf_dwf = Cx * (rw / vs);

  D(0,3) = 2.0 * t.Fx_f * dFxf_dvx;
  D(0,4) = 2.0 * t.Fy_f * dFyf_dvy;
  D(0,5) = 2.0 * t.Fy_f * dFyf_dom;
  D(0,6) = 2.0 * t.Fy_f * dFyf_ddc;
  D(0,7) = 2.0 * t.Fx_f * dFxf_dwf;

  // ── 뒷바퀴 (행 1) ─────────────────────────────────────────
  const double dFyr_dvy = Ca * t.dar_dvy;
  const double dFyr_dom = Ca * t.dar_dom;
  const double dFxr_dvx = Cx * (-1.0 / vs);
  const double dFxr_dwr = Cx * (rw / vs);

  D(1,3) = 2.0 * t.Fx_r * dFxr_dvx;
  D(1,4) = 2.0 * t.Fy_r * dFyr_dvy;
  D(1,5) = 2.0 * t.Fy_r * dFyr_dom;
  D(1,8) = 2.0 * t.Fx_r * dFxr_dwr;

  // ∂G/∂u = E (2×2): 입력이 u=[v_delta, T_drive]
  // v_delta → δc 상태 변화 → Fy_f 변화 (간접, δc통해)
  // T_drive → ωr → Fx_r 변화 (간접)
  // 직접 미분은 0 (입력이 가속도 형태이므로 1스텝 lag)
  // 논문에서 E = ∂G/∂u → 입력에 대한 직접 편미분은 0
  // (상태를 통한 간접 영향은 D·B로 처리됨)
  (void)cos_d; (void)sin_d;
}

// ============================================================
// 메인 솔버: computeControl
// ============================================================
DBMCommand DBMRTINMPCController::computeControl(
    const DBMState& state,
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    double speed_mps)
{
  DBMCommand out;
  const int N  = params_.N;
  const int nx = 9;
  const int nu = 2;
  const double dt = params_.Ts;

  if (N < 2 || static_cast<int>(path.size()) < 3) { return out; }

  const auto t0 = std::chrono::high_resolution_clock::now();

  // ── 현재 상태 벡터 ──────────────────────────────────────
  Eigen::Matrix<double,9,1> x0;
  x0 << state.px, state.py, state.psi,
        state.vx, state.vy, state.omega,
        state.delta_c, state.omega_f, state.omega_r;

  // ── warm start 초기화 ────────────────────────────────────
  const double dc_max    = params_.delta_c_max;
  const double vd_max    = params_.v_delta_max;
  const double T_max     = params_.T_drive_max;
  const double omega_ref = speed_mps / std::max(params_.r_w_m, 0.01);

  if (!warm_valid_ ||
      static_cast<int>(warm_u_.size()) != N ||
      static_cast<int>(warm_x_.size()) != N + 1)
  {
    warm_u_.resize(N);
    for (int k = 0; k < N; ++k) {
      warm_u_[k](0) = 0.0;
      warm_u_[k](1) = 0.0;
    }
    warm_x_.resize(N + 1);
    warm_x_[0] = x0;
    for (int k = 0; k < N; ++k) {
      warm_x_[k+1] = dbmImplicitEuler(warm_x_[k], warm_u_[k]);
    }
    warm_valid_ = true;
  } else {
    warm_x_[0] = x0;
  }

  // ── 참조 궤적 ────────────────────────────────────────────
  const int path_n = static_cast<int>(path.size());
  // 경로에서 가장 가까운 인덱스 찾기
  int nearest_idx = 0;
  double best_d2 = std::numeric_limits<double>::max();
  for (int i = 0; i < std::min(path_n, 60); ++i) {
    const double dx = path[i].pose.position.x - state.px;
    const double dy = path[i].pose.position.y - state.py;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) { best_d2 = d2; nearest_idx = i; }
  }

  std::vector<Eigen::Matrix<double,9,1>> x_ref(N);
  const double omega_wref = std::abs(speed_mps) / std::max(params_.r_w_m, 0.01);
  for (int k = 0; k < N; ++k) {
    const int idx = std::min(nearest_idx + k, path_n - 1);
    const double path_yaw = [&](){
      int i0 = std::max(0, idx - 1);
      int i1 = std::min(path_n - 1, idx + 1);
      const double dx = path[i1].pose.position.x - path[i0].pose.position.x;
      const double dy = path[i1].pose.position.y - path[i0].pose.position.y;
      return (std::hypot(dx, dy) > 1e-9) ? std::atan2(dy, dx) : state.psi;
    }();
    x_ref[k](0) = path[idx].pose.position.x;
    x_ref[k](1) = path[idx].pose.position.y;
    x_ref[k](2) = path_yaw;
    x_ref[k](3) = speed_mps;
    x_ref[k](4) = 0.0;
    x_ref[k](5) = 0.0;
    x_ref[k](6) = 0.0;
    x_ref[k](7) = omega_wref;
    x_ref[k](8) = omega_wref;
  }

  // ── Jacobian 계산 및 이산화 (논문 식 10) ─────────────────
  // Implicit Euler 이산화:
  //   Ad = (I - Ac·dt)^{-1}
  //   Bd = Ad · Bc · dt
  using M99 = Eigen::Matrix<double,9,9>;
  using M92 = Eigen::Matrix<double,9,2>;
  using V9  = Eigen::Matrix<double,9,1>;
  const M99 I9 = M99::Identity();

  std::vector<M99> Ad(N);
  std::vector<M92> Bd(N);
  std::vector<V9>  fd(N);

  for (int k = 0; k < N; ++k) {
    const V9&  xg = warm_x_[k];
    const Eigen::Matrix<double,2,1>& ug = warm_u_[k];

    M99 Ac;
    M92 Bc;
    dbmJacobian(xg, ug, Ac, Bc);

    // Implicit Euler 이산화
    M99 lhs = I9 - Ac * dt;
    M99 lhs_inv = lhs.inverse();
    Ad[k] = lhs_inv;
    Bd[k] = lhs_inv * (Bc * dt);

    // affine 보정항
    V9 fd_raw = warm_x_[k+1] - Ad[k] * xg - Bd[k] * ug;
    fd_raw(2) = wrapAngle(fd_raw(2));
    fd[k] = fd_raw;
  }

  // ── 배치 행렬 (논문 식 11-15) ────────────────────────────
  const int Nx = nx * N;
  const int Nu = nu * N;

  Eigen::MatrixXd A_batch = Eigen::MatrixXd::Zero(Nx, nx);
  Eigen::MatrixXd B_batch = Eigen::MatrixXd::Zero(Nx, Nu);
  Eigen::VectorXd F_batch = Eigen::VectorXd::Zero(Nx);

  M99 Phi = I9;
  for (int k = 0; k < N; ++k) {
    A_batch.block(k*nx, 0, nx, nx) = Phi;

    if (k == 0) { F_batch.segment(0, nx) = fd[0]; }
    else { F_batch.segment(k*nx, nx) = Ad[k] * F_batch.segment((k-1)*nx, nx) + fd[k]; }
    F_batch.segment(k*nx, nx)(2) = wrapAngle(F_batch.segment(k*nx, nx)(2));

    B_batch.block(k*nx, k*nu, nx, nu) = Bd[k];
    for (int j = k+1; j < N; ++j) {
      // 논문 식 15: Ad[j-1] 사용 (LTV 올바른 전파)
      B_batch.block(j*nx, k*nu, nx, nu) =
        Ad[j-1] * B_batch.block((j-1)*nx, k*nu, nx, nu);
    }
    Phi = Ad[k] * Phi;
  }

  // ── 비용 행렬 Q, R ────────────────────────────────────────
  Eigen::VectorXd q_diag(nx);
  q_diag << params_.w_xy, params_.w_xy, params_.w_psi,
            params_.w_vx, params_.w_vy, params_.w_omega,
            params_.w_delta_c, params_.w_wheel, params_.w_wheel;

  Eigen::VectorXd r_diag(nu);
  r_diag << params_.w_u_delta, params_.w_u_torque;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(Nx, Nx);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(Nu, Nu);
  for (int k = 0; k < N; ++k) {
    Q.block(k*nx, k*nx, nx, nx) = q_diag.asDiagonal();
    R.block(k*nu, k*nu, nu, nu) = r_diag.asDiagonal();
  }

  // ── delta 형식 QP ─────────────────────────────────────────
  V9 delta_x0 = x0 - warm_x_[0];
  delta_x0(2) = wrapAngle(delta_x0(2));

  Eigen::VectorXd E_vec(Nx);
  for (int k = 0; k < N; ++k) {
    E_vec.segment(k*nx, nx) = warm_x_[k] - x_ref[k];
    E_vec.segment(k*nx, nx)(2) = wrapAngle(E_vec.segment(k*nx, nx)(2));
  }

  Eigen::VectorXd eps = A_batch * delta_x0 + F_batch + E_vec;

  Eigen::VectorXd Ug(Nu);
  for (int k = 0; k < N; ++k) {
    Ug(k*nu+0) = warm_u_[k](0);
    Ug(k*nu+1) = warm_u_[k](1);
  }

  Eigen::MatrixXd H  = B_batch.transpose() * Q * B_batch + R;
  Eigen::VectorXd gv = B_batch.transpose() * Q * eps + R * Ug;
  H = 0.5 * (H + H.transpose());
  H += 1e-8 * Eigen::MatrixXd::Identity(Nu, Nu);

  // ── 제약 행렬 구성 (논문 식 20) ──────────────────────────
  // 블록 1: 입력 박스   (Nu 행)
  // 블록 2: δc 상태 박스 (N  행)
  // 블록 3: 마찰원 선형 (2N 행) - 논문 식(30)
  const int n_box   = Nu;
  const int n_dc    = N;
  const int n_fc    = 2 * N;
  const int n_con   = n_box + n_dc + n_fc;

  Eigen::MatrixXd A_con = Eigen::MatrixXd::Zero(n_con, Nu);
  Eigen::VectorXd l_con = Eigen::VectorXd::Zero(n_con);
  Eigen::VectorXd u_con = Eigen::VectorXd::Zero(n_con);

  // 블록 1: 입력 박스
  A_con.block(0, 0, Nu, Nu) = Eigen::MatrixXd::Identity(Nu, Nu);
  for (int k = 0; k < N; ++k) {
    l_con(k*nu+0) = -vd_max - warm_u_[k](0);
    u_con(k*nu+0) =  vd_max - warm_u_[k](0);
    l_con(k*nu+1) = -T_max  - warm_u_[k](1);
    u_con(k*nu+1) =  T_max  - warm_u_[k](1);
  }

  // 블록 2: δc 상태 제약 (B_batch의 δc 행 추출)
  for (int k = 0; k < N; ++k) {
    A_con.row(n_box + k) = B_batch.row(k*nx + 6);
    const double dc_warm = warm_x_[k+1](6);
    l_con(n_box + k) = -dc_max - dc_warm;
    u_con(n_box + k) =  dc_max - dc_warm;
  }

  // 블록 3: 마찰원 선형 근사 (논문 식 30)
  // G_f ≤ 0, G_r ≤ 0 → Gg + D·δx + E·δu ≤ 0
  const double mu_mg = params_.mu * params_.mass_kg * 9.81;
  const double mu2   = mu_mg * mu_mg;
  for (int k = 0; k < N; ++k) {
    const auto& xg = warm_x_[k];
    const auto   t  = computeTireForces(xg);

    // warm start 힘 크기 (Gg)
    const double Gf_warm = t.Fy_f * t.Fy_f + t.Fx_f * t.Fx_f - mu2;
    const double Gr_warm = t.Fy_r * t.Fy_r + t.Fx_r * t.Fx_r - mu2;

    // D(2×9), E(2×2)
    Eigen::Matrix<double,2,9> D_k;
    Eigen::Matrix<double,2,2> E_k;
    frictionCircleJacobian(xg, t, D_k, E_k);

    // ∂G/∂delta_U = D · B_batch[k*nx:(k+1)*nx, :]
    Eigen::RowVectorXd row_f = D_k.row(0) * B_batch.block(k*nx, 0, nx, Nu);
    Eigen::RowVectorXd row_r = D_k.row(1) * B_batch.block(k*nx, 0, nx, Nu);

    A_con.row(n_box + n_dc + 2*k + 0) = row_f;
    A_con.row(n_box + n_dc + 2*k + 1) = row_r;

    // upper: -Gg (제약 ≤ 0이므로 우변 = -Gg_warm)
    constexpr double kInf = 1e20;
    l_con(n_box + n_dc + 2*k + 0) = -kInf;
    l_con(n_box + n_dc + 2*k + 1) = -kInf;
    u_con(n_box + n_dc + 2*k + 0) = -Gf_warm;
    u_con(n_box + n_dc + 2*k + 1) = -Gr_warm;
  }

  const auto t_model_end = std::chrono::high_resolution_clock::now();
  out.model_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t_model_end - t0).count();

  // ── OSQP ─────────────────────────────────────────────────
  Eigen::SparseMatrix<double> P_sp = H.sparseView();
  Eigen::SparseMatrix<double> A_sp = A_con.sparseView();
  P_sp.makeCompressed(); A_sp.makeCompressed();

  std::vector<c_float> Pd, Av, qv, lv, uv;
  std::vector<c_int>   Pi, Pp, Ai, Ap;

  for (int i = 0; i < P_sp.nonZeros(); ++i) {
    Pd.push_back(static_cast<c_float>(*(P_sp.valuePtr() + i)));
    Pi.push_back(static_cast<c_int>(*(P_sp.innerIndexPtr() + i)));
  }
  for (int i = 0; i <= Nu; ++i) { Pp.push_back(static_cast<c_int>(*(P_sp.outerIndexPtr() + i))); }
  for (int i = 0; i < A_sp.nonZeros(); ++i) {
    Av.push_back(static_cast<c_float>(*(A_sp.valuePtr() + i)));
    Ai.push_back(static_cast<c_int>(*(A_sp.innerIndexPtr() + i)));
  }
  for (int i = 0; i <= Nu; ++i) { Ap.push_back(static_cast<c_int>(*(A_sp.outerIndexPtr() + i))); }
  for (int i = 0; i < Nu; ++i) { qv.push_back(static_cast<c_float>(gv(i))); }
  for (int i = 0; i < n_con; ++i) {
    lv.push_back(static_cast<c_float>(l_con(i)));
    uv.push_back(static_cast<c_float>(u_con(i)));
  }

  OSQPData data{};
  data.n = Nu; data.m = n_con;
  data.P = csc_matrix(Nu, Nu, P_sp.nonZeros(), Pd.data(), Pi.data(), Pp.data());
  data.q = qv.data();
  data.A = csc_matrix(n_con, Nu, A_sp.nonZeros(), Av.data(), Ai.data(), Ap.data());
  data.l = lv.data(); data.u = uv.data();

  OSQPSettings settings{};
  osqp_set_default_settings(&settings);
  settings.verbose  = 0;
  settings.polish   = 1;
  settings.max_iter = 4000;
  settings.eps_abs  = 1e-4;
  settings.eps_rel  = 1e-4;

  const auto t_solver_start = std::chrono::high_resolution_clock::now();
  OSQPWorkspace* work = nullptr;
  if (osqp_setup(&work, &data, &settings) != 0) {
    c_free(data.P); c_free(data.A);
    warm_valid_ = false;
    return out;
  }
  osqp_solve(work);
  const auto t_solver_end = std::chrono::high_resolution_clock::now();
  out.solver_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
      t_solver_end - t_solver_start).count();

  const bool ok = (work->info->status_val == OSQP_SOLVED ||
                   work->info->status_val == OSQP_SOLVED_INACCURATE ||
                   work->info->status_val == OSQP_MAX_ITER_REACHED);

  if (ok) {
    const double du_vd = static_cast<double>(work->solution->x[0]);
    const double du_T  = static_cast<double>(work->solution->x[1]);

    // 최적 절대 입력
    const double vd_opt = std::clamp(warm_u_[0](0) + du_vd, -vd_max, vd_max);
    const double T_opt  = std::clamp(warm_u_[0](1) + du_T,  -T_max,  T_max);

    // δc 갱신 (1스텝 적분)
    const double dc_new = std::clamp(state.delta_c + vd_opt * dt, -dc_max, dc_max);

    out.delta_c_opt  = dc_new;
    out.T_drive_opt  = T_opt;
    out.kappa_cmd    = std::tan(dc_new) / std::max(params_.wheelbase, 0.01);
    out.v_cmd        = std::clamp(std::abs(speed_mps), params_.min_velocity, params_.max_velocity);
    out.omega_cmd    = out.v_cmd * out.kappa_cmd * (speed_mps >= 0.0 ? 1.0 : -1.0);
    out.solved       = true;

    // RTI warm start shift (논문 Algorithm 1)
    std::vector<Eigen::Matrix<double,2,1>> new_u(N);
    std::vector<V9> new_x(N+1);
    new_x[0] = x0;
    for (int i = 0; i < N-1; ++i) {
      new_u[i](0) = std::clamp(warm_u_[i+1](0) + static_cast<double>(work->solution->x[(i+1)*nu+0]), -vd_max, vd_max);
      new_u[i](1) = std::clamp(warm_u_[i+1](1) + static_cast<double>(work->solution->x[(i+1)*nu+1]), -T_max, T_max);
    }
    new_u[N-1] = new_u[N-2];
    for (int k = 0; k < N; ++k) { new_x[k+1] = dbmImplicitEuler(new_x[k], new_u[k]); }
    warm_u_ = new_u;
    warm_x_ = new_x;

    // 예측 궤적
    out.predicted_xy.reserve(N);
    for (int k = 1; k <= N; ++k) {
      out.predicted_xy.push_back({warm_x_[k](0), warm_x_[k](1), 0.0});
    }
  } else {
    warm_valid_ = false;
  }

  osqp_cleanup(work);
  c_free(data.P); c_free(data.A);
  return out;
}

// ============================================================
// MPCControllerCpp (레거시 호환 wrapper)
// ============================================================
MPCControllerCpp::MPCControllerCpp() {
  // DBM 파라미터를 소형 스케일에 맞게 초기화
  dbm_params_ = DBMRTINMPCParams{};
  dbm_ctrl_.setConfig(dbm_params_);
}

void MPCControllerCpp::update_parameters(const LTVMPCParams& p) {
  params_ = p;
  dbm_params_.N            = p.N;
  dbm_params_.Ts           = p.Ts;
  dbm_params_.wheelbase    = p.l;
  dbm_params_.max_velocity = p.max_velocity;
  dbm_params_.min_velocity = p.min_velocity;
  dbm_params_.mu           = p.mu;
  dbm_params_.w_u_delta    = p.wu;
  dbm_params_.w_u_torque   = p.wu;
  dbm_ctrl_.setConfig(dbm_params_);
}

void MPCControllerCpp::reset_state() {
  dbm_ctrl_.reset();
  est_vx_ = est_vy_ = est_omega_ = 0.0;
  est_delta_ = est_omegaf_ = est_omegar_ = 0.0;
}

void MPCControllerCpp::setDBMParams(const DBMRTINMPCParams& p) {
  dbm_params_ = p;
  dbm_ctrl_.setConfig(dbm_params_);
}

void MPCControllerCpp::setEstimatedState(
    double vx, double vy, double omega,
    double delta_c, double omega_f, double omega_r)
{
  est_vx_    = vx;
  est_vy_    = vy;
  est_omega_ = omega;
  est_delta_ = delta_c;
  est_omegaf_ = omega_f;
  est_omegar_ = omega_r;
}

double MPCControllerCpp::quatToYaw(const geometry_msgs::msg::Quaternion& q) {
  const double n = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  auto wrap = [](double a) {
    while (a > kPi)  a -= 2.0*kPi;
    while (a < -kPi) a += 2.0*kPi;
    return a;
  };
  if (!std::isfinite(n) || std::abs(n - 1.0) > 0.15) return wrap(q.z);
  const double x = q.x/n, y = q.y/n, z = q.z/n, w = q.w/n;
  return wrap(std::atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)));
}

ControlOutput MPCControllerCpp::compute_control(
    const geometry_msgs::msg::Pose& current_pose,
    const std::vector<geometry_msgs::msg::PoseStamped>& local_path,
    const CollisionConstraints&)
{
  ControlOutput out;
  if (local_path.empty()) return out;

  DBMState state;
  state.px      = current_pose.position.x;
  state.py      = current_pose.position.y;
  state.psi     = quatToYaw(current_pose.orientation);
  state.vx      = est_vx_;
  state.vy      = est_vy_;
  state.omega   = est_omega_;
  state.delta_c = est_delta_;
  state.omega_f = est_omegaf_;
  state.omega_r = est_omegar_;

  const double speed = (est_vx_ >= 0.0 ? 1.0 : -1.0) *
    std::max(std::abs(est_vx_), params_.min_velocity);

  const DBMCommand cmd = dbm_ctrl_.computeControl(state, local_path, speed);
  if (!cmd.solved) return out;

  out.velocity         = cmd.v_cmd;
  out.angular_velocity = cmd.omega_cmd;
  for (const auto& pt : cmd.predicted_xy) {
    out.predicted_trajectory.push_back(pt);
  }
  return out;
}

}  // namespace bisa