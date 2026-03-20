#include "bisa/mpc_controller_cpp.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace bisa {

// ================================================================
// 수정된 VelocityPlanner 구현부 (mpc_controller_cpp.cpp 내 해당 부분 교체)
// ================================================================

// ── 1. setup() ── 변경 없음, a_lat_max는 vp_ 내부에 포함됨
void VelocityPlanner::setup(const VelocityPlannerParams& vp,
                             double v_max, double v_min,
                             double mu, double g, double Ts) {
    vp_ = vp; v_max_ = v_max; v_min_ = v_min;
    mu_ = mu; g_ = g; Ts_ = Ts;
}

// ── 2. 핵심 변경: 횡가속도 기반 속도 제한 ──────────────────────
// [구] kamm_speed_limit: sqrt(mu*g/kappa)*kamm_margin
//      mu=0.8, g=9.81 → kappa=1일 때 v_lim=2.80 m/s (로봇 최대속도 0.44 훨씬 초과)
//      → 실질적으로 속도 제한 전혀 안 걸림
//
// [신] lateral_speed_limit: sqrt(a_lat_max/kappa)*kamm_margin
//      a_lat_max=0.08 m/s² → kappa=1일 때 v_lim=0.283 m/s (유효 제한!)
//      kappa=0.5 → 0.400, kappa=2.0 → 0.200, kappa=3.0 → 0.163
double VelocityPlanner::lateral_speed_limit(double kappa_r) const {
    const double kappa_abs = std::min(std::abs(kappa_r), 3.0);
    if (kappa_abs < 1e-4) return v_max_;
    // 로봇 스케일 횡가속도 제한: v = sqrt(a_lat_max / kappa)
    const double v_lat = std::sqrt(vp_.a_lat_max / kappa_abs) * vp_.kamm_margin;
    return std::max(v_min_, std::min(v_lat, v_max_));
}

// 하위호환: kamm_speed_limit → lateral_speed_limit 위임
double VelocityPlanner::kamm_speed_limit(double kappa_r) const {
    return lateral_speed_limit(kappa_r);
}

// ── 3. backward_pass ── a_max 파라미터 활용 (구조 동일, 값만 현실화)
// a_max=0.4 m/s², Ts=0.05s → 감속량 = sqrt(v_next² + 2*0.4*0.05) = sqrt(v² + 0.04)
// kappa=1 코너(v_cap=0.283)에서 후방 전파:
//   1스텝 전: sqrt(0.283² + 0.04) = 0.341 m/s
//   2스텝 전: sqrt(0.341² + 0.04) = 0.393 m/s
//   3스텝 전: sqrt(0.393² + 0.04) = 0.441 → v_max 도달
// → 코너 진입 0.15초(3스텝) 전부터 예측 감속 시작 (단순공식과 핵심 차별점)
void VelocityPlanner::backward_pass(std::vector<double>& v_cap) const {
    const int N = static_cast<int>(v_cap.size());
    for (int k = N - 2; k >= 0; --k) {
        // 다음 스텝 속도에서 a_max로 1스텝 역추적 가능한 최대 속도
        const double v_reach = std::sqrt(v_cap[k+1]*v_cap[k+1] + 2.0*vp_.a_max*Ts_);
        v_cap[k] = std::min(v_cap[k], v_reach);
    }
}

// ── 4. forward_pass ── 부드러운 가속 (w_accel 강화로 더 smooth)
void VelocityPlanner::forward_pass(double v_current,
                                    const std::vector<double>& v_cap,
                                    std::vector<double>& v_opt) const {
    const int N = static_cast<int>(v_cap.size());
    v_opt.resize(N);
    double v_prev = v_current;
    const double dv_max = vp_.a_max * Ts_;  // 0.4*0.05 = 0.02 m/s/step
    for (int k = 0; k < N; ++k) {
        // 가속/감속 레이트 제한
        const double v_lo = std::max(v_min_, v_prev - dv_max);
        const double v_hi = std::min(v_cap[k], v_prev + dv_max);
        // 목표 속도를 향해 부드럽게 블렌딩
        // w_accel=10: alpha=1/(1+10)=0.091 → 이전 속도 비중 높아 매우 부드러운 전환
        const double alpha = vp_.w_velocity / (vp_.w_velocity + vp_.w_accel);
        double v_blend = alpha * std::min(v_cap[k], v_max_) + (1.0 - alpha) * v_prev;
        v_blend = std::clamp(v_blend, std::max(v_lo, 0.0), std::max(v_hi, v_lo));
        v_opt[k] = v_blend;
        v_prev   = v_blend;
    }
}

// ── 5. solve() ── 변경 없음 (구조 유지)
std::vector<double> VelocityPlanner::solve(double v_current,
                                            const std::vector<double>& kappa_r_seq) {
    const int N = static_cast<int>(kappa_r_seq.size());
    if (N == 0) return {};
    std::vector<double> v_cap(N);
    // lateral_speed_limit 적용 (kamm_speed_limit 내부에서 호출됨)
    for (int k = 0; k < N; ++k) v_cap[k] = kamm_speed_limit(kappa_r_seq[k]);
    backward_pass(v_cap);
    std::vector<double> v_opt;
    forward_pass(v_current, v_cap, v_opt);
    return v_opt;
}

// ── 6. generate_velocity_profile ── VelocityPlanner 초기화 파라미터 수정
std::vector<double> MPCControllerCpp::generate_velocity_profile(
    const std::vector<geometry_msgs::msg::PoseStamped>& reference_path) {

    const int N = params_.N;
    std::vector<double> v_profile(N, params_.max_velocity);

    if (static_cast<int>(reference_path.size()) < 3) return v_profile;

    // kappa_r_seq 공통 추출 (N스텝 예측 구간 곡률)
    std::vector<double> kappa_r_seq;
    kappa_r_seq.reserve(N);
    for (int k = 0; k < N; ++k) {
        const size_t idx = std::min(static_cast<size_t>(k), reference_path.size() - 3);
        const double x1 = reference_path[idx].pose.position.x;
        const double y1 = reference_path[idx].pose.position.y;
        const double x2 = reference_path[idx+1].pose.position.x;
        const double y2 = reference_path[idx+1].pose.position.y;
        const double x3 = reference_path[idx+2].pose.position.x;
        const double y3 = reference_path[idx+2].pose.position.y;
        kappa_r_seq.push_back(compute_curvature_3points(x1,y1,x2,y2,x3,y3));
    }

    if (params_.use_velocity_planner) {
        // ── VelocityPlanner 경로 ─────────────────────────────────
        VelocityPlannerParams vp;
        vp.w_velocity  = 1.0;
        vp.w_accel     = 10.0;    // 부드러운 속도 전환 (구: 8.0)
        // a_max: 로봇 실제 종방향 가속도 한계
        //   max_accel 파라미터(3.5 m/s²)는 MPC 제어 입력 한계이지
        //   실제 로봇 가속 능력이 아님 → 0.4 m/s²로 현실화
        vp.a_max       = 0.4;
        // a_lat_max: 핵심! 로봇 횡가속도 제한
        //   v_limit = sqrt(a_lat_max / kappa)
        //   0.08 m/s² → kappa≥0.5 구간부터 유효 속도 제한 발생
        vp.a_lat_max   = 0.08;
        vp.kamm_margin = 1.0;     // 별도 마진 불필요 (a_lat_max 자체가 안전값)
        velocity_planner_.setup(vp, params_.max_velocity, params_.min_velocity,
                                 params_.mu, params_.g, params_.Ts);
        v_profile = velocity_planner_.solve(prev_velocity_, kappa_r_seq);
        if (static_cast<int>(v_profile.size()) < N)
            v_profile.resize(N, params_.max_velocity);
    } else {
        // ── 기존 단순 공식 경로 (비교 대조군) ──────────────────────
        // v = v_max / (1 + alpha * |kappa|)
        // 선형 반응식: 현재 곡률에만 즉시 반응, 예측/look-ahead 없음
        const double alpha = 3.0;
        for (int k = 0; k < N; ++k) {
            double v_k = params_.max_velocity / (1.0 + alpha * std::abs(kappa_r_seq[k]));
            v_profile[k] = std::max(params_.min_velocity, std::min(v_k, params_.max_velocity));
        }
    }

    return v_profile;
}

} // namespace bisa
