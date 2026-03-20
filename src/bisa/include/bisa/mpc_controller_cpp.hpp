#ifndef BISA_MPC_CONTROLLER_CPP_HPP_
#define BISA_MPC_CONTROLLER_CPP_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <osqp/osqp.h>
#include <vector>
#include <array>
#include <cmath>
#include <map>

namespace bisa {

struct LTVMPCParams {
    // === 예측 파라미터 ===
    int N = 20;
    double Ts = 0.2;
    int NS = 4;

    // === 차량 파라미터 ===
    double l = 0.33;
    double vehicle_width = 0.15;

    // === Equation (9): 입력 제약 ===
    double u_min = -2.0;
    double u_max = 2.0;

    // === 곡률 제약 ===
    double kappa_min_delta = -0.25;
    double kappa_max_delta = 0.25;

    // === Equation (10): 비용 함수 가중치 ===
    double wd = 10.0;
    double wtheta = 5.0;
    double wkappa = 2.0;
    double wu = 0.5;

    // === Equation (21): Slack variable 가중치 ===
    double k1_upper = 1e5;
    double k1_lower = 1e5;
    double k2_upper = 1e6;
    double k2_lower = 1e6;

    // === 마찰 파라미터 (하위호환용, VelocityPlanner에서 직접 사용 안함) ===
    double mu = 0.8;
    double g = 9.81;

    // === 속도 제한 ===
    double max_velocity = 3.0;
    double min_velocity = 0.5;

    // === 속도 플래너 선택 ===
    bool use_velocity_planner = true;
};

// ===================================================================
// 속도 최적화 솔버 파라미터
// ===================================================================
struct VelocityPlannerParams {
    double w_velocity   = 1.0;
    double w_accel      = 10.0;  // 강화된 스무딩 (구: 8.0)
    double w_jerk       = 2.0;
    // ── 핵심 변경 ────────────────────────────────────────────────
    // a_max: 로봇 종방향 가속/감속 한계 [m/s²]
    //   구: 2.0 (비물리적 — 로봇 실제 가속도보다 훨씬 큼)
    //   신: 0.4 (현실적 값, backward-pass가 코너 수 스텝 전부터 감속 시작)
    double a_max        = 0.4;
    // a_lat_max: 로봇 횡방향 가속도 한계 [m/s²]
    //   Kamm's circle(mu*g) 대체 → 속도 제한 식: v_lim = sqrt(a_lat_max / kappa)
    //   0.08 설정 시 kappa=1.0 → 0.283 m/s, kappa=0.5 → 0.400 m/s (유효 제한 발생)
    //   구: mu*g = 0.8*9.81 = 7.848 → kappa=1.0에서도 v_lim=2.8 m/s (전혀 제한 안됨)
    double a_lat_max    = 0.08;
    double kamm_margin  = 1.0;   // a_lat_max에 대한 안전 마진 스케일
};

// ===================================================================
// VelocityPlanner: LTV-MPC v_bar 공급용 독립 속도 최적화 솔버
// ===================================================================
class VelocityPlanner {
public:
    VelocityPlanner() = default;
    void setup(const VelocityPlannerParams& vp,
               double v_max, double v_min,
               double mu, double g, double Ts);
    std::vector<double> solve(double v_current,
                              const std::vector<double>& kappa_r_seq);
private:
    VelocityPlannerParams vp_;
    double v_max_ = 1.0, v_min_ = 0.1;
    double mu_ = 0.8, g_ = 9.81, Ts_ = 0.05;

    // 핵심 변경: Kamm's circle 대신 횡가속도 제한 사용
    double lateral_speed_limit(double kappa_r) const;

    // 하위호환용 (내부에서 lateral_speed_limit 호출)
    double kamm_speed_limit(double kappa_r) const;

    void backward_pass(std::vector<double>& v_cap) const;
    void forward_pass(double v_current,
                      const std::vector<double>& v_cap,
                      std::vector<double>& v_opt) const;
};

struct ControlOutput {
    double velocity;
    double angular_velocity;
    std::vector<std::array<double, 3>> predicted_trajectory;
};

struct CollisionConstraints {
    std::map<std::string, double> bounds;
};

class MPCControllerCpp {
public:
    MPCControllerCpp();
    ~MPCControllerCpp();

    void update_parameters(const LTVMPCParams& params);
    void reset_state();

    ControlOutput compute_control(
        const geometry_msgs::msg::Pose& current_pose,
        const std::vector<geometry_msgs::msg::PoseStamped>& local_path,
        const CollisionConstraints& constraints = CollisionConstraints()
    );

private:
    LTVMPCParams params_;
    double current_kappa_ = 0.0;
    double prev_velocity_  = 0.0;
    VelocityPlanner velocity_planner_;

    void compute_system_matrices_continuous(double v, Eigen::MatrixXd& Ac,
        Eigen::VectorXd& Bc, Eigen::VectorXd& Ec);
    void compute_output_matrix(Eigen::MatrixXd& C);
    void discretize_system(const Eigen::MatrixXd& Ac, const Eigen::VectorXd& Bc,
        const Eigen::VectorXd& Ec, double Ts, Eigen::MatrixXd& A,
        Eigen::MatrixXd& B, Eigen::MatrixXd& E);
    void build_prediction_matrices(const std::vector<double>& v_profile,
        Eigen::MatrixXd& A_bar, Eigen::MatrixXd& B_bar,
        Eigen::MatrixXd& E_bar, Eigen::MatrixXd& C_bar);
    void compute_cost_matrices(const std::vector<double>& v_profile,
        Eigen::MatrixXd& Q_bar, Eigen::MatrixXd& R_bar);
    bool formulate_qp(const Eigen::VectorXd& x0, const Eigen::VectorXd& z,
        const std::vector<double>& v_profile,
        const CollisionConstraints& constraints,
        Eigen::SparseMatrix<double>& P, Eigen::VectorXd& q,
        Eigen::SparseMatrix<double>& A_qp, Eigen::VectorXd& l,
        Eigen::VectorXd& u);
    bool solve_qp_osqp(const Eigen::SparseMatrix<double>& P,
        const Eigen::VectorXd& q, const Eigen::SparseMatrix<double>& A,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        Eigen::VectorXd& solution);
    Eigen::VectorXd compute_state_vector(const geometry_msgs::msg::Pose& current_pose,
        const std::vector<geometry_msgs::msg::PoseStamped>& reference_path);
    Eigen::VectorXd compute_disturbance_signal(
        const std::vector<geometry_msgs::msg::PoseStamped>& reference_path);
    std::vector<double> generate_velocity_profile(
        const std::vector<geometry_msgs::msg::PoseStamped>& reference_path);
    void compute_curvature_limits(double v, double& kappa_min, double& kappa_max);
    double compute_curvature_3points(double x1, double y1, double x2, double y2,
        double x3, double y3);
    double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q);
    int find_closest_waypoint(const geometry_msgs::msg::Pose& current_pose,
        const std::vector<geometry_msgs::msg::PoseStamped>& path);
    Eigen::SparseMatrix<double> dense_to_sparse(const Eigen::MatrixXd& dense);
};

}  // namespace bisa

#endif  // BISA_MPC_CONTROLLER_CPP_HPP_