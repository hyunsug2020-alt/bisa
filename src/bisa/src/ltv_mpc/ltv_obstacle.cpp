#include "ltv_obstacle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bisa {

double LTVObstacle::wrapAngle(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double LTVObstacle::quatToYaw(const geometry_msgs::msg::Quaternion& q) {
  const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  const bool valid_quat =
      std::isfinite(n) && std::abs(n - 1.0) <= 0.15 &&
      std::abs(q.x) <= 1.0 + 1e-3 && std::abs(q.y) <= 1.0 + 1e-3 &&
      std::abs(q.z) <= 1.0 + 1e-3 && std::abs(q.w) <= 1.0 + 1e-3;

  if (!valid_quat) {
    double yaw = q.z;
    if (!std::isfinite(yaw)) return 0.0;
    if (std::abs(yaw) > 2.0 * M_PI + 0.5) yaw *= M_PI / 180.0;
    return wrapAngle(yaw);
  }

  const double x = q.x / n;
  const double y = q.y / n;
  const double z = q.z / n;
  const double w = q.w / n;
  return wrapAngle(std::atan2(2.0 * (w * z + x * y),
                              1.0 - 2.0 * (y * y + z * z)));
}

double LTVObstacle::sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

LTVObstacle::LTVObstacle(const ObstacleAvoidanceConfig& cfg) : cfg_(cfg) {}

void LTVObstacle::setConfig(const ObstacleAvoidanceConfig& cfg) {
  cfg_ = cfg;
}

FrenetObstacle LTVObstacle::projectToFrenet(
    const geometry_msgs::msg::Pose& ego_pose,
    const geometry_msgs::msg::Pose& obs_pose,
    const std::vector<geometry_msgs::msg::PoseStamped>& path) const {
  FrenetObstacle result;
  if (path.size() < 3) return result;

  const int n = static_cast<int>(path.size());
  const double ox = obs_pose.position.x;
  const double oy = obs_pose.position.y;
  const double ex = ego_pose.position.x;
  const double ey = ego_pose.position.y;

  int ego_idx = 0;
  double best_ego_d2 = std::numeric_limits<double>::max();
  const int ego_search = std::min(n, 80);
  for (int i = 0; i < ego_search; ++i) {
    const double dx = path[i].pose.position.x - ex;
    const double dy = path[i].pose.position.y - ey;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_ego_d2) {
      best_ego_d2 = d2;
      ego_idx = i;
    }
  }

  int obs_idx = ego_idx;
  double best_obs_d2 = std::numeric_limits<double>::max();
  double s_scan = 0.0;
  for (int i = ego_idx; i < n; ++i) {
    const double dx = path[i].pose.position.x - ox;
    const double dy = path[i].pose.position.y - oy;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_obs_d2) {
      best_obs_d2 = d2;
      obs_idx = i;
    }
    if (i + 1 < n) {
      const double dsx = path[i + 1].pose.position.x - path[i].pose.position.x;
      const double dsy = path[i + 1].pose.position.y - path[i].pose.position.y;
      s_scan += std::hypot(dsx, dsy);
    }
    if (s_scan > cfg_.detect_range_s * 1.5) break;
  }

  if (obs_idx <= ego_idx) return result;

  const double max_lateral_proximity =
      std::max(std::abs(cfg_.road_d_upper), std::abs(cfg_.road_d_lower)) +
      0.5 * cfg_.obs_body_width + 0.10;
  if (best_obs_d2 > max_lateral_proximity * max_lateral_proximity) {
    return result;
  }

  double s_along = 0.0;
  for (int i = ego_idx; i < obs_idx && i + 1 < n; ++i) {
    const double dx = path[i + 1].pose.position.x - path[i].pose.position.x;
    const double dy = path[i + 1].pose.position.y - path[i].pose.position.y;
    s_along += std::hypot(dx, dy);
  }

  if (s_along > cfg_.detect_range_s + cfg_.obs_body_length) return result;

  const int i0 = obs_idx;
  const int i1 = (obs_idx + 1 < n) ? obs_idx + 1 : obs_idx - 1;
  if (i1 < 0 || i1 >= n || i1 == i0) return result;

  const double tx = path[i1].pose.position.x - path[i0].pose.position.x;
  const double ty = path[i1].pose.position.y - path[i0].pose.position.y;
  const double tnorm = std::hypot(tx, ty);
  if (tnorm < 1e-6) return result;

  const double tnx = tx / tnorm;
  const double tny = ty / tnorm;
  const double vx = ox - path[i0].pose.position.x;
  const double vy = oy - path[i0].pose.position.y;
  const double d_center = -tny * vx + tnx * vy;

  const double d_reject_upper =
      cfg_.road_d_upper * 1.5 + 0.5 * cfg_.obs_body_width;
  const double d_reject_lower =
      cfg_.road_d_lower * 1.5 - 0.5 * cfg_.obs_body_width;
  if (d_center > d_reject_upper || d_center < d_reject_lower) {
    return result;
  }

  const double path_yaw = std::atan2(ty, tx);
  const double obs_yaw = quatToYaw(obs_pose.orientation);
  const double delta = wrapAngle(obs_yaw - path_yaw);

  const double half_l = 0.5 * cfg_.obs_body_length;
  const double half_w = 0.5 * cfg_.obs_body_width;
  const double s_extent = std::abs(std::cos(delta)) * half_l +
                          std::abs(std::sin(delta)) * half_w +
                          cfg_.obs_s_margin;
  const double d_extent = std::abs(std::sin(delta)) * half_l +
                          std::abs(std::cos(delta)) * half_w;
  const double s_rear = s_along - s_extent;
  const double s_front = s_along + s_extent;

  if (s_front < -0.10) return result;
  if (s_rear > cfg_.detect_range_s) return result;

  result.detected = true;
  result.s_obs = s_along;
  result.d_obs = d_center;
  result.d_min = d_center - d_extent;
  result.d_max = d_center + d_extent;
  result.s_rear = s_rear;
  result.s_front = s_front;
  result.d_extent = d_extent;
  result.s_extent = s_extent;
  return result;
}

bool LTVObstacle::chooseAvoidRight(
    const FrenetObstacle& obs,
    double d_upper,
    double d_lower) const {
  if (obs.d_max <= 0.0) return false;
  if (obs.d_min >= 0.0) return true;

  const double right_free = (obs.d_min - cfg_.d_safe) - d_lower;
  const double left_free = d_upper - (obs.d_max + cfg_.d_safe);

  if (right_free > left_free + 1e-3) return true;
  if (left_free > right_free + 1e-3) return false;
  return cfg_.preferred_sign >= 0.0;
}

void LTVObstacle::computeTimeVaryingBounds(
    const FrenetObstacle& obs,
    const Eigen::VectorXd& v_bar,
    double Ts,
    int N,
    std::vector<double>& d_upper_seq,
    std::vector<double>& d_lower_seq) const {
  d_upper_seq.assign(N, cfg_.road_d_upper);
  d_lower_seq.assign(N, cfg_.road_d_lower);
  if (!obs.detected) return;

  const bool avoid_right =
      chooseAvoidRight(obs, cfg_.road_d_upper, cfg_.road_d_lower);
  const double sigma = std::max(0.2, cfg_.activation_sigma_s);
  const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
  double s_pred = 0.0;

  for (int k = 0; k < N; ++k) {
    if (k > 0) {
      const double vk =
          (k - 1 < v_bar.size()) ? v_bar(k - 1) : v_bar(v_bar.size() - 1);
      s_pred += std::max(0.05, vk) * Ts;
    }

    const double ds_back = s_pred - obs.s_rear;
    const double ds_front = obs.s_front - s_pred;
    const double gate_back =
        sigmoid((ds_back + cfg_.activation_back_s) / cfg_.activation_ramp_s);
    const double gate_front =
        sigmoid((ds_front + cfg_.activation_front_s) / cfg_.activation_ramp_s);
    const double gate = gate_back * gate_front;

    double ds_interval = 0.0;
    if (s_pred < obs.s_rear) {
      ds_interval = obs.s_rear - s_pred;
    } else if (s_pred > obs.s_front) {
      ds_interval = s_pred - obs.s_front;
    }

    const double w = gate * std::exp(-ds_interval * ds_interval * inv_2sigma2);
    const double dist_scale = std::clamp(
        1.0 - 0.6 * (ds_interval / std::max(cfg_.detect_range_s, 0.5)), 0.4,
        1.0);
    const double d_safe_eff =
        (cfg_.d_safe + cfg_.extra_push * gate) * dist_scale;

    if (obs.d_min >= 0.0) {
      const double target =
          std::min(cfg_.road_d_upper, obs.d_min - d_safe_eff);
      d_upper_seq[k] = (1.0 - w) * cfg_.road_d_upper + w * target;
    } else if (obs.d_max <= 0.0) {
      const double target =
          std::max(cfg_.road_d_lower, obs.d_max + d_safe_eff);
      d_lower_seq[k] = (1.0 - w) * cfg_.road_d_lower + w * target;
    } else {
      if (avoid_right) {
        const double target =
            std::min(cfg_.road_d_upper, obs.d_min - d_safe_eff);
        d_upper_seq[k] = (1.0 - w) * cfg_.road_d_upper + w * target;
      } else {
        const double target =
            std::max(cfg_.road_d_lower, obs.d_max + d_safe_eff);
        d_lower_seq[k] = (1.0 - w) * cfg_.road_d_lower + w * target;
      }
    }

    if (d_upper_seq[k] < d_lower_seq[k] + 0.03) {
      const double mid = 0.5 * (d_upper_seq[k] + d_lower_seq[k]);
      d_upper_seq[k] = mid + 0.015;
      d_lower_seq[k] = mid - 0.015;
    }
  }
}

void LTVObstacle::buildAsymmetricLateralConstraints(
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
    int nvar) const {
  (void)w_slack_lin;
  (void)w_slack_quad;

  const int N = static_cast<int>(d_upper_seq.size());
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

}  // namespace bisa
