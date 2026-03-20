#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy, csv, os, math, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, Accel
from nav_msgs.msg import Path
from bisa.msg import LapInfo
from rcl_interfaces.srv import GetParameters


class SpeedDataLogger(Node):
    def __init__(self):
        super().__init__("speed_data_logger")

        self.declare_parameter("cav_id", 1)
        self.declare_parameter("target_laps", 1)
        self.declare_parameter("output_dir", os.path.expanduser("~/speed_logs"))

        self.cav_id   = self.get_parameter("cav_id").value
        self.target_laps = self.get_parameter("target_laps").value
        self.output_dir  = self.get_parameter("output_dir").value

        id_str = f"{self.cav_id:02d}"
        os.makedirs(self.output_dir, exist_ok=True)

        # use_velocity_planner 값 읽기 (파라미터 서비스 호출)
        self.vp_mode = self._get_velocity_planner_mode(id_str)
        label = "velocity_planner" if self.vp_mode else "simple_formula"

        # 파일명에 모드 포함
        ts = time.strftime("%Y%m%d_%H%M%S")
        mode_str = "VP_ON" if self.vp_mode else "VP_OFF"
        self.csv_path = os.path.join(
            self.output_dir,
            f"speed_{mode_str}_{label}_cav{id_str}_{ts}.csv"
        )

        self.pose       = None
        self.local_path = []
        self.v_cmd, self.w_cmd = 0.0, 0.0
        self.lap_count  = 0
        self.logging_active = False
        self.start_lap  = None
        self.rows       = []
        self.t0         = None
        self.label      = label

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(PoseStamped, f"/CAV_{id_str}",      self.pose_cb,  qos)
        self.create_subscription(Accel,       f"/CAV_{id_str}_accel", self.accel_cb, qos)
        self.create_subscription(Path,        f"/cav{id_str}/local_path", self.path_cb, qos)
        self.create_subscription(LapInfo,     f"/cav{id_str}/lap_info",   self.lap_cb,  10)

        self.create_timer(0.05, self.log_tick)

        self.get_logger().info(
            f"SpeedDataLogger 시작: CAV{self.cav_id} | "
            f"VelocityPlanner={'ON' if self.vp_mode else 'OFF'} | "
            f"저장={self.csv_path}"
        )

    def _get_velocity_planner_mode(self, id_str):
        """mpc_tracker_cav01 노드에서 use_velocity_planner 파라미터 읽기"""
        try:
            node_name = f"/mpc_tracker_cav{id_str}"
            client = self.create_client(GetParameters, f"{node_name}/get_parameters")
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn("파라미터 서비스 연결 실패, 기본값 False 사용")
                return False
            req = GetParameters.Request()
            req.names = ["use_velocity_planner"]
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.result() and future.result().values:
                return future.result().values[0].bool_value
        except Exception as e:
            self.get_logger().warn(f"파라미터 읽기 실패: {e}")
        return False

    def pose_cb(self, msg):  self.pose = msg
    def accel_cb(self, msg):
        self.v_cmd = msg.linear.x
        self.w_cmd = msg.angular.z
    def path_cb(self, msg):  self.local_path = msg.poses

    def lap_cb(self, msg):
        if msg.lap_count > self.lap_count:
            self.lap_count = msg.lap_count
            self.get_logger().info(f"랩 완료: {self.lap_count}랩")

            if not self.logging_active:
                self.logging_active = True
                self.start_lap = self.lap_count
                self.t0 = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info("★ 데이터 기록 시작 (lap 0→1)")

            elif self.logging_active and (self.lap_count - self.start_lap) >= self.target_laps:
                self.logging_active = False
                self.save_csv()
                self.get_logger().info(f"★ 기록 완료 → {self.csv_path}")
                raise SystemExit

    def log_tick(self):
        if not self.logging_active or self.pose is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9 - (self.t0 or 0.0)
        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        cte, heading_err, kappa_r = self._compute_path_metrics(px, py)
        self.rows.append({
            "time_s":      round(t, 4),
            "v_cmd":       round(self.v_cmd, 5),
            "w_cmd":       round(self.w_cmd, 5),
            "cte_m":       round(cte, 5),
            "heading_err": round(heading_err, 5),
            "kappa_r":     round(kappa_r, 6),
            "pos_x":       round(px, 4),
            "pos_y":       round(py, 4),
            "lap":         self.lap_count,
            "label":       self.label,
            "vp_mode":     self.vp_mode,
        })

    def _compute_path_metrics(self, ex, ey):
        path = self.local_path
        if len(path) < 3:
            return 0.0, 0.0, 0.0
        min_d, closest = float("inf"), 0
        for i, ps in enumerate(path):
            d = math.hypot(ps.pose.position.x - ex, ps.pose.position.y - ey)
            if d < min_d:
                min_d, closest = d, i
        cte = heading_err = 0.0
        if closest + 1 < len(path):
            p0 = path[closest].pose.position
            p1 = path[closest+1].pose.position
            path_yaw = math.atan2(p1.y - p0.y, p1.x - p0.x)
            rx, ry   = ex - p0.x, ey - p0.y
            cte      = -math.sin(path_yaw)*rx + math.cos(path_yaw)*ry
            q        = self.pose.pose.orientation
            norm     = math.sqrt(q.x**2+q.y**2+q.z**2+q.w**2)
            yaw      = q.z if abs(norm-1.0) > 0.15 else math.atan2(
                2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
            heading_err = path_yaw - yaw
            while heading_err >  math.pi: heading_err -= 2*math.pi
            while heading_err < -math.pi: heading_err += 2*math.pi
        kappa_r = 0.0
        idx = min(closest, len(path)-3)
        x1,y1 = path[idx].pose.position.x,   path[idx].pose.position.y
        x2,y2 = path[idx+1].pose.position.x, path[idx+1].pose.position.y
        x3,y3 = path[idx+2].pose.position.x, path[idx+2].pose.position.y
        a = math.hypot(x2-x1,y2-y1); b = math.hypot(x3-x2,y3-y2)
        c = math.hypot(x3-x1,y3-y1)
        area2 = abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
        if area2 > 1e-9 and a > 1e-6 and b > 1e-6 and c > 1e-6:
            kappa_r = min(2.0*area2/(a*b*c), 3.0)
        return cte, heading_err, kappa_r

    def save_csv(self):
        if not self.rows:
            return
        fieldnames = ["time_s","v_cmd","w_cmd","cte_m","heading_err",
                      "kappa_r","pos_x","pos_y","lap","label","vp_mode"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        self.get_logger().info(f"CSV 저장: {len(self.rows)}행 → {self.csv_path}")


def main(args=None):
    rclpy.init(args=args)
    node = SpeedDataLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        if node.rows and node.logging_active:
            node.save_csv()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()