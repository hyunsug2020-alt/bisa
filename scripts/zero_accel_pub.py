#!/usr/bin/env python3

from typing import List

import rclpy
from geometry_msgs.msg import Accel
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class ZeroAccelPublisher(Node):
    def __init__(self) -> None:
        super().__init__("zero_accel_pub")

        self.declare_parameter("cav_ids", [2, 3, 4])
        self.declare_parameter("publish_hz", 20.0)

        cav_ids_param = self.get_parameter("cav_ids").get_parameter_value().integer_array_value
        self.cav_ids: List[int] = [int(v) for v in cav_ids_param] if cav_ids_param else [2, 3, 4]
        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publishers = {
            cav_id: self.create_publisher(Accel, f"/sim/cav{cav_id:02d}/accel", qos)
            for cav_id in self.cav_ids
        }

        self.zero_msg = Accel()
        self.zero_msg.linear.x = 0.0
        self.zero_msg.angular.z = 0.0

        self.timer = self.create_timer(1.0 / publish_hz, self._publish_zero)
        cav_list = ", ".join(f"CAV{cav_id:02d}" for cav_id in self.cav_ids)
        self.get_logger().info(f"Publishing zero accel to: {cav_list} at {publish_hz:.1f} Hz")

    def _publish_zero(self) -> None:
        for pub in self.publishers.values():
            pub.publish(self.zero_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ZeroAccelPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
