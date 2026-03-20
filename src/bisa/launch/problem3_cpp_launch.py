#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# ==============================================================================
# [1] 차량별 경로(Node Sequence) 데이터베이스 (CAV용)
# ==============================================================================
CAV_PATH_SETTINGS = [
    [21, 51, 46, 40, 43, 9, 56, 59, 18, 21],  # CAV 1
    [60, 52, 24, 37, 39, 49, 55, 12, 15, 60], # CAV 2
    [19, 22, 25, 36, 38, 48, 58, 19],         # CAV 3
    [16, 61, 47, 41, 42, 8, 11, 16],          # CAV 4
]

def load_yaml_file(file_path):
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {}


def resolve_mpc_node_params(full_config, cav_id, slot_index):
    id_str = f"{int(cav_id):02d}"
    slot_str = f"{int(slot_index) + 1:02d}"
    yaml_section_name = f"mpc_tracker_cav{id_str}"
    yaml_section_fallback = f"mpc_tracker_cav{slot_str}"
    if yaml_section_name in full_config:
        return full_config[yaml_section_name].get("ros__parameters", {})
    if yaml_section_fallback in full_config:
        return full_config[yaml_section_fallback].get("ros__parameters", {})
    return {}

def generate_launch_description():
    pkg_dir = get_package_share_directory("bisa")

    config_file = os.path.join(pkg_dir, "config", "cav_config.yaml")
    rviz_config = os.path.join(pkg_dir, "rviz", "bisa.rviz")

    full_config = load_yaml_file(config_file)

    try:
        ros_params_dict = full_config["/**"]["ros__parameters"]
    except KeyError:
        ros_params_dict = {"cav_ids": [1, 2, 3, 4]}

    hv_settings = full_config.get("hv_settings", [])

    nodes = []

    # ---------------------------------------------------------
    # RViz2
    # ---------------------------------------------------------
    nodes.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            additional_env={"ROS_DOMAIN_ID": "100"},
        )
    )

    # ---------------------------------------------------------
    # HDMap Visualizer
    # ---------------------------------------------------------
    nodes.append(
        Node(
            package="bisa",
            executable="hdmap_visualizer.py",
            name="hdmap_visualizer",
            output="screen",
            additional_env={"ROS_DOMAIN_ID": "100"},
        )
    )

    # ---------------------------------------------------------
    # HV 차량
    # ---------------------------------------------------------
    for hv in hv_settings:
        hv_id = hv["id"]
        hv_path = hv["node_sequence"]
        nodes.append(
            Node(
                package="bisa",
                executable="global_path_pub_multi.py",
                name=f"global_path_pub_hv{hv_id}",
                output="screen",
                parameters=[
                    {"cav_id": hv_id, "node_sequence": hv_path, "rviz_slot": -1}
                ],
                remappings=[("/user_global_path", f"/hv{hv_id}/global_path")],
                additional_env={"ROS_DOMAIN_ID": str(hv_id)},
            )
        )

    # ---------------------------------------------------------
    # 동적 CAV 차량 노드 생성
    # ---------------------------------------------------------
    active_ids = ros_params_dict.get("cav_ids", [1, 2, 3, 4])

    # Runtime monitor GUIs
    nodes.append(
        Node(
            package="bisa",
            executable="cav_runtime_log_gui.py",
            name="cav_runtime_log_gui",
            output="screen",
            parameters=[{"cav_ids": active_ids}],
            additional_env={"ROS_DOMAIN_ID": "100"},
        )
    )
    nodes.append(
        Node(
            package="bisa",
            executable="cav_path_error_gui.py",
            name="cav_path_error_gui",
            output="screen",
            parameters=[
                {
                    "cav_ids": active_ids,
                    "focus_cav_id": 1,
                    "gui_frame_ms": 30,
                    "update_period_sec": 0.20,
                    "error_history_sec": 45.0,
                    "overshoot_error_m_threshold": 0.08,
                }
            ],
            additional_env={"ROS_DOMAIN_ID": "100"},
        )
    )

    loop_count = min(len(active_ids), len(CAV_PATH_SETTINGS))
    for index in range(loop_count):
        cav_id = active_ids[index]
        node_seq = CAV_PATH_SETTINGS[index]
        node_params = resolve_mpc_node_params(full_config, cav_id, index)
        id_str = f"{cav_id:02d}"
        rviz_slot = index
        cav_prefix = f"/cav{id_str}"

        # (A) Global Path Publisher
        nodes.append(
            Node(
                package="bisa",
                executable="global_path_pub_multi.py",
                name=f"global_path_pub_cav{id_str}",
                output="screen",
                parameters=[
                    {
                        "cav_id": cav_id,
                        "node_sequence": node_seq,
                        "rviz_slot": rviz_slot,
                    }
                ],
                remappings=[
                    ("/user_global_path", f"{cav_prefix}/global_path"),
                ],
                additional_env={"ROS_DOMAIN_ID": "100"},
            )
        )

        # (B) Local Path Publisher
        nodes.append(
            Node(
                package="bisa",
                executable="local_path_pub_cpp",
                name=f"local_path_pub_cav{id_str}",
                output="screen",
                parameters=[
                    ros_params_dict,
                    {
                        "target_cav_id": cav_id,
                        "rviz_slot": rviz_slot,
                        "local_path_size": 220,
                    },
                ],
                remappings=[
                    (f"/user_global_path_cav{id_str}", f"{cav_prefix}/global_path"),
                    (f"/local_path_cav{id_str}", f"{cav_prefix}/local_path"),
                    (f"/car_marker_{id_str}", f"{cav_prefix}/car_marker"),
                    (f"/lap_info_cav{id_str}", f"{cav_prefix}/lap_info"),
                ],
                additional_env={"ROS_DOMAIN_ID": "100"},
            )
        )

        # (C) MPC Path Tracker - 직접 sim 토픽으로 출력
        nodes.append(
            Node(
                package="bisa",
                executable="mpc_path_tracker_cpp",
                name=f"mpc_tracker_cav{id_str}",
                output="screen",
                parameters=[node_params, {"target_cav_id": cav_id, "publish_accel_cmd": True}],
                remappings=[
                    ("/local_path", f"{cav_prefix}/local_path"),
                    ("/Ego_pose", f"/CAV_{id_str}"),
                    ("/Accel", f"/CAV_{id_str}_accel"),
                    ("/mpc_predicted_path", f"{cav_prefix}/mpc_predicted_path"),
                    ("/mpc_performance", f"{cav_prefix}/mpc_performance"),
                ],
                additional_env={"ROS_DOMAIN_ID": "100"},
            )
        )

    return LaunchDescription(nodes)
