"""
Planar HSA motor babbling launch file
"""
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import numpy as np
import os


# datetime object containing current date and time
now = datetime.now()

RECORD = True  # Record data to rosbag file
BAG_PATH = f"/home/gfranzese/Documents/sources/rosbags/rosbag2_{now.strftime('%Y%m%d_%H%M%S')}"
LOG_LEVEL = "warn"

hsa_material = "fpu"
kappa_b_eq = 0.0
sigma_sh_eq = 0.0
sigma_a_eq = [1.0, 1.0]

if hsa_material == "fpu":
    phi_max = 200 / 180 * np.pi
elif hsa_material == "epu":
    phi_max = 270 / 180 * np.pi
else:
    raise ValueError(f"Unknown HSA material: {hsa_material}")

common_params = {
    "hsa_material": hsa_material,
    "kappa_b_eq": kappa_b_eq,
    "sigma_sh_eq": sigma_sh_eq,
    "sigma_a_eq": sigma_a_eq,
    "phi_max": phi_max,
}
motor_babbling_params = common_params | {
    "mode": "sinusoidal_extension",
    "duration": 60.0,
    "frequency": 0.2
}


def generate_launch_description():
    # Create the NatNet client node
    natnet_config_path = os.path.join(
        get_package_share_directory("mocap_optitrack_client"),
        "config",
        "natnetclient.yaml",
    )
    # Create the world to base client
    w2b_config_path = os.path.join(
        get_package_share_directory("hsa_inverse_kinematics"),
        "config",
        "world_to_base_y_up.yaml",
    )

    launch_actions = [
        Node(
            package="mocap_optitrack_client",
            executable="mocap_optitrack_client",
            name="natnet_client",
            parameters=[natnet_config_path, {"record": RECORD}],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="mocap_optitrack_w2b",
            executable="mocap_optitrack_w2b",
            name="world_to_base",
            parameters=[w2b_config_path],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_inverse_kinematics",
            executable="planar_cs_ik_node",
            name="inverse_kinematics",
            parameters=[common_params],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_velocity_estimation",
            executable="planar_hsa_velocity_estimator_node",
            name="velocity_estimator",
            parameters=[common_params],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="dynamixel_control",
            executable="sync_read_single_write_node",
            name="dynamixel_control",
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_actuation",
            executable="planar_motor_babbling_node",
            name="motor_babbling",
            parameters=[motor_babbling_params],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        # Node(
        #     package="hsa_visualization",
        #     executable="planar_viz_node",
        #     name="visualization",
        #     parameters=[common_params],
        # ),
    ]

    if RECORD:
        launch_actions.append(
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-a",
                    "-o",
                    BAG_PATH,
                    "-s",
                    "sqlite3",
                ],
                output="screen",
            )
        )

    return LaunchDescription(launch_actions)
