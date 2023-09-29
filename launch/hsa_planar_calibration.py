"""
Planar HSA calibration launch file
"""
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os


# datetime object containing current date and time
now = datetime.now()

LOG_LEVEL = "warn"

ros_params = {
    "hsa_material": "fpu",
    "payload_mass": 0.0,  # kg
}

def generate_launch_description():
    # Create the NatNet client node
    natnet_config = os.path.join(
        get_package_share_directory("mocap_optitrack_client"),
        "config",
        "natnetclient.yaml",
    )
    # Create the world to base client
    w2b_config = os.path.join(
        get_package_share_directory("hsa_inverse_kinematics"),
        "config",
        "world_to_base_y_up.yaml",
    )

    launch_actions = [
        Node(
            package="mocap_optitrack_client",
            executable="mocap_optitrack_client",
            name="natnet_client",
            parameters=[natnet_config, {"record": False}],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="mocap_optitrack_w2b",
            executable="mocap_optitrack_w2b",
            name="world_to_base",
            parameters=[w2b_config],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_inverse_kinematics",
            executable="planar_cs_ik_node",
            name="inverse_kinematics",
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
            parameters=[ros_params],
        ),
        Node(
            package="hsa_planar_control",
            executable="calibration_node",
            name="calibration",
            parameters=[ros_params],
        ),
    ]

    return LaunchDescription(launch_actions)
