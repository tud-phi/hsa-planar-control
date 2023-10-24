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

HSA_MATERIAL = "fpu"
END_EFFECTOR_ATTACHED = True  # whether our 3D printed end effector is attached to the HSA platform


ros_params = {
    "hsa_material": HSA_MATERIAL,
    "mpl": 0.0,  # payload mass [kg]
}
if END_EFFECTOR_ATTACHED:
    # the end-effector is moved by 25mm in the y-dir relative to the top surface of the HSA platform
    ros_params["chiee_off"] = [0.0, 0.025, 0.0]
    ros_params["mpl"] = 0.018  # the end-effector attachment has a mass of 18g
    # the end-effector attachment has a center of gravity of 3.63mm in y-dir from its base.
    # as it has a thickness of 25mm, this is -21.37mm from the top surface (i.e., end-effector position)
    ros_params["CoGpl"] = [0.0, -0.02137]


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
