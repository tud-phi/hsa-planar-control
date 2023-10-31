# Planar HSA control launch file
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
BAG_PATH = f"/home/mstoelzle/phd/rosbags/rosbag2_{now.strftime('%Y%m%d_%H%M%S')}"
LOG_LEVEL = "warn"

JOY_SIGNAL_SOURCE = "openvibe"  # "openvibe" or "keyboard"

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
    "phi_delta": np.pi / 250,  # step for each stimulation [rad]
    "phi_max": phi_max,
}
planning_params = common_params | {
    "planning_frequency": 0.025  # period of 40s between setpoints
}
viz_params = common_params | {"rendering_frequency": 20.0, "invert_colors": True}


def generate_launch_description():
    launch_actions = [
        Node(
            package="hsa_sim",
            executable="planar_sim_node",
            name="simulation",
            parameters=[common_params],
        ),
        Node(
            package="hsa_visualization",
            executable="planar_viz_node",
            name="visualization",
            parameters=[viz_params],
        ),
        TimerAction(
            period=40.0,  # delay start of control node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_joy_control",
                    executable="planar_hsa_bending_joy_control_node",
                    name="joy_control",
                    parameters=[common_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
        TimerAction(
            period=10.0,  # delay start of planner node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_trajectory_planner",
                    executable="planar_bending_trajectory_node",
                    name="trajectory_planner",
                    parameters=[planning_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
    ]

    if JOY_SIGNAL_SOURCE == "openvibe":
        launch_actions.append(
            Node(
                package="joylike_operation",
                executable="openvibe_stimulation_to_joy_node",
                name="openvibe_teleop",
                parameters=[{"joy_control_mode": "bending", "host": "145.94.234.212"}],
                arguments=["--ros-args", "--log-level", LOG_LEVEL],
            ),
        )
    elif JOY_SIGNAL_SOURCE == "keyboard":
        keyboard2joy_filepath = os.path.join(
            get_package_share_directory("joylike_operation"),
            "config",
            "keystroke2joy_bending.yaml",
        )
        launch_actions.extend(
            [
                Node(
                    package="keyboard",
                    executable="keyboard",
                    name="keyboard",
                ),
                Node(
                    package="joylike_operation",
                    executable="keyboard_to_joy_node",
                    name="keyboard_teleop",
                    parameters=[{"config_filepath": str(keyboard2joy_filepath)}],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown joy signal source: {JOY_SIGNAL_SOURCE}")

    if RECORD:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH, "-s", "sqlite3"], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
