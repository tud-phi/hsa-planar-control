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

SYSTEM_TYPE = "robot"  # "sim" or "robot"
JOY_SIGNAL_SOURCE = "openvibe"  # "openvibe" or "keyboard"

hsa_material = "fpu"
kappa_b_eq = 0.0
sigma_sh_eq = 0.0
sigma_a_eq1, sigma_a_eq2 = 1.0, 1.0
controller_type = "operational_space_pd_plus_nonlinear_actuation"

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
    "sigma_a_eq1": sigma_a_eq1,
    "sigma_a_eq2": sigma_a_eq2,
    "phi_max": phi_max,
}
planning_params = common_params | {
    "planning_frequency": 0.025  # period of 40s between setpoints
}
viz_params = common_params | {"rendering_frequency": 20.0, "invert_colors": True}
joy_control_params = common_params | {
    "cartesian_delta": 1e-4,  # step for moving the attractor [m]
    "pee_y0": 0.11,  # initial y coordinate position of the attractor [m]
}
control_params = common_params | {
    "controller_type": controller_type,
    "control_frequency": 50.0,
    "setpoint_topic": "/attractor",
}
sim_params = None
if SYSTEM_TYPE == "sim":
    sim_params = common_params | {
        "sim_dt": 4e-5,
        "control_frequency": control_params["control_frequency"],
    }
    control_params[
        "present_planar_actuation_topic"
    ] = "/control_input"  # we neglect the actuation dynamics
if controller_type == "basic_operational_space_pid":
    control_params.update(
        {
            "Kp": 1.0e1,  # [rad/m]
            "Ki": 1.1e2,  # [rad/(ms)]
            "Kd": 2.5e-1,  # [rad s/m]
        }
    )
elif controller_type == "operational_space_pd_plus_linearized_actuation":
    if hsa_material == "fpu":
        control_params.update(
            {
                "Kp": 5e1,  # [N/m]
                "Kd": 0e0,  # [N s/m]
            }
        )
    elif hsa_material == "epu":
        control_params.update(
            {
                "Kp": 2e2,  # [N/m]
                "Kd": 1e0,  # [N s/m]
            }
        )
    else:
        raise ValueError(f"Unknown HSA material: {hsa_material}")
elif controller_type == "operational_space_pd_plus_nonlinear_actuation":
    control_params.update(
        {
            "Kp": 3e3,  # [N/m]
            "Kd": 1e0,  # [N s/m]
        }
    )
else:
    raise ValueError(f"Unknown controller type: {controller_type}")


def generate_launch_description():
    launch_actions = [
        Node(
            package="hsa_visualization",
            executable="planar_viz_node",
            name="visualization",
            parameters=[viz_params],
        ),
        Node(
            package="hsa_planar_control",
            executable="model_based_control_node",
            name="model_based_control",
            parameters=[control_params],
        ),
        TimerAction(
            period=40.0,  # delay start of control node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_joy_control",
                    executable="planar_hsa_cartesian_joy_control_node",
                    name="joy_control",
                    parameters=[joy_control_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
        TimerAction(
            period=10.0,  # delay start of setpoint generation node for simulation to be fully compiled and ready
            actions=[
                Node(
                    package="hsa_planar_control",
                    executable="random_setpoints_node",
                    name="random_setpoints_generator",
                    parameters=[planning_params],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ],
        ),
    ]

    if SYSTEM_TYPE == "sim":
        launch_actions.append(
            Node(
                package="hsa_sim",
                executable="planar_sim_node",
                name="simulation",
                parameters=[sim_params],
            ),
        )
    elif SYSTEM_TYPE == "robot":
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
        launch_actions.extend(
            [
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
                    executable="hsa_planar_actuation_by_msg_node",
                    name="actuation",
                    parameters=[
                        common_params,
                        {
                            "present_motor_angles_frequency": control_params[
                                "control_frequency"
                            ]
                        },
                    ],
                    arguments=["--ros-args", "--log-level", LOG_LEVEL],
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown system type {SYSTEM_TYPE}.")

    if JOY_SIGNAL_SOURCE == "openvibe":
        launch_actions.append(
            Node(
                package="joylike_operation",
                executable="openvibe_stimulation_to_joy_node",
                name="openvibe_teleop",
                parameters=[
                    {"joy_control_mode": "cartesian", "host": "145.94.234.212"}
                ],
                arguments=["--ros-args", "--log-level", LOG_LEVEL],
            ),
        )
    elif JOY_SIGNAL_SOURCE == "keyboard":
        keyboard2joy_filepath = os.path.join(
            get_package_share_directory("joylike_operation"),
            "config",
            "keystroke2joy_cartesian.yaml",
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
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH], output="screen"
            )
        )

    return LaunchDescription(launch_actions)
