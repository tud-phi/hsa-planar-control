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
HSA_MATERIAL = "fpu"
END_EFFECTOR_ATTACHED = True  # whether our 3D printed end effector is attached to the HSA platform
JOY_SIGNAL_SOURCE = "keyboard"  # "openvibe" or "keyboard"
JOY_CONTROL_MODE = "cartesian"  # "bending", "cartesian" or "cartesian_switch"
PUSH_BUTTON_MODE = False  # True or False

kappa_b_eq = 0.0
sigma_sh_eq = 0.0
sigma_a_eq = [1.0, 1.0]
controller_type = "operational_space_impedance_control_nonlinear_actuation"

if HSA_MATERIAL == "fpu":
    phi_max = 200 / 180 * np.pi
elif HSA_MATERIAL == "epu":
    phi_max = 270 / 180 * np.pi
else:
    raise ValueError(f"Unknown HSA material: {HSA_MATERIAL}")

# if PUSH_BUTTON_MODE is active, we need to have an end-effector attached
if PUSH_BUTTON_MODE:
    assert END_EFFECTOR_ATTACHED, "PUSH_BUTTON_MODE requires END_EFFECTOR_ATTACHED"

common_params = {
    "hsa_material": HSA_MATERIAL,
    "kappa_b_eq": kappa_b_eq,
    "sigma_sh_eq": sigma_sh_eq,
    "sigma_a_eq": sigma_a_eq,
    "phi_max": phi_max,
    "chiee_off": [0.0, 0.0, 0.0],  # end-effector offset [m]
    "mpl": 0.0,  # payload mass [kg]
    "CoGpl": [0.0, 0.0],  # payload center of gravity [m]
}
if END_EFFECTOR_ATTACHED:
    # the end-effector is moved by 25mm in the y-dir relative to the top surface of the HSA platform
    common_params["chiee_off"] = [0.0, 0.025, 0.0]
    common_params["mpl"] = 0.018  # the end-effector attachment has a mass of 18g
    # the end-effector attachment has a center of gravity of 3.63mm in y-dir from its base.
    # as it has a thickness of 25mm, this is -21.37mm from the top surface (i.e., end-effector position)
    common_params["CoGpl"] = [0.0, -0.02137]

planning_params = common_params | {
    "setpoint_frequency": 0.0166  # period of 60s between setpoints
}
viz_params = common_params | {
    "rendering_frequency": 20.0, 
    "invert_colors": True, 
    "draw_operational_workspace": True,
    "cartesian_switch_state_topic": "cartesian_switch_state" if JOY_CONTROL_MODE == "cartesian_switch" else "None",
}
joy_control_params = common_params | {
    "cartesian_delta": 2e-4,  # step for moving the attractor [m]
    "pee_y0": 0.11 + common_params["chiee_off"][1],  # initial y coordinate position of the attractor [m]
}
control_params = common_params | {
    "controller_type": controller_type,
    "control_frequency": 50.0,
    "setpoint_topic": "/attractor",
}
sim_params = None
if SYSTEM_TYPE == "sim":
    sim_params = common_params | {
        "sim_dt": 6e-5,
        "control_frequency": control_params["control_frequency"],
        "damping_multiplier": 20.0,
    }
    control_params[
        "present_planar_actuation_topic"
    ] = "/control_input"  # we neglect the actuation dynamics
Ki = np.zeros((2, 2))
if controller_type == "basic_operational_space_pid":
    Kp = 1.0e1 * np.eye(2)  # [rad/m]
    Ki = 1.1e2 * np.eye(2)  # [rad/(ms)]
    Kd = 2.5e-1 * np.eye(2)  # [Ns/m]
elif controller_type == "operational_space_pd_plus_linearized_actuation":
    if HSA_MATERIAL == "fpu":
        Kp = 5e0 * np.eye(2)  # [N/m]
        Kd = 0e0 * np.eye(2)  # [Ns/m]
    elif HSA_MATERIAL == "epu":
        Kp = 2e1 * np.eye(2)  # [N/m]
        Kd = 1e-1 * np.eye(2)  # [Ns/m]
    else:
        raise ValueError(f"Unknown HSA material: {HSA_MATERIAL}")
elif controller_type in ["operational_space_pd_plus_nonlinear_actuation", "operational_space_impedance_control_nonlinear_actuation"]:
    if PUSH_BUTTON_MODE:
        push_direction = 90 / 180 * np.pi  # rotation of impedance with respect to the x-axis [rad]
        # local impedance matrix
        Kp_local = np.diag(np.array([5e2, 5e1]))
        Kd_local = np.diag(np.array([1e0, 1e0]))
        # rotation matrix
        rot = np.array([[np.cos(push_direction), -np.sin(push_direction)], [np.sin(push_direction), np.cos(push_direction)]])
        # global impedance matrix
        Kp = rot @ Kp_local @ rot.T
        Kd = rot @ Kd_local @ rot.T
        print("Kp:\n", Kp)
        print("Kd:\n", Kd)
    else:
        Kp = 3e2 * np.eye(2)  # [N/m]
        Kd = 1e0 * np.eye(2)  # [Ns/m]

    if controller_type == "operational_space_impedance_control_nonlinear_actuation":
        # here, we cancel the natural damping. Therefore, we need to increase the Cartesian damping to make things stable
        Kd = 1.5 * np.eye(2)  # [Ns/m]
else:
    raise ValueError(f"Unknown controller type: {controller_type}")
control_params.update(
    {
        "Kp": Kp.flatten().tolist(),
        "Ki": Ki.flatten().tolist(),
        "Kd": Kd.flatten().tolist(),
    }
)


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
            period=105.0,  # delay start of joy control node for computational controller to be fully compiled and ready
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
            period=40.0,  # delay start of setpoint generation node for simulation to be fully compiled and ready
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
                parameters=[sim_params, {"sigma_a_eq": [1.0, 1.0]}],
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

    joylike_operation_params = {
        "joy_control_mode": JOY_CONTROL_MODE,
        "host": "145.94.224.240"
    }
    if JOY_SIGNAL_SOURCE == "openvibe":
        launch_actions.append(
            Node(
                package="joylike_operation",
                executable="openvibe_stimulation_to_joy_node",
                name="openvibe_teleop",
                parameters=[joylike_operation_params],
                arguments=["--ros-args", "--log-level", LOG_LEVEL],
            ),
        )
    elif JOY_SIGNAL_SOURCE == "keyboard":
        joylike_operation_params['config_filepath'] = str(os.path.join(
            get_package_share_directory("joylike_operation"),
            "config",
            f"keystroke2joy_{joylike_operation_params['joy_control_mode']}.yaml",
        ))
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
                    parameters=[joylike_operation_params],
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
