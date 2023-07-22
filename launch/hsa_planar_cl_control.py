"""
Planar HSA control launch file

Command to convert mcap to sqlite3 rosbag:
output_options.yaml:
    output_bags:
    - uri: rosbag2_20230714_135545_sqlite3
      storage_id: sqlite3
      all: true
ros2 bag convert --input /home/mstoelzle/phd/rosbags/rosbag2_20230714_135545 -o output_options.yaml

Convert sqlite3 to csv:
python3 main.py ../rosbag2_20230714_135545_sqlite3/rosbag2_20230714_135545_sqlite3_0.db3

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

RECORD_BAG = True  # Record data to rosbag file
BAG_PATH = f"/home/mstoelzle/phd/rosbags/rosbag2_{now.strftime('%Y%m%d_%H%M%S')}"
LOG_LEVEL = "warn"

phi_max = 200 / 180 * np.pi
sigma_a_eq = 1.0

inverse_kinematics_params = {
    "sigma_a_eq": sigma_a_eq,
}
planning_params = {
    "phi_max": phi_max,
    "sigma_a_eq": sigma_a_eq,
    "setpoint_mode": "image",
    "image_type": "star",
}
control_params = {
    # implemented controllers:
    # - P_satI_D_collocated_form_plus_steady_state_actuation
    # - P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation
    # - P_satI_D_plus_steady_state_actuation
    # - basic_operational_space_pid
    "controller_type": "P_satI_D_collocated_form_plus_steady_state_actuation",
    "Kp": 5.0e-1,  # [-]
    "Ki": 2.3e-1,  # [1/s]
    "Kd": 1.0e-1,  # [s]
    "gamma": 1e2,
    "phi_max": phi_max,
    "sigma_a_eq": sigma_a_eq,
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
            parameters=[natnet_config],
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
            parameters=[inverse_kinematics_params],
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="dynamixel_control",
            executable="sync_read_single_write_node",
            name="dynamixel_control",
            arguments=["--ros-args", "--log-level", LOG_LEVEL],
        ),
        Node(
            package="hsa_planar_control",
            executable="model_based_control_node",
            name="model_based_control",
            parameters=[control_params],
        ),
        TimerAction(
            period=0.0,  # delay start of planning node
            actions=[
                Node(
                    package="hsa_planar_control",
                    executable="static_inversion_planning_node",
                    name="static_inversion_planning",
                    parameters=[planning_params],
                ),
            ],
        ),
    ]

    if RECORD_BAG:
        launch_actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-a", "-o", BAG_PATH, "-s", "sqlite3"],
                output="screen",
            )
        )

    return LaunchDescription(launch_actions)
