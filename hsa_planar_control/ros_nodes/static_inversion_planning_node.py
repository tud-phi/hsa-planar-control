from ament_index_python.packages import get_package_share_directory
import derivative
from example_interfaces.msg import Float64MultiArray
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path
import os

from hsa_control_interfaces.msg import PlanarSetpoint
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
    statically_invert_actuation_to_task_space_scipy_rootfinding,
    statically_invert_actuation_to_task_space_projected_descent,
)
from hsa_planar_control.planning.task_space_trajectory_generation import (
    generate_task_space_trajectory_from_image,
)


class StaticInversionPlanningNode(Node):
    def __init__(self):
        super().__init__("static_inversion_planning_node")

        self.declare_parameter("setpoint_topic", "setpoint")
        self.pub = self.create_publisher(
            PlanarSetpoint, self.get_parameter("setpoint_topic").value, 10
        )

        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_hsa_ns-1_nrs-2.dill"
        )
        (
            _,
            _,
            _,
            inverse_kinematics_end_effector_fn,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.params = PARAMS_CONTROL.copy()

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(
            self.params["sigma_a_eq"]
        )

        self.declare_parameter("phi_max", self.params["phi_max"].mean().item())
        self.params["phi_max"] = self.get_parameter("phi_max").value * jnp.ones_like(
            self.params["phi_max"]
        )

        # define residual function for static inversion optimization
        self.residual_fn = jit(
            static_inversion_factory(
                self.params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
            )
        )

        self.declare_parameter("setpoint_mode", "manual")
        setpoint_mode = self.get_parameter("setpoint_mode").value

        if setpoint_mode == "manual":
            # use accurate but slow scipy rootfinding for manual setpoints
            self.planning_fn = partial(
                statically_invert_actuation_to_task_space_scipy_rootfinding,
                params=self.params,
                residual_fn=self.residual_fn,
                inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
                maxiter=10000,
                verbose=False,
            )

            default_planning_frequency = 0.1
            # desired end-effector positions
            self.pee_des_sps = jnp.array(
                [
                    [0.0, 0.120],
                    [+0.00479247, 0.12781018],
                    [-0.035, 0.122],
                    [-0.00782133, 0.13024847],
                    [0.00823294, 0.114643],
                    [-0.01417039, 0.12388105],
                    [0.0, 0.140],
                    [0.02524261, 0.1304036],
                    [-0.0059703, 0.13986947],
                    [0.0073023, 0.11479653],
                    [0.00567301, 0.1271345],
                ]
            )
        elif setpoint_mode == "image":
            # use fast but slightly inaccurate projected descent for image setpoints
            self.planning_fn = jit(
                partial(
                    statically_invert_actuation_to_task_space_projected_descent,
                    params=self.params,
                    residual_fn=self.residual_fn,
                    inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
                    maxiter=250,
                    verbose=False,
                )
            )

            default_planning_frequency = 5

            self.declare_parameter("image_type", "star")
            image_type = self.get_parameter("image_type").value

            if image_type == "star":
                image_path = os.path.join(
                    get_package_share_directory("hsa_planar_control"),
                    "assets",
                    "star.png",
                )
                pee_centroid = jnp.array([0.0, 0.127])
                max_radius = jnp.array(0.013)
            elif image_type == "tud-flame":
                image_path = os.path.join(
                    get_package_share_directory("hsa_planar_control"),
                    "assets",
                    "tud_flame.jpeg",
                )
                pee_centroid = jnp.array([0.0, 0.130])
                max_radius = jnp.array(0.013)
            elif image_type == "mit-csail":
                image_path = os.path.join(
                    get_package_share_directory("hsa_planar_control"),
                    "assets",
                    "mit_csail.png",
                )
                pee_centroid = jnp.array([0.0, 0.127])
                max_radius = jnp.array(0.013)
            else:
                raise ValueError(f"Unknown image type: {image_type}")

            self.pee_des_sps = generate_task_space_trajectory_from_image(
                image_type=image_type,
                image_path=image_path,
                pee_centroid=pee_centroid,
                max_radius=max_radius,
                verbose=False,
                show_images=False,
            )
        else:
            raise ValueError(f"Unknown setpoint mode: {setpoint_mode}")

        # run the planning function once to compile it
        (
            chiee_des_dummy,
            q_des_dummy,
            phi_ss_dummy,
            optimality_error_dummy,
        ) = self.planning_fn(pee_des=jnp.array([0.0, 0.110]))
        self.get_logger().info("Done compiling static inversion planning function!")

        # initial setpoint index
        self.setpoint_idx = 0

        self.declare_parameter("planning_frequency", default_planning_frequency)
        self.timer = self.create_timer(
            1 / self.get_parameter("planning_frequency").value, self.timer_callback
        )

    def timer_callback(self):
        planning_start_time = self.get_clock().now()

        chiee_des, q_des, phi_ss, optimality_error = self.planning_fn(
            pee_des=self.pee_des_sps[self.setpoint_idx]
        )

        # Log how long the planning took
        self.get_logger().info(
            f"Planning took: {(self.get_clock().now() - planning_start_time).nanoseconds / 1e6} ms"
        )

        # Log the setpoint
        self.get_logger().info(
            f"chiee_des: {chiee_des}, q_des: {q_des}, phi_ss: {phi_ss}, optimality_error: {optimality_error}"
        )

        # skip setpoint if optimality error is too large
        if optimality_error > 1e-3:
            self.get_logger().warn("Skipping setpoint due to large optimality error.")
            self.setpoint_idx += 1
            return

        msg = PlanarSetpoint()
        msg.chiee_des.x = chiee_des[0].item()
        msg.chiee_des.y = chiee_des[1].item()
        msg.chiee_des.theta = chiee_des[2].item()
        msg.q_des.header.stamp = self.get_clock().now().to_msg()
        msg.q_des.kappa_b = q_des[0].item()
        msg.q_des.sigma_sh = q_des[1].item()
        msg.q_des.sigma_a = q_des[2].item()
        msg.phi_ss = phi_ss.tolist()
        msg.optimality_error = optimality_error.item()
        self.pub.publish(msg)

        self.setpoint_idx += 1


def main(args=None):
    rclpy.init(args=args)
    print("Hi from the static inversion planning node.")

    node = StaticInversionPlanningNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
