from ament_index_python.packages import get_package_share_directory
import derivative
from example_interfaces.msg import Float64MultiArray
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit, random
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path
import os

from hsa_control_interfaces.msg import PlanarSetpoint
from hsa_planar_control.simulation import simulate_steady_state
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration


class RandonSetpointsNode(Node):
    def __init__(self):
        super().__init__("random_setpoints_node")

        # set random seed
        self.rng = random.PRNGKey(seed=0)

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
            forward_kinematics_virtual_backbone_fn,
            forward_kinematics_end_effector_fn,
            jacobian_end_effector_fn,
            inverse_kinematics_end_effector_fn,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.declare_parameter("hsa_material", "fpu")
        hsa_material = self.get_parameter("hsa_material").value
        if hsa_material == "fpu":
            self.params = PARAMS_FPU_CONTROL.copy()
        elif hsa_material == "epu":
            self.params = PARAMS_EPU_CONTROL.copy()
        else:
            raise ValueError(f"Unknown HSA material: {hsa_material}")

        # parameters for specifying different rest strains
        self.declare_parameter("kappa_b_eq", self.params["kappa_b_eq"].mean().item())
        self.declare_parameter("sigma_sh_eq", self.params["sigma_sh_eq"].mean().item())
        self.declare_parameter("sigma_a_eq1", self.params["sigma_a_eq"][0, 0].item())
        self.declare_parameter("sigma_a_eq2", self.params["sigma_a_eq"][0, 1].item())
        kappa_b_eq = self.get_parameter("kappa_b_eq").value
        sigma_sh_eq = self.get_parameter("sigma_sh_eq").value
        sigma_a_eq1 = self.get_parameter("sigma_a_eq1").value
        sigma_a_eq2 = self.get_parameter("sigma_a_eq2").value
        self.params["kappa_b_eq"] = kappa_b_eq * jnp.ones_like(
            self.params["kappa_b_eq"]
        )
        self.params["sigma_sh_eq"] = sigma_sh_eq * jnp.ones_like(
            self.params["sigma_sh_eq"]
        )
        self.params["sigma_a_eq"] = jnp.array([[sigma_a_eq1, sigma_a_eq2]])
        # external payload mass (assumed to be at end effector)
        self.declare_parameter("payload_mass", 0.0)
        self.params["mpl"] = self.get_parameter("payload_mass").value

        self.declare_parameter("phi_max", self.params["phi_max"].mean().item())
        self.params["phi_max"] = self.get_parameter("phi_max").value * jnp.ones_like(
            self.params["phi_max"]
        )

        # increase damping on the bending strain
        self.params["zetab"] = 4 * self.params["zetab"]
        self.params["zetash"] = 4 * self.params["zetash"]
        self.params["zetaa"] = 4 * self.params["zetaa"]

        # specify a jitted version of the forward kinematics function
        self.forward_kinematics_end_effector_fn = jit(
            partial(forward_kinematics_end_effector_fn, self.params)
        )

        # define residual function for static inversion optimization
        sim_dt = 1e-4  # time step for simulation [s]
        duration = 30.0  # duration of simulation [s]
        self.simulate_steady_state_fn = jit(
            partial(
                simulate_steady_state,
                dynamical_matrices_fn,
                self.params,
                sim_dt=sim_dt,
                duration=duration,
            )
        )

        # jit the forward kinematics and steady state simulation function
        phi_ss_dummy = self.params["phi_max"].flatten()
        q_ss_dummy, q_d_ss_dummy = self.simulate_steady_state_fn(phi_ss_dummy)
        assert q_d_ss_dummy.mean() < 1e-2, (
            "The simulation hasn't converged to a steady state which means that likely the duration is too short."
            + f"q_ss_dummy: {q_ss_dummy}, q_d_ss_dummy: {q_d_ss_dummy}"
        )
        chiee_ss_dummy = self.forward_kinematics_end_effector_fn(q_ss_dummy)
        self.get_logger().info(
            f"Finished jitting the forward kinematics and steady state simulation function."
        )

        # initial setpoint index
        self.setpoint_idx = 0

        self.declare_parameter("setpoint_frequency", 0.04)
        self.timer = self.create_timer(
            1 / self.get_parameter("setpoint_frequency").value, self.timer_callback
        )

    def timer_callback(self):
        # split PRNG key
        self.rng, rng_setpoint = random.split(self.rng)

        # sample the actuation magnitude
        phi_ss_mag_lb = 0.2 * jnp.ones_like(self.params["phi_max"].flatten())  # lower bound for sampling [rad]
        phi_ss_mag_ub = self.params["phi_max"].flatten() - 0.2  # upper bound for sampling [rad]
        phi_ss_mag = random.uniform(
            rng_setpoint, shape=phi_ss_mag_lb.shape, minval=phi_ss_mag_lb, maxval=phi_ss_mag_ub
        )
        # compensate for the handedness of the rods
        phi_ss = phi_ss_mag * self.params["h"].flatten()

        rollout_start_time = self.get_clock().now()
        q_ss, q_d_ss = self.simulate_steady_state_fn(phi_ss)
        chiee_ss = self.forward_kinematics_end_effector_fn(q_ss)

        # Log how long the rollout took
        self.get_logger().info(
            f"Rollout to steady-state took: {(self.get_clock().now() - rollout_start_time).nanoseconds / 1e6} ms"
        )

        # Log the setpoint
        self.get_logger().info(f"chiee_ss: {chiee_ss}, q_ss: {q_ss}, phi_ss: {phi_ss}")

        msg = PlanarSetpoint()
        msg.chiee_des.x = chiee_ss[0].item()
        msg.chiee_des.y = chiee_ss[1].item()
        msg.chiee_des.theta = chiee_ss[2].item()
        msg.q_des.header.stamp = self.get_clock().now().to_msg()
        msg.q_des.kappa_b = q_ss[0].item()
        msg.q_des.sigma_sh = q_ss[1].item()
        msg.q_des.sigma_a = q_ss[2].item()
        msg.phi_ss = phi_ss.tolist()
        self.pub.publish(msg)

        self.setpoint_idx += 1


def main(args=None):
    rclpy.init(args=args)
    print("Hi from the random setpoints node.")

    node = RandonSetpointsNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
