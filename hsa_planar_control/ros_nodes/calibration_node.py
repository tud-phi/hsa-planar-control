import derivative
from example_interfaces.msg import Float64MultiArray
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_SYSTEM_ID
from jsrm.systems import planar_hsa
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path

from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

from hsa_planar_control.system_identification.optimization.linear_lq import (
    linear_lq_optim_problem_factory,
    optimize_with_closed_form_linear_lq,
)


class CalibrationNode(Node):
    """
    Calibration node for the planar HSA.
    Assumes that the system is at rest and in its nominal configuration (i.e. phi = 0).
    """

    def __init__(self):
        super().__init__("model_based_control_node")
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_topic").value,
            self.configuration_listener_callback,
            10,
        )

        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_hsa_ns-1_nrs-4.dill"
        )
        (
            _,
            _,
            _,
            _,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)

        self.known_params = PARAMS_SYSTEM_ID
        # rest strains used for inverse kinematics
        self.xi_eq_ik = sys_helpers["rest_strains_fn"](self.known_params)

        # external payload mass (assumed to be at end effector)
        self.declare_parameter("payload_mass", 0.0)
        self.params["mpl"] = self.get_parameter("payload_mass").value

        params_to_be_idd_names = ["sigma_a_eq"]
        self.Pi_syms, self.cal_a_fn, self.cal_b_fn = linear_lq_optim_problem_factory(
            sym_exp_filepath,
            dynamical_matrices_fn,
            sys_helpers,
            self.known_params,
            params_to_be_idd_names,
            mode="static",
        )

        # initialize state
        self.q = jnp.zeros_like(self.xi_eq_ik)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates
        self.n_phi = self.known_params["roff"].flatten().shape[0]  # number of actuators

        # history of configurations
        # the longer the history, the more delays we introduce, but the less noise we get
        self.declare_parameter("history_length_for_diff", 20)
        self.t_hs = jnp.zeros((self.get_parameter("history_length_for_diff").value,))
        self.q_hs = jnp.zeros(
            (self.get_parameter("history_length_for_diff").value, self.n_q)
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

    def configuration_listener_callback(self, msg):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        self.q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

        # update history
        self.t_hs = jnp.roll(self.t_hs, shift=-1, axis=0)
        self.t_hs = self.t_hs.at[-1].set(t)
        self.q_hs = jnp.roll(self.q_hs, shift=-1, axis=0)
        self.q_hs = self.q_hs.at[-1].set(self.q)

    def timer_callback(self):
        if jnp.any(self.t_hs == 0.0):
            # buffer is not full yet
            return

        data_hs = {
            "t_ts": self.t_hs,
            "xi_ts": self.xi_eq_ik + self.q_hs,
            "xi_d_ts": jnp.zeros_like(self.q_hs),
            "xi_dd_ts": jnp.zeros_like(self.q_hs),
            "phi_ts": jnp.zeros((self.t_hs.shape[0], self.n_phi)),
            "mpl_ts": self.params["mpl"] * jnp.ones_like(self.t_hs),
        }

        Pi_est = optimize_with_closed_form_linear_lq(
            self.cal_a_fn,
            self.cal_b_fn,
            data_hs,
        )

        self.get_logger().info(f"Pi_est: {Pi_est}")


def main(args=None):
    rclpy.init(args=args)
    print("Hi from the planar calibration node.")

    node = CalibrationNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
