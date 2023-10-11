from copy import deepcopy
import derivative
from example_interfaces.msg import Float64MultiArray
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax
from jax import Array, jit
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path

from geometry_msgs.msg import Pose2D
from hsa_control_interfaces.msg import (
    PlanarSetpoint,
    PlanarSetpointControllerInfo,
    Pose2DStamped,
)
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

from hsa_planar_control.collocated_form import mapping_into_collocated_form_factory
from hsa_planar_control.controllers.configuration_space_controllers import (
    P_satI_D_plus_steady_state_actuation,
    P_satI_D_collocated_form_plus_steady_state_actuation,
    P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation,
)
from hsa_planar_control.controllers.operational_space_controllers import (
    basic_operational_space_pid,
    operational_space_impedance_control_linearized_actuation
)
from hsa_planar_control.controllers.saturation import saturate_control_inputs


class ModelBasedControlNode(Node):
    def __init__(self):
        super().__init__("model_based_control_node")
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_topic").value,
            self.configuration_listener_callback,
            10,
        )
        self.declare_parameter("configuration_velocity_topic", "configuration_velocity")
        self.configuration_velocity_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_velocity_topic").value,
            self.configuration_velocity_listener_callback,
            10,
        )

        self.declare_parameter("end_effector_pose_topic", "end_effector_pose")
        self.end_effector_pose_sub = self.create_subscription(
            Pose2DStamped,
            self.get_parameter("end_effector_pose_topic").value,
            self.end_effector_pose_listener_callback,
            10,
        )
        self.declare_parameter("end_effector_velocity_topic", "end_effector_velocity")
        self.end_effector_velocity_sub = self.create_subscription(
            Pose2DStamped,
            self.get_parameter("end_effector_velocity_topic").value,
            self.end_effector_velocity_listener_callback,
            10,
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
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        kappa_b_eq = self.get_parameter("kappa_b_eq").value
        sigma_sh_eq = self.get_parameter("sigma_sh_eq").value
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
        self.params["kappa_b_eq"] = kappa_b_eq * jnp.ones_like(
            self.params["kappa_b_eq"]
        )
        self.params["sigma_sh_eq"] = sigma_sh_eq * jnp.ones_like(
            self.params["sigma_sh_eq"]
        )
        self.params["sigma_a_eq"] = sigma_a_eq * jnp.ones_like(
            self.params["sigma_a_eq"]
        )
        # actual rest strain
        self.xi_eq = sys_helpers["rest_strains_fn"](self.params)  # rest strains
        # external payload mass (assumed to be at end effector)
        self.declare_parameter("payload_mass", 0.0)
        self.params["mpl"] = self.get_parameter("payload_mass").value

        self.declare_parameter("phi_max", self.params["phi_max"].mean().item())
        self.params["phi_max"] = self.get_parameter("phi_max").value * jnp.ones_like(
            self.params["phi_max"]
        )

        # initialize system measurements
        self.q = jnp.zeros_like(self.xi_eq)  # generalized coordinates
        self.n_q = self.q.shape[0]  # number of generalized coordinates
        self.q_d = jnp.zeros_like(self.q)  # velocity of generalized coordinates
        # end-effector pose
        self.chiee = forward_kinematics_end_effector_fn(self.params, self.q)
        self.chiee_d = jnp.zeros_like(self.chiee)  # velocity of end-effector pose

        # present actuation coordinates
        self.phi = jnp.zeros_like(self.params["roff"].flatten())
        self.declare_parameter("present_planar_actuation_topic", "present_planar_actuation")
        self.phi_sub = self.create_subscription(
            Float64MultiArray, self.get_parameter("present_planar_actuation_topic").value, self.actuation_coordinates_listener_callback, 10
        )
        # publisher of control input
        self.declare_parameter("control_input_topic", "control_input")
        self.control_input_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter("control_input_topic").value, 10
        )

        self.declare_parameter("setpoint_topic", "setpoint")
        self.setpoints_sub = self.create_subscription(
            PlanarSetpoint,
            self.get_parameter("setpoint_topic").value,
            self.setpoint_listener_callback,
            10,
        )
        self.q_des = jnp.zeros_like(self.q)
        self.chiee_des = jnp.zeros((3,))
        self.phi_ss = jnp.zeros_like(self.phi)
        self.setpoint_msg = None
        self.declare_parameter("reset_integral_error_on_setpoint_change", False)
        self.reset_integral_error = self.get_parameter(
            "reset_integral_error_on_setpoint_change"
        ).value

        self.declare_parameter(
            "controller_type", "P_satI_D_collocated_form_plus_steady_state_actuation"
        )
        self.controller_type = self.get_parameter("controller_type").value
        # it seems that roughly 45 Hz is the maximum at the moment
        self.declare_parameter("control_frequency", 40)
        self.control_frequency = self.get_parameter("control_frequency").value
        control_dt = 1 / self.control_frequency
        self.declare_parameter("Kp", 0.0)
        Kp = self.get_parameter("Kp").value * jnp.eye(self.phi.shape[0])
        self.declare_parameter("Ki", 0.0)
        Ki = self.get_parameter("Ki").value * jnp.eye(self.phi.shape[0])
        self.declare_parameter("Kd", 0.0)
        Kd = self.get_parameter("Kd").value * jnp.eye(self.phi.shape[0])
        self.declare_parameter("gamma", 1.0)
        gamma = self.get_parameter("gamma").value * jnp.ones_like(self.phi)
        self.controller_state = {
            "integral_error": jnp.zeros_like(self.phi),
        }
        map_into_collocated_form_fn, _ = mapping_into_collocated_form_factory(
            sym_exp_filepath, sys_helpers
        )

        if (
            self.controller_type
            == "P_satI_D_collocated_form_plus_steady_state_actuation"
        ):
            self.control_fn = jit(
                partial(
                    P_satI_D_collocated_form_plus_steady_state_actuation,
                    map_into_collocated_form_fn=partial(
                        map_into_collocated_form_fn, self.params
                    ),
                    dt=control_dt,
                    Kp=Kp,
                    Ki=Ki,
                    Kd=Kd,
                    gamma=gamma,
                )
            )
        elif (
            self.controller_type
            == "P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation"
        ):
            self.control_fn = jit(
                partial(
                    P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation,
                    dynamical_matrices_fn=partial(dynamical_matrices_fn, self.params),
                    map_into_collocated_form_fn=partial(
                        map_into_collocated_form_fn, self.params
                    ),
                    dt=control_dt,
                    Kp=Kp,
                    Ki=Ki,
                    Kd=Kd,
                    gamma=gamma,
                )
            )
        elif self.controller_type == "P_satI_D_plus_steady_state_actuation":
            self.control_fn = jit(
                partial(
                    P_satI_D_plus_steady_state_actuation,
                    dynamical_matrices_fn=partial(dynamical_matrices_fn, self.params),
                    dt=control_dt,
                    Kp=Kp,
                    Ki=Ki,
                    Kd=Kd,
                    gamma=gamma,
                )
            )
        elif self.controller_type == "basic_operational_space_pid":
            self.control_fn = jit(
                partial(
                    basic_operational_space_pid,
                    dt=control_dt,
                    phi_ss=self.params["phi_max"].squeeze() / 2,
                    Kp=Kp,
                    Ki=Ki,
                    Kd=Kd,
                )
            )
        elif self.controller_type == "operational_space_impedance_control_linearized_actuation":
            self.control_fn = partial(
                operational_space_impedance_control_linearized_actuation,
                jacobian_end_effector_fn=partial(jacobian_end_effector_fn, self.params),
                dynamical_matrices_fn=partial(dynamical_matrices_fn, self.params),
                operational_space_dynamical_matrices_fn=partial(
                    sys_helpers["operational_space_dynamical_matrices_fn"], self.params
                ),
                Kp=Kp,
                Kd=Kd,
                logger=self.get_logger(),
            )
        else:
            raise NotImplementedError(
                "Controller type {} not implemented".format(self.controller_type)
            )

        if self.controller_type == "basic_operational_space_pid":
            phi_des_dummy, _, _ = self.control_fn(
                0.0,
                self.chiee,
                self.chiee_d,
                self.phi,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
            )
        elif self.controller_type == "operational_space_impedance_control_linearized_actuation":
            phi_des_dummy, _= self.control_fn(
                0.0,
                self.chiee,
                self.chiee_d,
                self.q,
                self.q_d,
                self.phi,
                pee_des=self.chiee_des[:2],
            )
        else:
            phi_des_dummy, _, _ = self.control_fn(
                0.0,
                self.q,
                self.q_d,
                self.phi,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
                q_des=self.q_des,
                phi_ss=self.phi_ss,
            )

        # initialize publisher for controller info
        self.controller_info_pub = self.create_publisher(
            PlanarSetpointControllerInfo, "controller_info", 10
        )

        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.call_controller
        )

        self.start_time = self.get_clock().now()

    def configuration_listener_callback(self, msg: PlanarCsConfiguration):
        # set the current configuration
        self.q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

    def configuration_velocity_listener_callback(self, msg: PlanarCsConfiguration):
        # set the current configuration velocity
        self.q_d = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

    def end_effector_pose_listener_callback(self, msg: Pose2DStamped):
        # set the current end-effector pose
        self.chiee = jnp.array([msg.pose.x, msg.pose.y, msg.pose.theta])

    def end_effector_velocity_listener_callback(self, msg: Pose2DStamped):
        # set the current end-effector velocity
        self.chiee_d = jnp.array([msg.pose.x, msg.pose.y, msg.pose.theta])

    def actuation_coordinates_listener_callback(self, msg: Float64MultiArray):
        # present actuation coordinates
        self.phi = jnp.array(msg.data)

    def setpoint_listener_callback(self, msg: PlanarSetpoint):
        self.setpoint_msg = msg
        self.q_des = jnp.array(
            [msg.q_des.kappa_b, msg.q_des.sigma_sh, msg.q_des.sigma_a]
        )
        self.chiee_des = jnp.array(
            [msg.chiee_des.x, msg.chiee_des.y, msg.chiee_des.theta]
        )
        self.phi_ss = jnp.array(msg.phi_ss)

        if self.reset_integral_error:
            # reset integral error
            self.controller_state["integral_error"] = jnp.zeros_like(
                self.controller_state["integral_error"]
            )

    def call_controller(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if self.setpoint_msg is None:
            # we have not received a setpoint yet so we cannot compute the control input
            return

        # save the current configuration and velocity
        q, q_d = self.q, self.q_d
        chiee, chiee_d = self.chiee, self.chiee_d

        # evaluate controller
        if self.controller_type == "basic_operational_space_pid":
            phi_des, self.controller_state, controller_info = self.control_fn(
                t,
                chiee,
                chiee_d,
                self.phi,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
            )
        elif self.controller_type == "operational_space_impedance_control_linearized_actuation":
            phi_des, controller_info = self.control_fn(
                t,
                chiee,
                chiee_d,
                q,
                q_d,
                self.phi,
                pee_des=self.chiee_des[:2],
            )
        else:
            phi_des, self.controller_state, controller_info = self.control_fn(
                t,
                q,
                q_d,
                self.phi,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
                q_des=self.q_des,
                phi_ss=self.phi_ss,
            )

        # compensate for the handedness specified in the parameters
        phi_des_unsat = self.params["h"].flatten() * phi_des

        # saturate the control input
        phi_sat, self.controller_state, controller_info = saturate_control_inputs(
            self.params,
            phi_des_unsat,
            controller_state=self.controller_state,
            controller_info=controller_info,
        )

        # self.get_logger().info(f"Saturated control inputs: {phi_sat}")

        # publish message with the control input
        phi_msg = Float64MultiArray(data=phi_sat.tolist())
        self.control_input_pub.publish(phi_msg)

        # publish controller info
        controller_info_msg = PlanarSetpointControllerInfo()
        controller_info_msg.header.stamp = self.get_clock().now().to_msg()
        controller_info_msg.planar_setpoint = self.setpoint_msg
        controller_info_msg.q = PlanarCsConfiguration(
            header=controller_info_msg.header,
            kappa_b=q[0].item(),
            sigma_sh=q[1].item(),
            sigma_a=q[2].item(),
        )
        controller_info_msg.q_d = PlanarCsConfiguration(
            header=controller_info_msg.header,
            kappa_b=q_d[0].item(),
            sigma_sh=q_d[1].item(),
            sigma_a=q_d[2].item(),
        )
        controller_info_msg.chiee = Pose2DStamped(
            header=controller_info_msg.header,
            pose=Pose2D(
                x=self.chiee[0].item(),
                y=self.chiee[1].item(),
                theta=self.chiee[2].item(),
            ),
        )
        controller_info_msg.chiee_d = Pose2DStamped(
            header=controller_info_msg.header,
            pose=Pose2D(
                x=self.chiee_d[0].item(),
                y=self.chiee_d[1].item(),
                theta=self.chiee_d[2].item(),
            ),
        )
        if "e_int" in controller_info:
            controller_info_msg.e_int = controller_info["e_int"].tolist()
        if "varphi" in controller_info:
            controller_info_msg.varphi = controller_info["varphi"].tolist()
        if "varphi_des" in controller_info:
            controller_info_msg.varphi_des = controller_info["varphi_des"].tolist()
        controller_info_msg.phi_des_unsat = phi_des_unsat.tolist()
        controller_info_msg.phi_des_sat = phi_sat.tolist()
        self.controller_info_pub.publish(controller_info_msg)


def main(args=None):
    rclpy.init(args=args)
    print("Hi from the planar model-based control node.")

    node = ModelBasedControlNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
