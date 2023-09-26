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
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from pathlib import Path

from geometry_msgs.msg import Pose2D
from hsa_control_interfaces.msg import PlanarSetpoint, PlanarSetpointControllerInfo, Pose2DStamped
from mocap_optitrack_interfaces.msg import PlanarCsConfiguration

from hsa_actuation.hsa_actuation_base_node import HsaActuationBaseNode
from hsa_planar_control.collocated_form import mapping_into_collocated_form_factory
from hsa_planar_control.controllers.configuration_space_controllers import (
    P_satI_D_plus_steady_state_actuation,
    P_satI_D_collocated_form_plus_steady_state_actuation,
    P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation,
)
from hsa_planar_control.controllers.operational_space_controllers import (
    basic_operational_space_pid,
)
from hsa_planar_control.controllers.saturation import saturate_control_inputs


class ModelBasedControlNode(HsaActuationBaseNode):
    def __init__(self):
        super().__init__("model_based_control_node")
        self.declare_parameter("configuration_topic", "configuration")
        self.configuration_sub = self.create_subscription(
            PlanarCsConfiguration,
            self.get_parameter("configuration_topic").value,
            self.configuration_listener_callback,
            10,
        )

        self.declare_parameter("end_effector_pose_topic", "end_effector_pose")
        self.end_effector_pose_sub = self.create_subscription(
            Pose2DStamped,
            self.get_parameter("end_effector_pose_topic").value,
            self.end_effector_pose_listener_callback,
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

        self.params = PARAMS_CONTROL.copy()

        # parameter for specifying a different axial rest strain
        self.declare_parameter("sigma_a_eq", self.params["sigma_a_eq"].mean().item())
        sigma_a_eq = self.get_parameter("sigma_a_eq").value
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

        # initial actuation coordinates
        phi0 = self.map_motor_angles_to_actuation_coordinates(self.present_motor_angles)

        # history of configurations
        # the longer the history, the more delays we introduce, but the less noise we get
        self.declare_parameter("history_length_for_diff", 16)
        self.tq_hs = jnp.zeros((self.get_parameter("history_length_for_diff").value,))
        self.q_hs = jnp.zeros(
            (self.get_parameter("history_length_for_diff").value, self.n_q)
        )
        # history of end-effector poses
        # the longer the history, the more delays we introduce, but the less noise we get
        self.tchiee_hs = jnp.zeros((self.get_parameter("history_length_for_diff").value,))
        self.chiee_hs = jnp.zeros(
            (self.get_parameter("history_length_for_diff").value, self.chiee.shape[0])
        )

        # method for computing derivative
        self.diff_method = derivative.Spline(s=1.0, k=3)
        self.declare_parameter("configuration_velocity_topic", "configuration_velocity")
        self.configuration_velocity_pub = self.create_publisher(
            PlanarCsConfiguration,
            self.get_parameter("configuration_velocity_topic").value,
            10,
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
        self.phi_ss = jnp.zeros_like(phi0)
        self.setpoint_msg = None
        self.declare_parameter("reset_integral_error_on_setpoint_change", False)
        self.reset_integral_error = self.get_parameter("reset_integral_error_on_setpoint_change").value

        self.declare_parameter(
            "controller_type", "P_satI_D_collocated_form_plus_steady_state_actuation"
        )
        self.controller_type = self.get_parameter("controller_type").value
         # it seems that roughly 45 Hz is the maximum at the moment
        self.declare_parameter("control_frequency", 40)
        self.control_frequency = self.get_parameter("control_frequency").value
        control_dt = 1 / self.control_frequency
        self.declare_parameter("Kp", 0.0)
        Kp = self.get_parameter("Kp").value * jnp.eye(phi0.shape[0])
        self.declare_parameter("Ki", 0.0)
        Ki = self.get_parameter("Ki").value * jnp.eye(phi0.shape[0])
        self.declare_parameter("Kd", 0.0)
        Kd = self.get_parameter("Kd").value * jnp.eye(phi0.shape[0])
        self.declare_parameter("gamma", 1.0)
        gamma = self.get_parameter("gamma").value * jnp.ones_like(phi0)
        self.controller_state = {
            "integral_error": jnp.zeros_like(phi0),
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
            self.control_fn = partial(
                basic_operational_space_pid,
                dt=control_dt,
                phi_ss=self.params["phi_max"].squeeze() / 2,
                Kp=Kp,
                Ki=Ki,
                Kd=Kd,
            )
        else:
            raise NotImplementedError(
                "Controller type {} not implemented".format(self.controller_type)
            )

        phi_dummy = self.map_motor_angles_to_actuation_coordinates(
            self.present_motor_angles
        )
        if self.controller_type == "basic_operational_space_pid":
            phi_des_dummy, _, _ = self.control_fn(
                0.0,
                self.chiee,
                self.chiee_d,
                phi_dummy,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
            )
        else:
            phi_des_dummy, _, _ = self.control_fn(
                0.0,
                self.q,
                self.q_d,
                phi_dummy,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
                q_des=self.q_des,
                phi_ss=self.phi_ss,
            )
        motor_goal_angles_dummy = self.map_actuation_coordinates_to_motor_angles(
            phi_des_dummy
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
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current configuration
        self.q = jnp.array([msg.kappa_b, msg.sigma_sh, msg.sigma_a])

        # update history
        self.tq_hs = jnp.roll(self.tq_hs, shift=-1, axis=0)
        self.tq_hs = self.tq_hs.at[-1].set(t)
        self.q_hs = jnp.roll(self.q_hs, shift=-1, axis=0)
        self.q_hs = self.q_hs.at[-1].set(self.q)

    def end_effector_pose_listener_callback(self, msg: Pose2DStamped):
        t = Time.from_msg(msg.header.stamp).nanoseconds / 1e9

        # set the current end-effector pose
        self.chiee = jnp.array([msg.pose.x, msg.pose.y, msg.pose.theta])

        # update history
        self.tchiee_hs = jnp.roll(self.tchiee_hs, shift=-1, axis=0)
        self.tchiee_hs = self.tchiee_hs.at[-1].set(t)
        self.chiee_hs = jnp.roll(self.chiee_hs, shift=-1, axis=0)
        self.chiee_hs = self.chiee_hs.at[-1].set(self.chiee)

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
            self.controller_state["integral_error"] = jnp.zeros_like(self.controller_state["integral_error"])

    def compute_q_d(self) -> Array:
        """
        Compute the velocity of the generalized coordinates from the history of configurations.
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tq_hs == 0.0):
            return self.q_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        t_hs = self.tq_hs - self.tq_hs[0]

        q_d = jnp.zeros_like(self.q)
        # iterate through configuration variables
        for i in range(self.q_hs.shape[-1]):
            # derivative of all time stamps for configuration variable i
            q_d_hs = self.diff_method.d(self.q_hs[:, i], t_hs)

            q_d = q_d.at[i].set(q_d_hs[-1])

        q_d_msg = PlanarCsConfiguration(
            kappa_b=q_d[0].item(), sigma_sh=q_d[1].item(), sigma_a=q_d[2].item()
        )
        q_d_msg.header.stamp = self.get_clock().now().to_msg()

        self.configuration_velocity_pub.publish(q_d_msg)

        return q_d
    

    def compute_chiee_d(self) -> Array:
        """
        Compute the velocity of the end-effector pose from the history of end-effector poses.
        """
        # if the buffer is not full yet, return the current velocity
        if jnp.any(self.tchiee_hs == 0.0):
            return self.chiee_d

        # subtract the first time stamp from all time stamps to avoid numerical issues
        tchiee_hs = self.tchiee_hs - self.tchiee_hs[0]

        chiee_d = jnp.zeros_like(self.chiee)
        # iterate through configuration variables
        for i in range(self.chiee_hs.shape[-1]):
            # derivative of all time stamps for configuration variable i
            chiee_d_hs = self.diff_method.d(self.chiee_hs[:, i], tchiee_hs)

            chiee_d = chiee_d.at[i].set(chiee_d_hs[-1])

        return chiee_d


    def map_motor_angles_to_actuation_coordinates(self, motor_angles: Array) -> Array:
        """
        Map the motor angles to the actuation coordinates. The actuation coordinates are defined as twist angle
        of an imagined rod on the left and right respectively.
        """
        control_handedness = self.params["h"][
            0
        ]  # handedness of rods in first segment in control model
        phi = jnp.stack(
            [
                (
                    motor_angles[2] * self.rod_handedness[2]
                    + motor_angles[3] * self.rod_handedness[3]
                )
                * control_handedness[0]
                / 2,
                (
                    motor_angles[0] * self.rod_handedness[0]
                    + motor_angles[1] * self.rod_handedness[1]
                )
                * control_handedness[1]
                / 2,
            ]
        )
        return phi

    def map_actuation_coordinates_to_motor_angles(self, phi: Array) -> Array:
        """
        We devise the control input in positive actuation coordinates of shape (2, ). However, we need to actuate
        four motors. This function maps the two actuation coordinates to the four motor angles.
        """
        control_handedness = self.params["h"][
            0
        ]  # handedness of rods in first segment in control model

        motor_angles = jnp.stack(
            [
                phi[1] * control_handedness[1] * self.rod_handedness[0],
                phi[1] * control_handedness[1] * self.rod_handedness[1],
                phi[0] * control_handedness[0] * self.rod_handedness[2],
                phi[0] * control_handedness[0] * self.rod_handedness[3],
            ]
        )
        return motor_angles

    def call_controller(self):
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if self.setpoint_msg is None:
            # we have not received a setpoint yet so we cannot compute the control input
            return

        # compute the velocity of the generalized coordinates and the end-effector pose
        self.q_d = self.compute_q_d()
        self.chiee_d = self.compute_chiee_d()

        # map motor angles to actuation coordinates
        phi = self.map_motor_angles_to_actuation_coordinates(self.present_motor_angles)

        # save the current configuration and velocity
        q, q_d = self.q, self.q_d
        chiee, chiee_d = self.chiee, self.chiee_d

        # evaluate controller
        if self.controller_type == "basic_operational_space_pid":
            phi_des, self.controller_state, controller_info = self.control_fn(
                t,
                chiee,
                chiee_d,
                phi,
                controller_state=self.controller_state,
                pee_des=self.chiee_des[:2],
            )
        else:
            phi_des, self.controller_state, controller_info = self.control_fn(
                t,
                q,
                q_d,
                phi,
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

        # map the actuation coordinates to motor angles
        motor_goal_angles = self.map_actuation_coordinates_to_motor_angles(phi_sat)

        # send motor goal angles to dynamixel motors
        self.set_motor_goal_angles(motor_goal_angles)

        # publish controller info
        controller_info_msg = PlanarSetpointControllerInfo()
        controller_info_msg.header.stamp = self.get_clock().now().to_msg()
        controller_info_msg.planar_setpoint = self.setpoint_msg
        controller_info_msg.q = PlanarCsConfiguration(
            header=controller_info_msg.header, kappa_b=q[0].item(), sigma_sh=q[1].item(), sigma_a=q[2].item()
        )
        controller_info_msg.q_d = PlanarCsConfiguration(
            header=controller_info_msg.header, kappa_b=q_d[0].item(), sigma_sh=q_d[1].item(), sigma_a=q_d[2].item()
        )
        controller_info_msg.chiee = Pose2DStamped(
            header=controller_info_msg.header,
            pose=Pose2D(
                x=self.chiee[0].item(),
                y=self.chiee[1].item(),
                theta=self.chiee[2].item(),
            )
        )
        controller_info_msg.chiee_d = Pose2DStamped(
            header=controller_info_msg.header,
            pose=Pose2D(
                x=self.chiee_d[0].item(),
                y=self.chiee_d[1].item(),
                theta=self.chiee_d[2].item(),
            )
        )
        if "e_int" in controller_info:
            controller_info_msg.e_int = controller_info["e_int"].tolist()
        if "varphi" in controller_info:
            controller_info_msg.varphi = controller_info["varphi"].tolist()
        if "varphi_des" in controller_info:
            controller_info_msg.varphi_des = controller_info["varphi_des"].tolist()
        controller_info_msg.phi_des_unsat = phi_des_unsat.tolist()
        controller_info_msg.phi_des_sat = phi_sat.tolist()
        controller_info_msg.motor_goal_angles = motor_goal_angles.tolist()
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
