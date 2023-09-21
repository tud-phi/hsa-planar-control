from functools import partial
from jax import Array, debug, jacfwd, jit
import jax.numpy as jnp
from typing import Callable, Dict, Tuple

from .actuation_linearization import map_configuration_space_torque_to_twist_angle


def basic_operational_space_pid(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    controller_state: Dict[str, Array],
    forward_kinematics_end_effector_fn: Callable,
    jacobian_end_effector_fn,
    dt: float,
    pee_des: Array,
    phi_ss: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    **kwargs,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Implement a basic PID controller in operational space.
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        controller_state: state of the controller (integral error)
        forward_kinematics_end_effector_fn: function that returns the pose of the end effector of shape (3, )
        jacobian_end_effector_fn: function that returns the Jacobian of the end effector of shape (3, n_q)
        dt: time step of controller [s]
        pee_des: desired end effector position of shape (2, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Ki: integral gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_state: state of the controller (integral error)
    """
    # jacobian of the end-effector mapping velocities from configuration space to operational space
    chiee = forward_kinematics_end_effector_fn(q)
    Jee = jacobian_end_effector_fn(q)

    # current position of end-effector
    pee = chiee[:2]
    # current velocity of end-effector
    p_d_ee = Jee[:2, :] @ q_d

    # control input in task space
    e_pee = pee_des - pee
    u = Kp @ e_pee + Ki @ controller_state["integral_error"] - Kd @ p_d_ee

    controller_state["e_pee"] = e_pee
    controller_state["integral_error"] += e_pee * dt

    # project control input to the actuation space
    phi_des = phi_ss + jnp.array([[1, 1], [-1, 1]]) @ u

    return phi_des, controller_state


def operational_space_computed_torque(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    forward_kinematics_end_effector_fn: Callable,
    jacobian_end_effector_fn,
    dynamical_matrices_fn: Callable,
    operational_space_dynamical_matrices_fn: Callable,
    pee_des: Array,
    Kp: Array,
    Kd: Array,
    consider_underactuation_model: bool = True,
    **kwargs,
) -> Array:
    """
    Implement a computed torque controller in operational space.
    References:
        - "A unified approach for motion and force control of robot manipulators: The operational space formulation"
            by Oussama Khatib
        - "Exact task execution in highly under-actuated soft limbs" by Della Santina et al.
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        forward_kinematics_end_effector_fn: function that returns the pose of the end effector of shape (3, )
        jacobian_end_effector_fn: function that returns the Jacobian of the end effector of shape (3, n_q)
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        operational_space_dynamical_matrices_fn: Callable with signature (q, q_d, B, C) -> Lambda, nu, JB_pinv
        pee_des: desired Cartesian-space position for end-effector of shape (2, )
        Kp: proportional gain matrix of shape (2, 2)
        Kd: derivative gain matrix of shape (2, 2)
        consider_underactuation_model: If True, the underactuation model is considered. Otherwise, the fully-actuated
            model is considered with the identity matrix as the actuation matrix.
            Be aware that if using jit, consider_underactuation_model cannot be changed during runtime.
    Returns:
        u: input to the system.
            - if consider_underactuation_model is True, then this is an array of shape (n_phi) with
                motor positions / twist angles of the proximal end of the rods
            - if consider_underactuation_model is False, then this is an array of shape (n_q) with
                the generalized torques
    """
    # jacobian of the end-effector mapping velocities from configuration space to operational space
    chiee = forward_kinematics_end_effector_fn(q)
    Jee = jacobian_end_effector_fn(q)

    # current position of end-effector
    pee = chiee[:2]
    # current velocity of end-effector
    p_d_ee = Jee[:2, :] @ q_d

    debug.print("pee = {pee}, p_d_ee={p_d_ee}", pee=pee, p_d_ee=p_d_ee)

    B, C, G, K, D, alpha = dynamical_matrices_fn(q, q_d, phi)
    Lambda, nu, JB_pinv = operational_space_dynamical_matrices_fn(q, q_d, B, C)

    # debug.print("Lambda = {Lambda}, nu={nu}, JB_pinv={JB_pinv}", Lambda=Lambda, nu=nu, JB_pinv=JB_pinv)

    debug.print("error p_ee = {error_p_ee}", error_p_ee=pee_des - pee)

    # desired force in operational space
    f_des = (
        Lambda[:2, :2] @ (Kp @ (pee_des - pee) - Kd @ p_d_ee)
        + nu[:2]
        + JB_pinv[:, :2].T @ (G + K + D @ q_d)
    )

    debug.print("f_des = {f_des}", f_des=f_des)

    # project end-effector force into configuration space
    tau_q_des = Jee[:2, :].T @ f_des

    debug.print("tau_q_des = {tau_q_des}", tau_q_des=tau_q_des)

    if consider_underactuation_model:
        phi_des = map_configuration_space_torque_to_twist_angle(
            q, phi, dynamical_matrices_fn, tau_q_des
        )
        debug.print("phi_des = {phi_des}", phi_des=phi_des)

        u = phi_des
    else:
        u = tau_q_des

    return u