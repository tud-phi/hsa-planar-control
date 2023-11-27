from jax import Array, debug, jacfwd
import jax.numpy as jnp
from typing import Callable, Dict, Tuple

from .generalized_torques_to_actuation import (
    linearize_actuation,
    map_generalized_torques_to_actuation_with_linearized_model,
)


def pd_plus_feedforward(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    dynamical_matrices_fn: Callable,
    q_des: Array,
    Kp: Array,
    Kd: Array,
    **kwargs,
) -> Tuple[Array, Dict[str, Array]]:
    """
    PD plus static feedforward controller in configuration space.
    Evaluates the gravity and stiffness at the desired configuration.
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        q_des: desired configuration vector of shape (n_q, )
        Kp: proportional gain matrix of shape (n_q, n_q)
        Kd: derivative gain matrix of shape (n_q, n_q)

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_info: dictionary with information about intermediate computations
    """
    # compute the desired torque in configuration space
    B_des, C_des, G_des, K_des, D_des, alpha_des = dynamical_matrices_fn(
        q_des, jnp.zeros_like(q_des), phi
    )
    tau_q_des = Kp @ (q_des - q) - Kd @ q_d + K_des + G_des

    phi_des = map_generalized_torques_to_actuation_with_linearized_model(
        dynamical_matrices_fn, q, phi, tau_q_des
    )

    controller_info = {}

    return phi_des, controller_info


def pd_plus_potential_compensation(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    dynamical_matrices_fn: Callable,
    q_des: Array,
    Kp: Array,
    Kd: Array,
    **kwargs,
) -> Tuple[Array, Dict[str, Array]]:
    """
    PD plus potential compensation controller in configuration space.
    Evaluates the gravity and the stiffness at the current configuration.
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        q_des: desired configuration vector of shape (n_q, )
        Kp: proportional gain matrix of shape (n_q, n_q)
        Kd: derivative gain matrix of shape (n_q, n_q)
    Returns:
        phi_des: desired motor positions (n_phi, )
        controller_info: dictionary with information about intermediate computations
    """
    # compute the desired torque in configuration space
    B, C, G, K, D, alpha = dynamical_matrices_fn(q, q_d, phi)
    tau_q_des = Kp @ (q_des - q) - Kd @ q_d + G + K

    phi_des = map_generalized_torques_to_actuation_with_linearized_model(
        dynamical_matrices_fn, q, phi, tau_q_des
    )

    controller_info = {}

    return phi_des, controller_info


def pd_plus_steady_state_actuation(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    dynamical_matrices_fn: Callable,
    q_des: Array,
    phi_ss: Array,
    Kp: Array,
    Kd: Array,
    **kwargs,
) -> Tuple[Array, Dict[str, Array]]:
    """
    PD plus steady state actuation
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        q_des: desired configuration vector of shape (n_q, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_info: dictionary with information about intermediate computations
    """
    # linearize the system around the desired configuration
    tau_eq, A = linearize_actuation(dynamical_matrices_fn, q_des, phi_ss)

    # implement the underactuated PD + feedforward controller
    phi_des = phi_ss + Kp @ A.T @ (q_des - q) - Kd @ A.T @ q_d

    controller_info = {}

    return phi_des, controller_info


def P_satI_D_plus_steady_state_actuation(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    controller_state: Dict[str, Array],
    dynamical_matrices_fn: Callable,
    dt: float,
    q_des: Array,
    phi_ss: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    gamma: Array = jnp.array(1.0),
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    P-saturation-I-D plus steady state actuation. As a saturation function, we use the hyperbolic tangent.
    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        controller_state: state of the controller (integral error)
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        dt: time step of controller [s]
        q_des: desired configuration vector of shape (n_q, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Ki: integral gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)
        gamma: horizontal compression factor of the hyperbolic tangent. Array of shape () or (n_phi, )

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_state: state of the controller (integral error)
        controller_info: dictionary with information about intermediate computations
    """
    # linearize the system around the desired configuration
    tau_eq, A = linearize_actuation(dynamical_matrices_fn, q, phi_ss)

    # error
    e_phi = A.T @ (q_des - q)

    # implement the underactuated PD + feedforward controller
    phi_des = (
        phi_ss + Kp @ e_phi - Kd @ A.T @ q_d + Ki @ controller_state["integral_error"]
    )

    controller_state["integral_error"] += jnp.tanh(gamma * e_phi) * dt

    controller_info = {"e_phi": e_phi, "e_int": controller_state["integral_error"]}

    return phi_des, controller_state, controller_info


def P_satI_D_collocated_form_plus_steady_state_actuation_for_constant_stiffness(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    controller_state: Dict[str, Array],
    dynamical_matrices_fn: Callable,
    dt: float,
    q_des: Array,
    phi_ss: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    gamma: Array = jnp.array(1.0),
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    P-saturation-I-D on the system in collocated form plus steady state actuation.
    As a saturation function, we use the hyperbolic tangent.
    **Attention**: only works with constant stiffness for now

    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        controller_state: state of the controller (integral error)
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        dt: time step of controller [s]
        q_des: desired configuration vector of shape (n_q, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Ki: integral gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)
        gamma: horizontal compression factor of the hyperbolic tangent. Array of shape () or (n_phi, )

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_state: state of the controller (integral error)
        controller_info: dictionary with information about intermediate computations
    """
    # linearize the system around the desired configuration
    tau_eq, A = linearize_actuation(dynamical_matrices_fn, q_des, phi_ss)

    def coordinate_transform_into_collocated_form(_q: Array) -> Array:
        """
        Transform the configuration vector into the collocated form.
        """
        _sigma_sh = _q[1:2]  # shear strain
        _varphi = jnp.concatenate([A.T @ _q, _sigma_sh], axis=0)
        return _varphi

    varphi = coordinate_transform_into_collocated_form(q)
    varphi_des = coordinate_transform_into_collocated_form(q_des)

    # compute the Jacobian of the coordinate transformation for computing the derivative
    Jvarphi = jacfwd(coordinate_transform_into_collocated_form)(q)
    varphi_d = Jvarphi @ q_d

    # error on the actuated coordinates
    e_y = varphi_des[:2] - varphi[:2]

    # implement the underactuated PD + feedforward controller
    phi_des = (
        phi_ss + Kp @ e_y - Kd @ varphi_d[:2] + Ki @ controller_state["integral_error"]
    )

    controller_state["integral_error"] += jnp.tanh(gamma * e_y) * dt

    controller_info = {
        "varphi": varphi,
        "varphi_des": varphi_des,
        "e_y": e_y,
        "e_int": controller_state["integral_error"],
    }

    return phi_des, controller_state, controller_info


def P_satI_D_collocated_form_plus_steady_state_actuation(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    controller_state: Dict[str, Array],
    map_into_collocated_form_fn: Callable,
    dt: float,
    q_des: Array,
    phi_ss: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    gamma: Array = jnp.array(1.0),
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    P-saturation-I-D on the system in collocated form plus steady state actuation.
    As a saturation function, we use the hyperbolic tangent.

    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        controller_state: state of the controller (integral error)
        map_into_collocated_form_fn: Callable that maps the configuration vector into the collocated form.
        dt: time step of controller [s]
        q_des: desired configuration vector of shape (n_q, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Ki: integral gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)
        gamma: horizontal compression factor of the hyperbolic tangent. Array of shape () or (n_phi, )

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_state: state of the controller (integral error)
        controller_info: dictionary with information about intermediate computations
    """
    varphi, Jvarphi = map_into_collocated_form_fn(q, phi_ss)
    varphi_des, Jvarphi_des = map_into_collocated_form_fn(q_des, phi_ss)

    # velocity of the current collocated coordinates
    varphi_d = Jvarphi @ q_d

    # error on the actuated coordinates y
    e_y = varphi_des[:2] - varphi[:2]

    # implement the underactuated PD + feedforward controller
    phi_des = (
        phi_ss + Kp @ e_y - Kd @ varphi_d[:2] + Ki @ controller_state["integral_error"]
    )

    controller_state["integral_error"] += jnp.tanh(gamma * e_y) * dt

    controller_info = {
        "varphi": varphi,
        "varphi_des": varphi_des,
        "e_y": e_y,
        "e_int": controller_state["integral_error"],
    }

    return phi_des, controller_state, controller_info


def P_satI_D_collocated_form_plus_gravity_cancellation_elastic_compensation(
    t: Array,
    q: Array,
    q_d: Array,
    phi: Array,
    *args,
    controller_state: Dict[str, Array],
    dynamical_matrices_fn: Callable,
    map_into_collocated_form_fn: Callable,
    dt: float,
    q_des: Array,
    phi_ss: Array,
    Kp: Array,
    Ki: Array,
    Kd: Array,
    gamma: Array = jnp.array(1.0),
    **kwargs,
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    P-saturation-I-D on the system in collocated form plus terms for gravity cancellation and elastic compensation.
    As a saturation function, we use the hyperbolic tangent.

    Args:
        t: time [s]
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        controller_state: state of the controller (integral error)
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
            Needs to conform to the signature: dynamical_matrices_fn(q, q_d) -> Tuple[B, C, G, K, D, alpha]
        map_into_collocated_form_fn: Callable that maps the configuration vector into the collocated form.
        dt: time step of controller [s]
        q_des: desired configuration vector of shape (n_q, )
        phi_ss: steady state actuation at the desired configuration. Vector of shape (n_phi, )
        Kp: proportional gain matrix of shape (n_phi, n_phi)
        Ki: integral gain matrix of shape (n_phi, n_phi)
        Kd: derivative gain matrix of shape (n_phi, n_phi)
        gamma: horizontal compression factor of the hyperbolic tangent. Array of shape () or (n_phi, )

    Returns:
        phi_des: desired motor positions (n_tau, )
        controller_state: state of the controller (integral error)
        controller_info: dictionary with information about intermediate computations
    """
    B, C, G, K, D, alpha = dynamical_matrices_fn(q, q_d, phi)
    B_des, C_des, G_des, K_des, D_des, alpha_des = dynamical_matrices_fn(
        q_des, jnp.zeros_like(q_d), phi
    )

    varphi, Jvarphi = map_into_collocated_form_fn(q, phi_ss)
    varphi_des, Jvarphi_des = map_into_collocated_form_fn(q_des, phi_ss)
    # inverse of Jacobian
    Jvarphi_inv = jnp.linalg.inv(Jvarphi)

    # project gravity vector into collocated coordinates
    G_varphi = Jvarphi_inv.T @ G
    G_varphi_des = Jvarphi_inv.T @ G_des

    # velocity of the current collocated coordinates
    varphi_d = Jvarphi @ q_d

    # error on the actuated coordinates y
    e_y = varphi_des[:2] - varphi[:2]

    # implement the underactuated PD + feedforward controller
    phi_des = (
        phi_ss
        - G_varphi_des[:2]
        + G_varphi[:2]
        + Kp @ e_y
        - Kd @ varphi_d[:2]
        + Ki @ controller_state["integral_error"]
    )

    controller_state["integral_error"] += jnp.tanh(gamma * e_y) * dt

    controller_info = {
        "varphi": varphi,
        "varphi_des": varphi_des,
        "varphi_d": varphi_d,
        "e_y": e_y,
        "e_int": controller_state["integral_error"],
    }

    return phi_des, controller_state, controller_info
