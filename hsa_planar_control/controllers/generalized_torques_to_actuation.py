from jax import Array, debug, jacfwd, jit
import jax.numpy as jnp
from jaxopt import GaussNewton, LevenbergMarquardt
from typing import Callable, Tuple


def linearize_actuation(
    dynamical_matrices_fn: Callable, q_eq: Array, phi_eq: Array
) -> Tuple[Array, Array]:
    """
    Linearize the actuation vector.
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
        q_eq: configuration vector of shape (n_q, )
        phi_eq: current motor positions vector of shape (n_phi, )
    Returns:
        tau_eq: actuation vector of shape (n_q, )
        A: linearized actuation vector of shape (n_q, n_phi)
    """

    def alpha_fn(_phi: Array) -> Array:
        _, _, _, _, _, _alpha = dynamical_matrices_fn(
            q_eq, jnp.zeros_like(q_eq), phi=_phi
        )
        return _alpha

    tau_eq = alpha_fn(phi_eq)  # torque at which the system is linearized
    # linearized actuation vector
    A = jacfwd(alpha_fn)(phi_eq)

    return tau_eq, A


def map_generalized_torques_to_actuation_with_linearized_model(
    dynamical_matrices_fn: Callable, q_eq: Array, phi_eq: Array, tau_q_des: Array
) -> Array:
    """
    Map a desired torque in configuration space to a desired twist angle by linearizing the actuation vector.
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
        q_eq: configuration vector of shape (n_q, )
        phi_eq: current motor positions vector of shape (n_phi, )
        tau_q_des: desired torque in configuration space of shape (n_q, )
    Returns:
        phi_des: desired motor positions (n_phi, )
    """

    tau_eq, A = linearize_actuation(dynamical_matrices_fn, q_eq, phi_eq)

    # compute the desired phi
    phi_des = phi_eq + jnp.linalg.pinv(A) @ (tau_q_des - tau_eq)

    return phi_des


def map_generalized_torques_to_actuation_with_nonlinear_optimization(
    dynamical_matrices_fn: Callable, q: Array, tau_q_des: Array, phi0: Array
) -> Tuple[Array, Array]:
    """
    Map a desired torque in configuration space to desired twist angles by performing nonlinear optimization
    Args:
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha.
        q: configuration vector of shape (n_q, )
        tau_q_des: desired torque in configuration space of shape (n_q, )
        phi0: initial guess for the motor positions of shape (n_phi, )
    Returns:
        phi_des: desired motor positions (n_phi, )
    """

    def residual_fn(_phi: Array) -> Array:
        _, _, _, _, _, _alpha = dynamical_matrices_fn(q, jnp.zeros_like(q), phi=_phi)
        return _alpha - tau_q_des

    # solve the nonlinear least squares problem
    lm = LevenbergMarquardt(residual_fun=residual_fn)
    sol = lm.run(phi0)

    # optimal phi
    phi_des = sol.params

    # compute the L2 optimality
    optimality_error = lm.l2_optimality_error(sol.params)

    return phi_des, optimality_error
