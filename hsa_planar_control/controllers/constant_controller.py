from jax import Array, debug
import jax.numpy as jnp


def constant_control(
    t: Array, q: Array, q_d: Array, phi: Array, *args, phi_des: Array, **kwargs
):
    """
    Dummy controller function always returning a constant phi
    Args:
        t: time
        q: configuration vector of shape (n_q, )
        q_d: configuration velocity vector of shape (n_q, )
        phi: current motor positions vector of shape (n_phi, )
        phi_des: constant desired control input of shape (n_phi, )
    Returns:
        phi_des: constant control input of shape (n_phi, )
    """
    return phi_des
