from jax import Array, debug
import jax.numpy as jnp
from typing import Dict, Tuple


def constant_control(
    t: Array, q: Array, q_d: Array, phi: Array, *args, phi_des: Array, **kwargs
) -> Tuple[Array, Dict[str, Array]]:
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
        controller_info: dictionary with information about intermediate computations
    """
    controller_info = {}
    return phi_des, controller_info
