from jax import Array, debug, jit
import jax.numpy as jnp
from typing import Dict, Tuple


@jit
def saturate_control_inputs(
    params: Dict[str, Array], phi_des: Array, controller_state: Dict[str, Array], controller_info: Dict[str, Array]
) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
    """
    Saturate the control inputs (compensated with handedness) to the range [0, phi_max].

    Parameters
    ----------
    params: Dict[str, Array]
        The parameters of the system.
    phi_des: Array
        The desired control inputs of shape (n_phi, ).
    controller_state: Dict[str, Array]
        The state of the controller.
    controller_info: Dict[str, Array]
        Information about intermediate computations.
    Returns
    -------
    phi_sat: Array
        The saturated control inputs.
    controller_state: Dict[str, Array]
        The updated state of the controller.
    controller_info: Dict[str, Array]
        Information about intermediate computations.
    """
    phi_max = params["phi_max"]
    h = params["h"]

    phi_sat = jnp.clip(h.flatten() * phi_des, 0.0, phi_max.flatten())

    controller_info["phi_des_unsat"] = phi_des
    controller_info["phi_des_sat"] = phi_sat

    return phi_sat, controller_state, controller_info
