from jax import Array
import jax.numpy as jnp
from typing import Callable, Tuple


def get_operational_workspace_boundaries(
    x_ps: Array = None, hsa_material: str = "fpu"
) -> Tuple[Array, Array]:
    """
    Get the boundaries of the operational workspace.
    Args:
        x_ps: x coordinates of the points where we should evaluate the boundaries as array of shape (N, )
        hsa_material: material of the HSA (either "fpu" or "epu")

    Returns:
        pee_ps_min: points of the lower boundary of the operational workspace as shape (N, 2)
        pee_ps_max: points of the upper boundary of the operational workspace as shape (N, 2)
    """
    if hsa_material == "fpu":
        if x_ps is None:
            x_ps = jnp.linspace(-0.04471130921323639, 0.04471130921323639, num=50)
        # coefficients of polynomial fit of lower (i.e., min) boundary of operational workspace
        pmin = jnp.array(
            [
                -4.66083730e28,
                5.39792934e22,
                4.83883600e26,
                -5.25500731e20,
                -2.15491588e24,
                2.16721408e18,
                5.38084317e21,
                -4.92836852e15,
                -8.26661107e18,
                6.73728237e12,
                8.07280065e15,
                -5.65931126e09,
                -5.00736196e12,
                2.86297272e06,
                1.92280334e09,
                -8.16784922e02,
                -4.39879657e05,
                1.13880450e-01,
                6.08493344e01,
                -5.45457238e-06,
                1.10730959e-01,
            ]
        )
        # coefficients of polynomial fit of upper (i.e., max) boundary of operational workspace
        pmax = jnp.array(
            [
                2.55205196e28,
                1.30459184e23,
                -2.67821421e26,
                -1.28270505e21,
                1.20710759e24,
                5.34961970e18,
                -3.05499007e21,
                -1.23215150e16,
                4.76537540e18,
                1.70925285e13,
                -4.73539170e15,
                -1.46040236e10,
                2.99725122e12,
                7.53768304e06,
                -1.17900445e09,
                -2.20302553e03,
                2.77347541e05,
                3.16510526e-01,
                -5.16215169e01,
                -1.57653245e-05,
                1.44094793e-01,
            ]
        )
        # evaluate the polynomial fit at the given x coordinates
        ymin_ps = jnp.polyval(pmin, x_ps)
        ymax_ps = jnp.polyval(pmax, x_ps)
        # concatenate the x coordinates with the y coordinates
        pee_min_ps = jnp.stack((x_ps, ymin_ps), axis=-1)
        pee_max_ps = jnp.stack((x_ps, ymax_ps), axis=-1)
        return pee_min_ps, pee_max_ps
    else:
        raise NotImplementedError(
            f"Operational workspace is not implemented for hsa_material: {hsa_material}"
        )
