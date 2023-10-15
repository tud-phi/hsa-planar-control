from jax import Array
import jax.numpy as jnp
from typing import Callable, Tuple


def get_operational_workspace_boundaries(x_ps: Array = None, hsa_material: str = "fpu") -> Tuple[Array, Array]:
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
        pmin = jnp.array([
            -4.66083730e+28,
            5.39792934e+22,
            4.83883600e+26,
            - 5.25500731e+20,
            - 2.15491588e+24,
            2.16721408e+18,
            5.38084317e+21,
            - 4.92836852e+15,
            - 8.26661107e+18,
            6.73728237e+12,
            8.07280065e+15,
            - 5.65931126e+09,
            - 5.00736196e+12,
            2.86297272e+06,
            1.92280334e+09,
            - 8.16784922e+02,
            - 4.39879657e+05,
            1.13880450e-01,
            6.08493344e+01,
            - 5.45457238e-06,
            1.10730959e-01
        ])
        # coefficients of polynomial fit of upper (i.e., max) boundary of operational workspace
        pmax = jnp.array([
            2.55205196e+28,
            1.30459184e+23,
            - 2.67821421e+26,
            - 1.28270505e+21,
            1.20710759e+24,
            5.34961970e+18,
            - 3.05499007e+21,
            - 1.23215150e+16,
            4.76537540e+18,
            1.70925285e+13,
            - 4.73539170e+15,
            - 1.46040236e+10,
            2.99725122e+12,
            7.53768304e+06,
            - 1.17900445e+09,
            - 2.20302553e+03,
            2.77347541e+05,
            3.16510526e-01,
            - 5.16215169e+01,
            - 1.57653245e-05,
            1.44094793e-01
        ])
        # evaluate the polynomial fit at the given x coordinates
        ymin_ps = jnp.polyval(pmin, x_ps)
        ymax_ps = jnp.polyval(pmax, x_ps)
        # concatenate the x coordinates with the y coordinates
        pee_min_ps = jnp.stack((x_ps, ymin_ps), axis=-1)
        pee_max_ps = jnp.stack((x_ps, ymax_ps), axis=-1)
        return pee_min_ps, pee_max_ps
    else:
        raise NotImplementedError(f"Operational workspace is not implemented for hsa_material: {hsa_material}")
