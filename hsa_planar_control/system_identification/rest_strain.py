from jax import Array
import jax.numpy as jnp
from os import PathLike
from typing import Dict

from .optimization.linear_lq import linear_lq_optim_problem_factory, optimize_with_closed_form_linear_lq


def identify_rest_strain_for_system_id_dataset(
    sym_exp_filepath: PathLike,
    sys_helpers: Dict,
    params: Dict[str, Array],
    data_ts: Dict[str, Array],
    num_time_steps: int = 5,
) -> Array:
    """
    Identify the axial rest strain for a system identification dataset.
    We take two key assumptions:
    1. The robot is at rest (i.e. steady-state properties) for the first few time steps of the experiment.
    2. The robot is in its rest state (i.e. no actuation) for the first few time steps of the experiment.
    This still allows us to take into account the gravitational and elastic forces acting on the robot.
    Args:
        sym_exp_filepath: path to the file with saved symbolic expressions
        sys_helpers: dictionary with helper entries for the HSA system
        params: dictionary with known robot parameters (except for the axial rest strain)
        data_ts: Dictionary with the preprocessed data. We assume that the robot is at rest for the first
            few time steps of the experiment.
        num_time_steps: number of time steps to use for the identification. Default: 5.
    Returns:
        sigma_a_eq: axial rest strain as Array of shape (num_segments, num_rods_per_segment)
    """
    Pi_syms, cal_a_fn, cal_b_fn = linear_lq_optim_problem_factory(
        sym_exp_filepath,
        sys_helpers,
        params,
        params_to_be_idd_names=["sigma_a_eq"],
        mode="static",
    )

    # construct the data_ts dictionary with the first num_time_steps time steps
    select_data_ts = {}
    for key, val in data_ts.items():
        select_data_ts[key] = val[:num_time_steps]

    Pi_est = optimize_with_closed_form_linear_lq(
        cal_a_fn,
        cal_b_fn,
        select_data_ts,
        verbose=False,
    )
    sigma_a_eq_scalar = Pi_est[0]
    print("Identified scalar axial rest strain: ", sigma_a_eq_scalar)

    sigma_a_eq = sigma_a_eq_scalar * jnp.ones_like(params["rout"])

    return sigma_a_eq
