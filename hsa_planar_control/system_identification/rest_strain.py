from jax import Array
import jax.numpy as jnp
from os import PathLike
from typing import Dict, Tuple

from .optimization.linear_lq import (
    linear_lq_optim_problem_factory,
    optimize_with_closed_form_linear_lq,
)


def identify_axial_rest_strain_for_system_id_dataset(
    sym_exp_filepath: PathLike,
    sys_helpers: Dict,
    params: Dict[str, Array],
    data_ts: Dict[str, Array],
    num_time_steps: int = 5,
    separate_rods: bool = False,
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
        separate_rods: whether to identify the axial rest strain for each rod separately. Default: False.
    Returns:
        sigma_a_eq: axial rest strain as Array of shape (num_segments, num_rods_per_segment)
    """
    shared_params_mapping = None  # we use the default procedure
    if separate_rods:
        num_segments = params["roff"].shape[0]
        num_rods_per_segment = params["roff"].shape[1]
        if num_rods_per_segment > 2:
            # we can only identify two axial rest strains
            # therefore, we need to employ parameter sharing
            params_to_be_idd_names = ["sigma_a_eqL", "sigma_a_eqR"]
            shared_params_mapping = {"sigma_a_eqL": [], "sigma_a_eqR": []}
            rod_idx = 0
            for i in range(num_segments):
                for j in range(num_rods_per_segment):
                    if jnp.sign(params["roff"][i, j]) == 1:
                        # positive x-coordinates means that the rod is on the right side of the robot
                        shared_params_mapping["sigma_a_eqR"].append(f"sigma_a_eq{rod_idx+1}")
                    else:
                        shared_params_mapping["sigma_a_eqL"].append(f"sigma_a_eq{rod_idx+1}")
                    rod_idx += 1
        else:
            params_to_be_idd_names = ["sigma_a_eq1", "sigma_a_eq2"]
    else:
        params_to_be_idd_names = ["sigma_a_eq"]

    # remove the axial rest strain from the known parameters
    params = params.copy()
    params.pop("sigma_a_eq")

    Pi_syms, cal_a_fn, cal_b_fn = linear_lq_optim_problem_factory(
        sym_exp_filepath,
        sys_helpers,
        params,
        params_to_be_idd_names=params_to_be_idd_names,
        mode="static",
        shared_params_mapping=shared_params_mapping,
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

    if separate_rods:
        sigma_a_eq1, sigma_a_eq2 = Pi_est[0], Pi_est[1]
        param_to_est_mapping = {}
        for param_est_idx, param_sym in enumerate(Pi_syms):
            param_to_est_mapping[param_sym.name] = Pi_est[param_est_idx].item()
        print(
            f"Identified axial rest strains:\n{param_to_est_mapping}"
        )

        if "sigma_a_eqL" and "sigma_a_eqR" in param_to_est_mapping.keys():
            sigma_a_eq = jnp.ones_like(params["roff"])
            sigma_a_eq = sigma_a_eq.at[params["roff"] >= 0].mul(param_to_est_mapping["sigma_a_eqR"])
            sigma_a_eq = sigma_a_eq.at[params["roff"] < 0].mul(param_to_est_mapping["sigma_a_eqL"])
        else:
            sigma_a_eq = jnp.repeat(jnp.array([[sigma_a_eq1, sigma_a_eq2]]), params["roff"].shape[0], axis=0)
    else:
        sigma_a_eq_scalar = Pi_est[0]
        print("Identified scalar axial rest strain: ", sigma_a_eq_scalar)

        sigma_a_eq = sigma_a_eq_scalar * jnp.ones_like(params["roff"])

    return sigma_a_eq


def identify_rest_strains_for_system_id_dataset(
    sym_exp_filepath: PathLike,
    sys_helpers: Dict,
    params: Dict[str, Array],
    data_ts: Dict[str, Array],
    num_time_steps: int = 5,
) -> Tuple[Array, Array, Array]:
    """
    Identify all rest strains for a system identification dataset.
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
        params_to_be_idd_names=["kappa_b_eq", "sigma_sh_eq", "sigma_a_eq"],
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
    kappa_b_eq_scalar, sigma_sh_eq_scalar, sigma_a_eq_scalar = (
        Pi_est[0],
        Pi_est[1],
        Pi_est[2],
    )
    print(
        f"Identified scalar rest strains:\n"
        f"kappa_b_eq={kappa_b_eq_scalar}, sigma_sh_eq={sigma_sh_eq_scalar}, sigma_a_eq={sigma_a_eq_scalar}"
    )

    kappa_b_eq = kappa_b_eq_scalar * jnp.ones_like(params["rout"])
    sigma_sh_eq = sigma_sh_eq_scalar * jnp.ones_like(params["rout"])
    sigma_a_eq = sigma_a_eq_scalar * jnp.ones_like(params["rout"])

    return kappa_b_eq, sigma_sh_eq, sigma_a_eq
