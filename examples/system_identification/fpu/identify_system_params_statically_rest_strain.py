from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_SYSTEM_ID
from jsrm.systems import planar_hsa
import matplotlib
import numpy as onp

matplotlib.use("WebAgg")
from pathlib import Path

from hsa_planar_control.system_identification.optimization.linear_lq import (
    linear_lq_optim_problem_factory,
    optimize_with_closed_form_linear_lq,
)
from hsa_planar_control.system_identification.preprocessing import preprocess_data


num_segments = 1
num_rods_per_segment = 4

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)


ones_rod = jnp.ones((num_segments, num_rods_per_segment))
known_params = PARAMS_FPU_SYSTEM_ID.copy()
# delete payload mass from known params so that we can specify it for each datapoint
known_params.pop("mpl")

experiment_configs = {
    # step of elongation to 90 deg
    "20230621_170058": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of elongation to 120 deg
    "20230621_170509": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of elongation to 180 deg
    "20230621_170624": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of elongation to 210 deg
    "20230621_170734": {
        "t_ss": jnp.array([0.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of bending to 90 deg
    "20230621_171040": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of bending to 120 deg
    "20230621_171232": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
    # step of bending to 180 deg
    "20230621_171345": {
        "t_ss": jnp.array([10.0, 22.0]),
        "mpl_ss": jnp.array([0.0, 0.0]),
    },
}
mocap_body_ids = {"base": 4, "platform": 5}
resampling_dt = 0.01  # [s]

if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    params_to_be_idd_names = ["sigma_a_eq"]
    Pi_syms, cal_a_fn, cal_b_fn = linear_lq_optim_problem_factory(
        sym_exp_filepath,
        dynamical_matrices_fn,
        sys_helpers,
        known_params,
        params_to_be_idd_names,
        mode="static",
    )

    data_ts = None
    for experiment_idx, (experiment_id, experiment_config) in enumerate(
        experiment_configs.items()
    ):
        print("Processing experiment: ", experiment_id)
        experiment_data_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "system_identification"
            / experiment_id
        )
        experiment_data_ts = preprocess_data(
            inverse_kinematics_end_effector_fn,
            experiment_data_path,
            known_params,
            mocap_body_ids,
            resampling_dt=resampling_dt,
            filter=True,
            derivative_method="savgol_filter",
            plotting=False,
        )

        t_ts = experiment_data_ts[
            "t_ts"
        ]  # array of timestamps of entire trajectory [s]
        t_ss = experiment_config[
            "t_ss"
        ]  # array of timestamps of steady-state data we want to select [s]

        # for each entry of t_ss, find the closest entry in t_ts
        t_ss = jnp.array([t_ts[jnp.argmin(jnp.abs(t_ts - t_ss_i))] for t_ss_i in t_ss])
        print("Selected the following timestamps for steady-state data: ", t_ss)
        # select all values of t_ts with entries in t_ss
        t_ts_selector = jnp.isin(t_ts, t_ss)

        experiment_selected_ss_data_ts = {
            "t_ts": t_ss,
            "xi_ts": experiment_data_ts["xi_ts"][t_ts_selector, ...],
            "xi_d_ts": experiment_data_ts["xi_d_ts"][t_ts_selector, ...],
            "xi_dd_ts": experiment_data_ts["xi_dd_ts"][t_ts_selector, ...],
            "phi_ts": experiment_data_ts["phi_ts"][t_ts_selector, ...],
            "mpl_ts": experiment_config["mpl_ss"],
        }

        if data_ts is None:
            data_ts = experiment_selected_ss_data_ts
        else:
            data_ts = {
                k: jnp.concatenate((v, experiment_selected_ss_data_ts[k]))
                for k, v in data_ts.items()
            }

    Pi_est = optimize_with_closed_form_linear_lq(
        cal_a_fn,
        cal_b_fn,
        data_ts,
    )

    print("Identified system params using steady-state samples:\n", Pi_est)
    onp.savetxt("Pi_fpu_static_est_rest_strain.csv", Pi_est, delimiter=",")
