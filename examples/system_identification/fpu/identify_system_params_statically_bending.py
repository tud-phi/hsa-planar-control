from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax.numpy as jnp
import jsrm
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

params_to_be_idd_names = ["S_b_hat", "S_sh_hat", "S_b_sh", "C_S_b", "C_S_sh"]

ones_rod = jnp.ones((num_segments, num_rods_per_segment))
known_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
    # length of the rigid proximal caps of the rods connecting to the base [m]
    "lpc": 25e-3 * jnp.ones((num_segments,)),
    # length of the rigid distal caps of the rods connecting to the platform [m]
    "ldc": 14e-3 * jnp.ones((num_segments,)),
    "sigma_a_eq": 1.0753753 * ones_rod,  # axial rest strains of each rod
    # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
    "C_varepsilon": 0.00984819 * ones_rod,  # Average: 0.009118994, Std: 0.000696435
    # outside radius of each rod [m]. The rows correspond to the segments.
    "rout": 25.4e-3 / 2 * ones_rod,  # this is for FPU rods
    # inside radius of each rod [m]. The rows correspond to the segments.
    "rin": (25.4e-3 / 2 - 2.43e-3) * ones_rod,  # this is for FPU rods
    # handedness of each rod. The rows correspond to the segments.
    "h": jnp.array([[1.0, -1.0, 1.0, -1.0]]),
    # offset [m] of each rod from the centerline. The rows correspond to the segments.
    "roff": 24e-3 * jnp.array([[1.0, 1.0, -1.0, -1.0]]),
    "pcudim": jnp.array(
        [[80e-3, 12e-3, 80e-3]]
    ),  # width, height, depth of the platform [m]
    # mass of FPU rod: 14 g, mass of EPU rod: 26 g
    # For FPU, this corresponds to a measure volume of 0000175355 m^3 --> rho = 798.38 kg/m^3
    "rhor": 798.38 * ones_rod,  # Volumetric density of rods [kg/m^3],
    # Volumetric density of platform [kg/m^3],
    # weight of platform + marker holder + cylinder top piece: 0.107 kg
    # subtracting 4 x 9g for distal cap: 0.071 kg
    # volume of platform (excluding proximal and distal caps): 0.0000768 m^3
    # --> rho = 925 kg/m^3
    "rhop": 925 * jnp.ones((num_segments,)),
    # volumetric density of the rigid end pieces [kg/m^3]
    # mass of 3D printed rod (between rin and rout): 8.5g
    # mass of the rigid end piece (up to rin): 9g
    # volume: pi*lpc*rout^2 = 0.0000126677 m^3
    # --> rho = 710.4 kg/m^3
    "rhoec": 710.4 * jnp.ones((num_segments,)),
    "g": jnp.array([0.0, 9.81]),
    "S_a_hat": 5.66472469 * ones_rod,  # Nominal axial stiffness of each rod [N]
    # Scaling of axial stiffness with twist strain [Nm/rad]
    "C_S_a": 0.01508165 * ones_rod,
    "lpl": 25e-3,  # length of payload [m] (100g weights)
    # center of origin of the payload relative to end-effector [m]
    # subtract 12 mm for the thickness of the platform
    # the 100g weights have a length of 25mm
    "CoGpl": jnp.array([0.0, -12e-3 - 12.5e-3]),
}

experiment_configs = {
    # # staircase of bending to 180 deg
    # "20230621_165020": {
    #     "t_ss": jnp.array([
    #         2.0, 3.0, 4.0,
    #         8.2, 10.0, 11.4,
    #         14.5, 16.0, 17.0,
    #         20.0, 21.0, 22.0,
    #         26, 27.0, 28.0,
    #         32.0, 33.0, 34.0,
    #     ]),
    #     "mpl_ss": jnp.array([
    #         0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0,
    #         0.0, 0.0, 0.0,
    #     ]),
    # },
    # ccw staircase bending with alternating payload
    "20230703_155911": {
        "t_ss": jnp.array([
            6.33, 21.5, 65.8,
            78.4, 99.3, 132.9,
            185,
            225, 257.5, 277,
            292, 317.5, 346,
            363, 402.5, 415.7,
        ]),
        "mpl_ss": jnp.array([
            0.0, 0.2, 0.0,
            0.0, 0.2, 0.0,
            0.0,
            0.0, 0.2, 0.0,
            0.0, 0.2, 0.0,
            0.0, 0.2, 0.0,
        ]),
    },
    # # cw staircase bending with alternating payload
    # "20230703_162136": {
    #     "t_ss": jnp.array(
    #         [
    #             11.3,
    #             36.5,
    #             68.2,
    #             82.5,
    #             111.5,
    #             138.33,
    #             147.7,
    #             171.3,
    #             206.6,
    #             213,
    #             233,
    #             276.3,
    #             289.2,
    #             321.4,
    #             345.6,
    #             415,
    #         ]
    #     ),
    #     "mpl_ss": jnp.array(
    #         [
    #             0.0,
    #             0.2,
    #             0.0,
    #             0.0,
    #             0.2,
    #             0.0,
    #             0.0,
    #             0.2,
    #             0.0,
    #             0.0,
    #             0.2,
    #             0.0,
    #             0.0,
    #             0.2,
    #             0.0,
    #             0.0,
    #         ]
    #     ),
    # }
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
            Path(__file__).parent.parent.parent.parent
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

        # subtract shear and bending of first time step to zero
        experiment_data_ts["xi_ts"] = (
            experiment_data_ts["xi_ts"]
            .at[:, :2]
            .add(-experiment_data_ts["xi_ts"][0, :2])
        )

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
    onp.savetxt("Pi_fpu_static_bending_est.csv", Pi_est, delimiter=",")
