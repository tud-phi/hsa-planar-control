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
from hsa_planar_control.system_identification.optimization.nonlinear_lq import (
    nonlinear_lq_optim_problem_factory,
    optimize_with_nonlinear_lq,
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

SYSTEM_ID_STEP = 0
PLOT_DATASET = False

ones_rod = jnp.ones((num_segments, num_rods_per_segment))
known_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
    # length of the rigid proximal caps of the rods connecting to the base [m]
    "lpc": 25e-3 * jnp.ones((num_segments,)),
    # length of the rigid distal caps of the rods connecting to the platform [m]
    "ldc": 14e-3 * jnp.ones((num_segments,)),
    "kappa_b_eq": 0.0 * ones_rod,  # bending rest strain [rad/m]
    "sigma_sh_eq": 0.0 * ones_rod,  # shear rest strain [-]
    "sigma_a_eq": 1.0 * ones_rod,  # axial rest strain [-]
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
    # Nominal bending stiffness of each rod [Nm^2]
    "S_b_hat": 5.4698261027774997e-5 * ones_rod,
    # Nominal shear stiffness of each rod [N]
    "S_sh_hat": 0.9620376027360 * ones_rod,
    # Elastic coupling between bending and shear [Nm/rad]
    "S_b_sh": 7.56629739e-03 * ones_rod,
    # Scaling of bending stiffness with twist strain [Nm^3/rad]
    "C_S_b": 7.92251400952920015e-7 * ones_rod,
    # Scaling of shear stiffness with twist strain [Nm/rad]
    "C_S_sh": -3.85580745914e-3 * ones_rod,
    # center of origin of the payload relative to end-effector [m]
    # subtract 12 mm for the thickness of the platform
    # the 100g weights have a length of 27mm
    "CoGpl": jnp.array(
        [0.0, -12e-3 - 5e-3]
    ),
    # rigid offset SE(2) transformation from the distal end of the platform to the end-effector
    "chiee_off": jnp.array([0.0, 0.0, 0.0]),
}

if SYSTEM_ID_STEP == 0:
    optimization_type = "llq"
    params_to_be_idd_names = ["sigma_a_eq", "S_a_hat"]
    Pi_init = jnp.array([9.98051512e-01, 0.820156460844])
    # identified parameters from step 0:
    # sigma_a_eq = 1.063278732
    # S_a_hat = 5.66472469

    # set dummy parameters for C_varepsilon and C_S_a
    # known_params["sigma_a_eq"] = 1.0 * 0.0 * ones_rod
    known_params["C_varepsilon"] = 0.0 * ones_rod
    known_params["C_S_a"] = 0.0 * ones_rod

    experiment_configs = {
        # Staircase elongation with changing mass up to 210 deg
        # At each step, first 0g payload mass, then 437g, then 637g, then 437 g, then 0g
        "20230703_115411": {
            "t_ss": jnp.array(
                [
                    1.6,
                    1.8,
                    2.26,
                    11.76,
                    14.4,
                    16.0,
                    22.4,
                    23.6,
                    24.8,
                    28.6,
                    29.4,
                    30.4,
                    37.2,
                    53.12,
                    68.4,
                ]
            ),
            "mpl_ss": jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.437,
                    0.437,
                    0.437,
                    0.637,
                    0.637,
                    0.637,
                    0.437,
                    0.437,
                    0.437,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        },
    }
elif SYSTEM_ID_STEP == 1:
    optimization_type = "llq"
    params_to_be_idd_names = ["C_varepsilon"]
    # identified parameters from step 1:
    # C_varepsilon = 0.00984819

    # previously identified parameters in step 0
    known_params["sigma_a_eq"] = 1.06327873 * ones_rod
    known_params["S_a_hat"] = 5.66472469 * ones_rod
    known_params["C_S_a"] = (
        0.0 * ones_rod
    )  # we assume that change of S_a is negligible without payload

    experiment_configs = {
        # Staircase elongation with changing mass up to 210 deg
        # only regard samples with 0g payload mass
        "20230703_115411": {
            "t_ss": jnp.array(
                [
                    72.2,
                    126.7,
                    140.6,
                    197.57,
                    222,
                    281.6,
                    296,
                    349.8,
                    415,
                ]
            ),
            "mpl_ss": jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        },
    }
elif SYSTEM_ID_STEP == 2:
    optimization_type = "llq"
    params_to_be_idd_names = ["C_S_a"]
    # identified parameters from step 2:
    # C_S_a = 0.01508165

    # previously identified parameters in step 0
    known_params["sigma_a_eq"] = 1.06327873 * ones_rod
    known_params["S_a_hat"] = 5.66472469 * ones_rod
    known_params["C_varepsilon"] = 0.00984819 * ones_rod

    experiment_configs = {
        # Staircase elongation with changing mass up to 210 deg
        # At each step, first 0g payload mass, then 437g, then 637g, then 437 g, then 0g
        "20230703_115411": {
            "t_ss": jnp.array(
                [
                    72.2,
                    83.7,
                    95.8,
                    105.13,
                    126.7,
                    140.6,
                    159.5,
                    168.16,
                    176.09,
                    197.57,
                    222,
                    232,
                    241,
                    272.3,
                    281.6,
                    296,
                    307.8,
                    316.5,
                    346.3,
                    349.8,
                    361.6,
                    377.6,
                    387.52,
                    415,
                ]
            ),
            "mpl_ss": jnp.array(
                [
                    0.0,
                    0.437,
                    0.637,
                    0.437,
                    0.0,
                    0.0,
                    0.437,
                    0.637,
                    0.437,
                    0.0,
                    0.437,
                    0.637,
                    0.437,
                    0.0,
                    0.0,
                    0.437,
                    0.637,
                    0.437,
                    0.0,
                    0.0,
                    0.437,
                    0.637,
                    0.437,
                    0.0,
                ]
            ),
        },
    }
elif SYSTEM_ID_STEP == 3:
    optimization_type = "llq"
    params_to_be_idd_names = ["sigma_a_eq"]
    # identification result for 20230621_165020:
    # sigma_a_eq = 1.03195326
    # identification result for 20230703_155911:
    # sigma_a_eq = 1.0753753

    # previously identified parameters in steps 0, 1, 2
    known_params["S_a_hat"] = 5.66472469 * ones_rod
    known_params["C_S_a"] = 0.01508165 * ones_rod
    known_params["C_varepsilon"] = 0.00984819 * ones_rod

    experiment_configs = {
        # staircase of elongation
        # "20230621_153408": {
        #     "t_ss": jnp.array([0.8, 2.0, 3.0, 4.0, 5.0, 6.0]),
        #     "mpl_ss": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # },
        # # staircase of bending to 180 deg
        # "20230621_165020": {
        #     "t_ss": jnp.array([0.8, 2.0, 3.0, 4.0, 5.0, 6.0]),
        #     "mpl_ss": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # },
        # cw staircase bending with alternating payload
        "20230703_155911": {
            "t_ss": jnp.array([3.3, 5.6, 9.0, 50, 51, 52, 53, 54, 55, 56]),
            "mpl_ss": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        # # GBN bending combined 180 deg
        # "20230621_183620": {
        #     "t_ss": jnp.array([1.05]),
        #     "mpl_ss": jnp.array([0.0]),
        # }
    }

else:
    raise ValueError("SYSTEM_ID_STEP must be 0, 1, 2, or 3.")

# experiment_configs = {
#     # staircase of elongation to 210 deg
#     "20230621_153408": {
#         "t_ss": jnp.array([5.0, 10.0, 16.6, 22.2, 28.6, 34.0]),
#         "mpl_ss": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
#     },
#     # step of elongation to 90 deg
#     "20230621_170058": {
#         "t_ss": jnp.array([10.0, 22.0]),
#         "mpl_ss": jnp.array([0.0, 0.0]),
#     },
#     # step of elongation to 120 deg
#     "20230621_170509": {
#         "t_ss": jnp.array([10.0, 22.0]),
#         "mpl_ss": jnp.array([0.0, 0.0]),
#     },
#     # step of elongation to 180 deg
#     "20230621_170624": {
#         "t_ss": jnp.array([10.0, 22.0]),
#         "mpl_ss": jnp.array([0.0, 0.0]),
#     },
#     # step of elongation to 210 deg
#     "20230621_170734": {
#         "t_ss": jnp.array([10.0, 22.0]),
#         "mpl_ss": jnp.array([0.0, 0.0]),
#     },
# }

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
        sys_helpers,
        known_params,
        params_to_be_idd_names,
        mode="static",
    )
    eom_residual_fn = nonlinear_lq_optim_problem_factory(
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
            plotting=PLOT_DATASET,
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

        # manually set bending and shear strains to zero
        experiment_selected_ss_data_ts["xi_ts"] = (
            experiment_selected_ss_data_ts["xi_ts"]
            .at[:, 0]
            .set(1e-4 * jnp.ones((len(t_ss),)))
        )  # almost zero to avoid singularities
        experiment_selected_ss_data_ts["xi_ts"] = (
            experiment_selected_ss_data_ts["xi_ts"]
            .at[:, 1]
            .set(jnp.zeros((len(t_ss),)))
        )

        if data_ts is None:
            data_ts = experiment_selected_ss_data_ts
        else:
            data_ts = {
                k: jnp.concatenate((v, experiment_selected_ss_data_ts[k]))
                for k, v in data_ts.items()
            }

    if optimization_type == "llq":
        print("Running linear least-squares optimization...")
        Pi_est = optimize_with_closed_form_linear_lq(
            cal_a_fn,
            cal_b_fn,
            data_ts,
        )
    elif optimization_type == "nlq":
        print("Running nonlinear least-squares optimization...")
        Pi_est = optimize_with_nonlinear_lq(
            eom_residual_fn,
            data_ts,
            Pi_init=Pi_init,
        )
    else:
        raise ValueError("Unknown optimization type: ", optimization_type)

    print(f"Identified system params {Pi_syms} using steady-state samples:\n", Pi_est)
    onp.savetxt("Pi_fpu_static_elongation_nlq_est.csv", Pi_est, delimiter=",")
