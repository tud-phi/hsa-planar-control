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

from hsa_planar_control.system_identification.analysis import (
    analyze_neutral_rod_length_model,
)
from hsa_planar_control.system_identification.optimization.linear_lq import (
    linear_lq_optim_problem_factory,
    optimize_with_closed_form_linear_lq,
)
from hsa_planar_control.system_identification.preprocessing import preprocess_data
from hsa_planar_control.system_identification.rest_strain import (
    identify_axial_rest_strain_for_system_id_dataset,
)


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
    # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
    "C_varepsilon": 0.0 * ones_rod,
    # outside radius of each rod [m]. The rows correspond to the segments.
    "rout": 25.4e-3 / 2 * ones_rod,  # this is for FPU rods
    # inside radius of each rod [m]. The rows correspond to the segments.
    "rin": (25.4e-3 / 2 - 4.76e-3) * ones_rod,  # this is for EPU rods
    # handedness of each rod. The rows correspond to the segments.
    "h": jnp.array([[1.0, -1.0, 1.0, -1.0]]),
    # offset [m] of each rod from the centerline. The rows correspond to the segments.
    "roff": 24e-3 * jnp.array([[1.0, 1.0, -1.0, -1.0]]),
    "pcudim": jnp.array(
        [[80e-3, 12e-3, 80e-3]]
    ),  # width, height, depth of the platform [m]
    # mass of EPU rod: 26 g
    # For EPU, this corresponds to a measure volume of 0000314034 m^3 --> rho = 827.94 kg/m^3
    "rhor": 827.94 * ones_rod,  # Volumetric density of rods [kg/m^3],
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
    "S_b_hat": 0.0 * ones_rod,
    # Nominal shear stiffness of each rod [N]
    "S_sh_hat": 0.0 * ones_rod,
    # Nominal axial stiffness of each rod [N]
    "S_a_hat": 0.0 * ones_rod,
    # Elastic coupling between bending and shear [Nm/rad]
    "S_b_sh": 0.0 * ones_rod,
    # Scaling of axial stiffness with twist strain [Nm/rad]
    "C_S_a": 0.0 * ones_rod,
    # Scaling of bending stiffness with twist strain [Nm^3/rad]
    "C_S_b": 0.0 * ones_rod,
    # Scaling of shear stiffness with twist strain [Nm/rad]
    "C_S_sh": 0.0 * ones_rod,
    # center of origin of the payload relative to end-effector [m]
    # subtract 12 mm for the thickness of the platform
    # the 100g weights have a length of 27mm
    "CoGpl": jnp.array([0.0, -12e-3 - 13.5e-3]),
    # rigid offset SE(2) transformation from the distal end of the platform to the end-effector
    "chiee_off": jnp.array([0.0, 0.0, 0.0]),
}

main_deformation_mode = None
if SYSTEM_ID_STEP == 0:
    params_to_be_idd_names = ["sigma_a_eq", "S_a_hat"]
    # identified parameters from step 0:
    # sigma_a_eq = 0.81200092
    # S_a_hat = 0.73610293
    main_deformation_mode = "elongation"

    # set dummy parameters for C_varepsilon and C_S_a
    known_params["C_varepsilon"] = 0.0 * ones_rod
    known_params["C_S_a"] = 0.0 * ones_rod

    # Staircase elongation with changing mass and actuation up to 210 deg
    # At each step, first 0g payload mass, then 200g, then 400g, then 400g g, then 0g
    experiment_id = "20230927_150929"
    t_ss = jnp.array(
        [
            1.6,
            2.8,
            5.0,
            11.3,
            12.2,
            13.2,
            19.8,
            20.8,
            21.3,
            27.3,
            28.6,
            29.5,
            43.4,
            50.3,
            55.3,
        ]
    )
    mpl_ss = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.2,
            0.4,
            0.4,
            0.4,
            0.2,
            0.2,
            0.2,
            0.0,
            0.0,
            0.0,
        ]
    )
elif SYSTEM_ID_STEP == 1:
    params_to_be_idd_names = ["C_varepsilon"]
    # identified parameters from step 1:
    # C_varepsilon = 0.0079049
    main_deformation_mode = "elongation"

    # previously identified parameters in step 0
    known_params["S_a_hat"] = 0.73610293 * ones_rod
    known_params["C_S_a"] = (
        0.0 * ones_rod
    )  # we assume that change of S_a is negligible without payload

    # Staircase elongation with changing mass up to 210 deg
    # only regard samples with 0g payload mass
    experiment_id = "20230927_150929"
    t_ss = jnp.array(
        [
            1.6,  # 0th step
            68.0,
            74.7,  # 1st step
            138.0,
            145.0,  # 2nd step
            207.1,
            211.4,  # 3rd step
            273.6,
            284.8,  # 4th step
            344.9,
            351.4,  # 5th step
            415.0,
        ]
    )
    mpl_ss = jnp.array(
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
            0.0,
            0.0,
            0.0,
        ]
    )
elif SYSTEM_ID_STEP == 2:
    params_to_be_idd_names = ["C_S_a"]
    # identified parameters from step 2:
    # C_S_a = 0.00981059
    main_deformation_mode = "elongation"

    # previously identified parameters in steps 0 and 1
    known_params["S_a_hat"] = 0.73610293 * ones_rod
    known_params["C_varepsilon"] = 0.0079049 * ones_rod

    # Staircase elongation with changing mass up to 210 deg
    # At each step, first 0g payload mass, then 200g, then 400g, then 200 g, then 0g
    experiment_id = "20230927_150929"
    t_ss = jnp.array(
        [
            5.0,  # 0th step
            13.2,
            21.3,
            29.5,
            55.3,
            76.1,  # 1st step
            92.0,
            104.4,
            119.3,
            138.3,
            145.1,  # 2nd step
            160.2,
            177.5,
            190.8,
            207.5,
            211.6,  # 3rd step
            227.0,
            239.4,
            248.8,
            276.8,
            286.6,  # 4th step
            298.4,
            311.8,
            323.3,
            346.2,
            353.5,  # 5th step
            370.0,
            382.2,
            390.8,
            414.5,
        ]
    )
    mpl_ss = jnp.array(
        [
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
        ]
    )
elif SYSTEM_ID_STEP == 3:
    params_to_be_idd_names = ["S_b_hat", "S_sh_hat", "S_b_sh", "C_S_b", "C_S_sh"]
    # identified parameters based on 20230927_170823:
    # S_b_hat = 4.02729650e-05, S_sh_hat = 9.69715686e-03, S_b_sh = 5.68054241e-05,
    # C_S_b = -2.93841224e-06, C_S_sh = 8.36960975e-03
    # identified parameters based on 20230927_171719:
    # S_b_hat = -2.53938133e-05, S_sh_hat = 4.28135773e-02, S_b_sh = 5.04204068e-04,
    # C_S_b = 3.90666351e-07, C_S_sh = 2.93344701e-03
    main_deformation_mode = "bending"

    # previously identified parameters in steps 0, 1, and 2
    known_params["S_a_hat"] = 0.73610293 * ones_rod
    known_params["C_varepsilon"] = 0.0079049 * ones_rod
    known_params["C_S_a"] = 0.00981059 * ones_rod

    # # Staircase bending cw with changing mass up to 210 deg
    # # At each step, first 0g payload mass, then 200g, then 400g, then 200 g, then 0g
    # experiment_id = "20230927_170823"
    # t_ss = jnp.array(
    #     [
    #         2.7,  # 0th step
    #         20.7,
    #         31.3,
    #         40.8,
    #         69.4,
    #         73.5,  # 1st step
    #         85.7,
    #         97.1,
    #         105.0,
    #         138.7,
    #         219.6,  # 3rd step
    #         235.8,
    #         248.7,
    #         259.0,
    #         277.2,
    #         284.2,  # 4th step
    #         301.4,
    #         312.8,
    #         321.8,
    #         346.4,
    #     ]
    # )
    # mpl_ss = jnp.array(
    #     [
    #         0.0,
    #         0.2,
    #         0.4,
    #         0.2,
    #         0.0,
    #         0.0,
    #         0.2,
    #         0.4,
    #         0.2,
    #         0.0,
    #         0.0,
    #         0.2,
    #         0.4,
    #         0.2,
    #         0.0,
    #         0.0,
    #         0.2,
    #         0.4,
    #         0.2,
    #         0.0,
    #     ]
    # )
    # Staircase bending cw with changing mass up to 270 deg
    # At each step, first 0g payload mass, then 200g, then 400g, then 200 g, then 0g
    experiment_id = "20230927_171719"
    t_ss = jnp.array(
        [
            2.7,  # 0th step
            15.5,
            25.7,
            33.2,
            69.1,
            74.6,  # 1st step
            87.0,
            99.5,
            107.0,
            138.7,
            144.3,  # 2nd step
            155.2,
            164.7,
            173.8,
            208.0,
            218.8,  # 3rd step
            232.7,
            244.8,
            252.9,
            276.0,
            285.4,  # 4th step
            301.1,
            314.0,
            321.0,
            346.0,
            353.9,  # 5th step
            365.6,
            378.0,
            388.5,
            416.0,
        ]
    )
    mpl_ss = jnp.array(
        [
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
            0.0,
            0.2,
            0.4,
            0.2,
            0.0,
        ]
    )
else:
    raise ValueError("SYSTEM_ID_STEP must be 0, 1, 2, or 3.")

mocap_body_ids = {"base": 3, "platform": 4}
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

    # identify axial rest strain
    if "sigma_a_eq" not in params_to_be_idd_names:
        known_params["sigma_a_eq"] = identify_axial_rest_strain_for_system_id_dataset(
            sym_exp_filepath,
            sys_helpers,
            known_params,
            experiment_data_ts,
        )

    t_ts = experiment_data_ts["t_ts"]  # array of timestamps of entire trajectory [s]

    # for each entry of t_ss, find the closest entry in t_ts
    t_ss = jnp.array([t_ts[jnp.argmin(jnp.abs(t_ts - t_ss_i))] for t_ss_i in t_ss])
    print("Selected the following timestamps for steady-state data: ", t_ss)
    # select all values of t_ts with entries in t_ss
    t_ts_selector = jnp.isin(t_ts, t_ss)

    data_ts = {
        "t_ts": t_ss,
        "xi_ts": experiment_data_ts["xi_ts"][t_ts_selector, ...],
        "xi_d_ts": experiment_data_ts["xi_d_ts"][t_ts_selector, ...],
        "xi_dd_ts": experiment_data_ts["xi_dd_ts"][t_ts_selector, ...],
        "phi_ts": experiment_data_ts["phi_ts"][t_ts_selector, ...],
        "mpl_ts": mpl_ss,
    }

    if main_deformation_mode == "elongation":
        # manually set bending and shear strains to zero
        data_ts["xi_ts"] = (
            data_ts["xi_ts"].at[:, 0].set(1e-4 * jnp.ones((len(t_ss),)))
        )  # almost zero to avoid singularities
        data_ts["xi_ts"] = data_ts["xi_ts"].at[:, 1].set(jnp.zeros((len(t_ss),)))
    elif main_deformation_mode == "bending":
        # subtract shear and bending of first time step to zero
        experiment_data_ts["xi_ts"] = (
            experiment_data_ts["xi_ts"]
            .at[:, :2]
            .add(-experiment_data_ts["xi_ts"][0, :2])
        )

    print("Running linear least-squares optimization...")
    Pi_syms, cal_a_fn, cal_b_fn = linear_lq_optim_problem_factory(
        sym_exp_filepath,
        sys_helpers,
        known_params,
        params_to_be_idd_names,
        mode="static",
    )
    Pi_est = optimize_with_closed_form_linear_lq(
        cal_a_fn,
        cal_b_fn,
        data_ts,
    )

    print(f"Identified system params {Pi_syms} using steady-state samples:\n", Pi_est)
    onp.savetxt("Pi_epu_static_elongation_nlq_est.csv", Pi_est, delimiter=",")

    if SYSTEM_ID_STEP == 1:
        params = known_params.copy()
        params["C_varepsilon"] = Pi_est[0]
        analyze_neutral_rod_length_model(params, data_ts)
