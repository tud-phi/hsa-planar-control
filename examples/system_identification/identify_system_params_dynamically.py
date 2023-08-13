from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
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


ones_rod = jnp.ones((num_segments, num_rods_per_segment))
known_params = {
    "th0": jnp.array(0.0),  # initial orientation angle [rad]
    "l": 59e-3 * jnp.ones((num_segments,)),  # length of each rod [m]
    # length of the rigid proximal caps of the rods connecting to the base [m]
    "lpc": 25e-3 * jnp.ones((num_segments,)),
    # length of the rigid distal caps of the rods connecting to the platform [m]
    "ldc": 14e-3 * jnp.ones((num_segments,)),
    "sigma_a_eq": 1.08094014 * ones_rod,  # axial rest strains of each rod
    # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
    "C_varepsilon": 0.00984819 * ones_rod,  # Previous: 9.1e-3
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
    "S_b_hat": 5.71346377e-04 * ones_rod,
    # Nominal shear stiffness of each rod [N]
    "S_sh_hat": 5.91462074e-01 * ones_rod,
    # Nominal axial stiffness of each rod [N]
    "S_a_hat": 5.66472469 * ones_rod,
    # Elastic coupling between bending and shear [Nm/rad]
    "S_b_sh": 4.48419541e-03 * ones_rod,
    # Scaling of bending stiffness with twist strain [Nm^3/rad]
    "C_S_b": -9.67560524e-06 * ones_rod,
    # Scaling of shear stiffness with twist strain [Nm/rad]
    "C_S_sh": -4.75687961e-04 * ones_rod,
    # Scaling of axial stiffness with twist strain [Nm/rad]
    "C_S_a": 0.01508165 * ones_rod,
}

experiment_ids = [
    "20230621_153408",  # staircase elongation
    "20230621_165020",  # staircase bending ccw
    "20230621_170734",  # step elongation 210 deg
    "20230621_171345",  # step bending cw 180 deg
    "20230621_182829",  # GBN elongation 180 deg
    "20230621_183620",  # GBN bending combined 180 deg
]
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

    params_to_be_idd_names = ["zetab", "zetash", "zetaa"]
    Pi_syms, cal_a_fn, cal_b_fn = linear_lq_optim_problem_factory(
        sym_exp_filepath,
        dynamical_matrices_fn,
        sys_helpers,
        known_params,
        params_to_be_idd_names,
        mode="dynamic",
    )

    Pi_est_es = jnp.zeros((len(experiment_ids), len(Pi_syms)))
    for experiment_idx, experiment_id in enumerate(experiment_ids):
        experiment_data_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "system_identification"
            / experiment_id
        )
        data_ts = preprocess_data(
            inverse_kinematics_end_effector_fn,
            experiment_data_path,
            known_params,
            mocap_body_ids,
            resampling_dt=resampling_dt,
            filter=True,
            derivative_method="savgol_filter",
            plotting=False,
        )

        Pi_est = optimize_with_closed_form_linear_lq(cal_a_fn, cal_b_fn, data_ts)
        Pi_est_es = Pi_est_es.at[experiment_idx, :].set(Pi_est)

    print("Identified system params for each experiment:\n", Pi_est_es)
    onp.savetxt("Pi_dynamic_est_es.csv", Pi_est_es, delimiter=",")
