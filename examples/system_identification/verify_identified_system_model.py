import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import Dopri5, Euler
from functools import partial
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_SYSTEM_ID, PARAMS_EPU_SYSTEM_ID
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Tuple

from hsa_planar_control.rendering.opencv_renderer import animate_robot
from hsa_planar_control.simulation import simulate_closed_loop_system
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

# experiment id
# experiment_id = "20230621_153408"  # FPU staircase elongation
# experiment_id = "20230621_165020"  # FPU staircase bending ccw
# experiment_id = "20230621_170734"  # FPU step elongation 210 deg
# experiment_id = "20230621_171345"  # FPU step bending cw 180 deg
# experiment_id = "20230621_182829"  # FPU GBN elongation 180 deg
# experiment_id = "20230621_183620"  # FPU GBN bending combined 180 deg
# experiment_id = "20230927_143724"  # EPU staircase bending cw 210 deg
# experiment_id = "20230927_143824"  # EPU staircase bending cw 270 deg
# experiment_id = "20230927_144354"  # EPU staircase bending ccw 210 deg
# experiment_id = "20230927_144511"  # EPU staircase bending ccw 270 deg
# experiment_id = "20230927_150331"  # EPU GBN bending combined 210 deg
experiment_id = "20230927_150452"  # EPU GBN bending combined 270 deg
hsa_material = "epu"

# set parameters
if hsa_material == "fpu":
    params = PARAMS_FPU_SYSTEM_ID.copy()
elif hsa_material == "epu":
    params = PARAMS_EPU_SYSTEM_ID.copy()

# settings for simulation
if hsa_material == "fpu":
    mocap_body_ids = {"base": 4, "platform": 5}
else:
    mocap_body_ids = {"base": 3, "platform": 4}
sim_dt = 1e-4  # time step for simulation [s]

if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    experiment_data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "system_identification"
        / experiment_id
    )
    data_ts = preprocess_data(
        inverse_kinematics_end_effector_fn,
        experiment_data_path,
        params,
        mocap_body_ids,
        filter=False,
        derivative_method="savgol_filter",
        plotting=False,
    )
    with open(
        experiment_data_path / f"preprocessed_data_history.dill", "wb"
    ) as data_ts_file:
        dill.dump(data_ts, data_ts_file)

    # identify rest strains
    params["sigma_a_eq"] = identify_axial_rest_strain_for_system_id_dataset(
        sym_exp_filepath,
        sys_helpers,
        params,
        data_ts,
        num_time_steps=1,
        separate_rods=True,
    )
    print(f"Using axial rest strains:\n{params['sigma_a_eq']}")

    data_dt = data_ts["t_ts"][1] - data_ts["t_ts"][0]
    # shift time stamps to start at zero
    data_ts["t_ts"] = data_ts["t_ts"] - data_ts["t_ts"][0]
    xi_eq = sys_helpers["rest_strains_fn"](params)
    # define initial configuration
    q0 = data_ts["xi_ts"][0] - xi_eq
    q_d0 = data_ts["xi_d_ts"][0]
    phi0 = data_ts["phi_ts"][0]

    def control_fn(
        t: Array, q: Array, q_d: Array, phi: Array, controller_state: Dict[str, Array]
    ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        time_idx = (t / data_dt).astype(int)
        phi_des = data_ts["phi_ts"][time_idx]

        controller_info = {}

        return phi_des, controller_state, controller_info

    sim_ts = simulate_closed_loop_system(
        dynamical_matrices_fn,
        params,
        q0=q0,
        q_d0=q_d0,
        phi0=phi0,
        sim_dt=sim_dt,
        control_dt=data_dt.item(),
        duration=data_ts["t_ts"][-1].item(),
        control_fn=control_fn,
        controller_state_init={},
        ode_solver_class=Dopri5,
    )
    sim_ts["xi_ts"] = xi_eq + sim_ts["q_ts"]
    sim_ts["chiee_ts"] = vmap(
        partial(forward_kinematics_end_effector_fn, params),
    )(sim_ts["q_ts"])
    with open(
        experiment_data_path / f"model_inference_history.dill", "wb"
    ) as sim_ts_file:
        dill.dump(sim_ts, sim_ts_file)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    B_xi = sys_helpers["B_xi"]
    plt.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 0],
        color=colors[0],
        label=r"$\kappa_\mathrm{b}$ [rad / m]",
    )
    plt.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 1],
        color=colors[1],
        label=r"$\sigma_\mathrm{sh}$ [-]",
    )
    plt.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 2],
        color=colors[2],
        label=r"$\sigma_\mathrm{a}$ [-]",
    )
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[0] + sim_ts["q_ts"][:, 0],
        color=colors[0],
        linestyle="--",
        label=r"$\hat{\kappa}_\mathrm{b}$ [rad / m]",
    )
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[1] + sim_ts["q_ts"][:, 1],
        color=colors[1],
        linestyle="--",
        label=r"$\hat{\sigma}_\mathrm{sh}$ [-]",
    )
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[2] + sim_ts["q_ts"][:, 2],
        color=colors[2],
        linestyle="--",
        label=r"$\hat{\sigma}_\mathrm{a}$ [-]",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Strain of virtual backbone")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(
        data_ts["t_ts"],
        data_ts["phi_ts"][:, 0],
        linestyle="-",
        label=r"$\phi_1^\mathrm{d}$ [rad]",
    )
    plt.plot(
        data_ts["t_ts"],
        data_ts["phi_ts"][:, 1],
        linestyle="-",
        label=r"$\phi_2^\mathrm{d}$ [rad]",
    )
    plt.plot(
        sim_ts["t_ts"],
        sim_ts["phi_ts"][:, 0],
        linestyle="--",
        label=r"$\hat{\phi}_1^\mathrm{d}$ [rad]",
    )
    plt.plot(
        sim_ts["t_ts"],
        sim_ts["phi_ts"][:, 1],
        linestyle="--",
        label=r"$\hat{\phi}_2^\mathrm{d}$ [rad]",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Twist angles [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
