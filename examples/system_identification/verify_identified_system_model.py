from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import Dopri5, Euler
from functools import partial
from jax import Array, jit, lax, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_SYSTEM_ID
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Tuple

from hsa_planar_control.collocated_form import mapping_into_collocated_form_factory
from hsa_planar_control.rendering.opencv_renderer import animate_robot
from hsa_planar_control.simulation import simulate_closed_loop_system
from hsa_planar_control.system_identification.preprocessing import preprocess_data

num_segments = 1
num_rods_per_segment = 4

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

# set parameters
params = PARAMS_SYSTEM_ID.copy()

# experiment id
# experiment_id = "20230621_153408"  # staircase elongation
experiment_id = "20230621_165020"  # staircase bending ccw
# experiment_id = "20230621_170734"  # step elongation 210 deg
# experiment_id = "20230621_171345"  # step bending cw 180 deg
# experiment_id = "20230621_182829"  # GBN elongation 180 deg
# experiment_id = "20230621_183620"  # GBN bending combined 180 deg
if experiment_id == "20230621_153408":
    params["sigma_a_eq"] = 1.0388612 * jnp.ones_like(params["sigma_a_eq"])
elif experiment_id == "20230621_165020":
    params["sigma_a_eq"] = 1.03195326 * jnp.ones_like(params["sigma_a_eq"])
elif experiment_id == "20230621_183620":
    params["sigma_a_eq"] = 1.102869 * jnp.ones_like(params["sigma_a_eq"])

# settings for simulation
mocap_body_ids = {"base": 4, "platform": 5}
sim_dt = 5e-5  # time step for simulation [s]

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
    ) -> Tuple[Array, Dict[str, Array]]:
        time_idx = (t / data_dt).astype(int)
        phi_des = data_ts["phi_ts"][time_idx]

        return phi_des, controller_state

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
        ode_solver_class=Euler,
    )

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