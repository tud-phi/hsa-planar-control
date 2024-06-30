from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import Dopri5, Euler, Tsit5
from functools import partial
from jax import Array, debug, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Tuple

from hsa_planar_control.controllers.operational_space_controllers import (
    basic_operational_space_pid,
    operational_space_pd_plus_linearized_actuation,
    operational_space_pd_plus_nonlinear_actuation,
    operational_space_impedance_control_nonlinear_actuation,
)
from hsa_planar_control.controllers.saturation import saturate_control_inputs
from hsa_planar_control.simulation import simulate_closed_loop_system

from jsrm.rendering.planar_hsa.opencv_renderer import animate_robot

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

# set parameters
params = PARAMS_FPU_CONTROL.copy()
rho_perturb_factor = (
    1.0  # perturb the mass density of the robot used for the simulation
)
params_perturbed = params.copy()
params_perturbed.update(
    {
        "rhor": rho_perturb_factor * params["rhor"],
        "rhop": rho_perturb_factor * params["rhop"],
        "rhoec": rho_perturb_factor * params["rhoec"],
    }
)

# define initial configuration
q0 = jnp.array([0.0, 0.0, 0.0])
q_d0 = jnp.zeros_like(q0)
phi0 = jnp.array([0.0, 0.0])

pee_des = jnp.array([0.000, 0.13])
# pee_des = jnp.array([0.037, 0.120])
# pee_des = jnp.array([-0.05, 0.11])
# pee_des = jnp.array([0.01, 0.120])

duration = 5.0
sim_dt = 5e-5  # time step

# control settings
# available controller types: basic_operational_space_pid, operational_space_impedance_control_nonlinear_actuation
controller_type = "operational_space_impedance_control_nonlinear_actuation"
control_dt = 2e-2  # control time step. corresponds to 50 Hz
match controller_type:
    case "basic_operational_space_pid":
        Kp = 1.0e1 * jnp.eye(2)  # [rad/m]
        Ki = 1.1e2 * jnp.eye(2)  # [rad/(ms)]
        Kd = 2.5e-1 * jnp.eye(2)  # [Ns/m]
    case "operational_space_impedance_control_nonlinear_actuation":
        Kp = 5e0 * jnp.eye(2)  # [N/m]
        Kd = 0e0 * jnp.eye(2)  # [Ns/m]
    case _:
        raise ValueError(
            f"There does not exist an implementation for {controller_type}"
        )

# video settings
video_path = Path("videos") / f"controlled_planar_hsa_{controller_type}.mp4"

if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    controller_state_init = None
    match controller_type:
        case "basic_operational_space_pid":
            control_fn = jit(
                partial(
                    basic_operational_space_pid,
                    dt=control_dt,
                    phi_ss=params["phi_max"].squeeze() / 2,
                    pee_des=pee_des,
                    Kp=Kp,
                    Ki=Ki,
                    Kd=Kd,
                )
            )
            controller_state_init = {
                "e_pee": jnp.zeros_like(pee_des),
                "integral_error": jnp.zeros_like(pee_des),
            }
        case (
            "operational_space_pd_plus_linearized_actuation"
            | "operational_space_pd_plus_nonlinear_actuation"
            | "operational_space_impedance_control_nonlinear_actuation"
        ):
            if controller_type == "operational_space_pd_plus_linearized_actuation":
                control_fn = operational_space_pd_plus_linearized_actuation
            elif controller_type == "operational_space_pd_plus_nonlinear_actuation":
                control_fn = operational_space_pd_plus_nonlinear_actuation
            else:
                control_fn = operational_space_impedance_control_nonlinear_actuation

            dynamics_eps = 1e-1
            control_fn = jit(
                partial(
                    control_fn,
                    dynamical_matrices_fn=partial(
                        dynamical_matrices_fn, params, eps=dynamics_eps
                    ),
                    operational_space_dynamical_matrices_fn=partial(
                        sys_helpers["operational_space_dynamical_matrices_fn"],
                        params,
                        eps=dynamics_eps,
                    ),
                    pee_des=pee_des,
                    Kp=Kp,
                    Kd=Kd,
                    eps=dynamics_eps,
                )
            )
        case _:
            raise ValueError(
                f"There does not exist an implementation for {controller_type}"
            )

    @jit
    def saturated_control_fn(
        *args, **kwargs
    ) -> Tuple[Array, Dict[str, Array], Dict[str, Array]]:
        controller_output = control_fn(*args, **kwargs)
        controller_output = saturate_control_inputs(params, *controller_output)
        return controller_output

    chiee = forward_kinematics_end_effector_fn(params, q0)
    chiee_d = jacobian_end_effector_fn(params, q0) @ q_d0
    phi_des, controller_info = control_fn(0.0, chiee, chiee_d, q=q0, q_d=q_d0, phi=phi0)

    sim_ts = simulate_closed_loop_system(
        dynamical_matrices_fn,
        params_perturbed,
        q0=q0,
        q_d0=q_d0,
        phi0=phi0,
        sim_dt=sim_dt,
        control_dt=control_dt,
        duration=duration,
        control_fn=saturated_control_fn,
        controller_state_init=controller_state_init,
        ode_solver_class=Euler,
        control_in_operational_space=True,
        forward_kinematics_end_effector_fn=forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn=jacobian_end_effector_fn,
    )

    print("Last configuration:", sim_ts["q_ts"][-1])
    print(
        "Last end-effector pose:",
        forward_kinematics_end_effector_fn(params, sim_ts["q_ts"][-1]),
    )

    sim_ts["chiee_ts"] = vmap(forward_kinematics_end_effector_fn, in_axes=(None, 0))(
        params, sim_ts["q_ts"]
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.plot(
        sim_ts["t_ts"],
        sim_ts["chiee_ts"][:, 0],
        color=colors[0],
        label=r"$p_\mathrm{ee,x}$",
    )
    plt.plot(
        sim_ts["t_ts"],
        sim_ts["chiee_ts"][:, 1],
        color=colors[1],
        label=r"$p_\mathrm{ee,y}$",
    )
    plt.plot(
        sim_ts["t_ts"],
        jnp.repeat(pee_des[0], repeats=len(sim_ts["t_ts"]), axis=0),
        linestyle="--",
        color=colors[0],
        label=r"$p_\mathrm{ee,x}^\mathrm{d}$",
    )
    plt.plot(
        sim_ts["t_ts"],
        jnp.repeat(pee_des[1], repeats=len(sim_ts["t_ts"]), axis=0),
        linestyle="--",
        color=colors[1],
        label=r"$p_\mathrm{ee,y}^\mathrm{d}$",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(sim_ts["t_ts"], sim_ts["phi_ts"][:, 0], label="$\\phi_1^\mathrm{d}$ [rad]")
    plt.plot(sim_ts["t_ts"], sim_ts["phi_ts"][:, 1], label="$\\phi_2^\mathrm{d}$ [rad]")
    plt.xlabel("Time [s]")
    plt.ylabel("Twist angles [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # render video
    animate_robot(
        forward_kinematics_virtual_backbone_fn,
        sys_helpers["forward_kinematics_rod_fn"],
        sys_helpers["forward_kinematics_platform_fn"],
        params,
        video_path,
        video_ts=sim_ts["t_ts"],
        q_ts=sim_ts["x_ts"][:, : (sim_ts["x_ts"].shape[1] // 2)],
    )
