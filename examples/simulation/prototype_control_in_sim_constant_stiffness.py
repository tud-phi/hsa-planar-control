from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from diffrax import Dopri5, Euler
from functools import partial
from jax import Array, jit, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict

from hsa_planar_control.controllers.constant_controller import constant_control
from hsa_planar_control.controllers.configuration_space_controllers import (
    pd_plus_feedforward,
    pd_plus_steady_state_actuation,
    P_satI_D_collocated_form_plus_steady_state_actuation_for_constant_stiffness,
)
from hsa_planar_control.controllers.operational_space_controllers import (
    operational_space_computed_torque,
)
from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
    statically_invert_actuation_to_task_space_scipy_rootfinding,
)
from hsa_planar_control.rendering.opencv_renderer import animate_robot
from hsa_planar_control.simulation import simulate_closed_loop_system

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

# set parameters
ones_rod = jnp.ones((num_segments, num_rods_per_segment))
params = PARAMS_FPU_CONTROL.copy()
params.update(
    {
        "Ehat": 1e4 * ones_rod,  # Elastic modulus of each rod [Pa]
        "Ghat": 1e3 * ones_rod,  # Shear modulus of each rod [Pa]
        # Constant to scale the Elastic modulus linearly with the twist strain [Pa/(rad/m)]
        "C_E": 0 * ones_rod,
        # Constant to scale the Shear modulus linearly with the twist strain [Pa/(rad/m)]
        "C_G": 0 * ones_rod,
    }
)
rho_perturb_factor = (
    1.2  # perturb the mass density of the robot used for the simulation
)
params_perturbed = params.copy()
params_perturbed.update(
    {
        "rhor": rho_perturb_factor * params["rhor"],
        "rhop": rho_perturb_factor * params["rhop"],
        "rhoec": rho_perturb_factor * params["rhoec"],
    }
)

# activate all strains (i.e. bending, shear, and axial)
strain_selector = jnp.ones((3 * num_segments,), dtype=bool)
consider_underactuation_model = True

# define initial configuration
q0 = jnp.array([jnp.pi, 0.0, 0.0])
q_d0 = jnp.zeros_like(q0)
phi0 = jnp.array([0.0, 0.0])

# phi_des = jnp.array([jnp.pi / 2, 0])  # motor actuation angles
q_des = jnp.array([-3.06644902, 0.01001036, 0.06821192])  # desired configuration
# pee_des = jnp.array([0.01465215, 0.10121633])  # desired end effector position
# pee_des = jnp.array([0.000, 0.13])
pee_des = jnp.array([-0.05, 0.18])

duration = 10.0
sim_dt = 1e-4  # time step

# control settings
control_dt = 2e-2  # control time step. corresponds to 50 Hz
Kp = 1e1 * jnp.eye(phi0.shape[0])
Ki = 5e0 * jnp.eye(phi0.shape[0])
Kd = 1e-1 * jnp.eye(phi0.shape[0])
controller_state_init = None

# video settings
video_path = Path("videos") / "controlled_planar_hsa_constant_stiffness.mp4"

if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath, strain_selector)

    residual_fn = jit(
        static_inversion_factory(
            params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
        )
    )
    planning_fn = partial(
        statically_invert_actuation_to_task_space_scipy_rootfinding,
        residual_fn=residual_fn,
        inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
    )
    chiee_des, q_des, phi_ss, optimality_error = planning_fn(params, pee_des=pee_des)
    print("Desired configuration: ", q_des, "steady-state actuation: ", phi_ss)

    # control_fn = partial(
    #     constant_control,
    #     phi_des=jnp.array([jnp.pi/4, jnp.pi/4]),
    # )

    # control_fn = partial(
    #     pd_plus_feedforward,
    #     dynamical_matrices_fn=partial(dynamical_matrices_fn, params),
    #     q_des=q_des,
    #     Kp=Kp,
    #     Kd=Kd,
    # )

    # control_fn = partial(
    #     pd_plus_steady_state_actuation,
    #     dynamical_matrices_fn=partial(dynamical_matrices_fn, params),
    #     q_des=q_des,
    #     phi_ss=phi_ss,
    #     Kp=Kp,
    #     Kd=Kd,
    # )

    control_fn = partial(
        P_satI_D_collocated_form_plus_steady_state_actuation_for_constant_stiffness,
        dynamical_matrices_fn=partial(dynamical_matrices_fn, params),
        dt=control_dt,
        q_des=q_des,
        phi_ss=phi_ss,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
    )
    controller_state_init = {
        "integral_error": jnp.zeros_like(phi0),
    }

    # control_fn = partial(
    #     operational_space_computed_torque,
    #     forward_kinematics_end_effector_fn=partial(forward_kinematics_end_effector_fn, params),
    #     jacobian_end_effector_fn=partial(jacobian_end_effector_fn, params),
    #     dynamical_matrices_fn=partial(dynamical_matrices_fn, params),
    #     operational_space_dynamical_matrices_fn=partial(
    #       sys_helpers["operational_space_dynamical_matrices_fn"],
    #       params
    #     ),
    #     pee_des=pee_des,
    #     Kp=1e2 * jnp.diag(jnp.array([1.0, 1.0])),
    #     Kd=1e-1 * jnp.diag(jnp.array([1.0, 1.0])),
    #     consider_underactuation_model=consider_underactuation_model
    # )

    sim_ts = simulate_closed_loop_system(
        dynamical_matrices_fn,
        params_perturbed,
        q0=q0,
        q_d0=q_d0,
        phi0=phi0,
        sim_dt=sim_dt,
        control_dt=control_dt,
        duration=duration,
        control_fn=control_fn,
        controller_state_init=controller_state_init,
        ode_solver_class=Euler,
        consider_underactuation_model=consider_underactuation_model,
    )

    print("Last configuration:", sim_ts["q_ts"][-1])
    print(
        "Last end-effector pose:",
        forward_kinematics_end_effector_fn(params, sim_ts["q_ts"][-1]),
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xi_eq, B_xi = sys_helpers["rest_strains_fn"](params), sys_helpers["B_xi"]
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[0] + sim_ts["q_ts"][:, 0],
        color=colors[0],
        label=r"$\kappa_\mathrm{b}$ [rad / m]",
    )
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[1] + sim_ts["q_ts"][:, 1],
        color=colors[1],
        label=r"$\sigma_\mathrm{sh}$ [-]",
    )
    plt.plot(
        sim_ts["t_ts"],
        xi_eq[2] + sim_ts["q_ts"][:, 2],
        color=colors[2],
        label=r"$\sigma_\mathrm{a}$ [-]",
    )
    if "q_des" in locals():
        plt.plot(
            sim_ts["t_ts"],
            xi_eq[0] + jnp.repeat(q_des[0], repeats=len(sim_ts["t_ts"]), axis=0),
            linestyle="--",
            color=colors[0],
            label=r"$\kappa_\mathrm{b}^\mathrm{d}$ [rad / m]",
        )
        plt.plot(
            sim_ts["t_ts"],
            xi_eq[1] + jnp.repeat(q_des[1], repeats=len(sim_ts["t_ts"]), axis=0),
            linestyle="--",
            color=colors[1],
            label=r"$\sigma_\mathrm{sh}^\mathrm{d}$ [-]",
        )
        plt.plot(
            sim_ts["t_ts"],
            xi_eq[2] + jnp.repeat(q_des[2], repeats=len(sim_ts["t_ts"]), axis=0),
            linestyle="--",
            color=colors[2],
            label=r"$\sigma_\mathrm{a}^\mathrm{d}$ [-]",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Strain of virtual backbone")
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
        x_ts=sim_ts["x_ts"],
    )
