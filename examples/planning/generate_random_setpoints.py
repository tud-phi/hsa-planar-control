""" Generate random setpoints for the planar HSA by simulating the open-loop system until steady-state is reached."""
from diffrax import Dopri5, Euler
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from functools import partial
from jax import Array, jit, random, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Tuple

from hsa_planar_control.controllers.constant_controller import constant_control
from hsa_planar_control.simulation import simulate_closed_loop_system


num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

hsa_material = "epu"
num_setpoints = 10

if hsa_material == "fpu":
    params = PARAMS_FPU_CONTROL.copy()
    params.update(
        {
            "phi_max": 200 / 180 * jnp.pi * jnp.ones_like(params["phi_max"]),
        }
    )
elif hsa_material == "epu":
    params = PARAMS_EPU_CONTROL.copy()
else:
    raise ValueError(f"Unknown hsa_material: {hsa_material}")

# set sampling range
phi_min = 0.2 * jnp.ones_like(params["phi_max"].flatten())
phi_max = params["phi_max"].flatten() - 0.2

# set seed
seed = 0
rng = random.PRNGKey(seed)

# set simulation params
duration = 10.0
sim_dt = 1e-4  # time step for simulation [s]
control_dt = 1e-1  # time step for control [s]
# define initial configuration
q0 = jnp.array([0.0, 0.0, 0.0])
q_d0 = jnp.zeros_like(q0)
phi0 = jnp.array([0.0, 0.0])


if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    pee_ss_sps = jnp.zeros((num_setpoints, 2))  # desired end-effector positions
    q_ss_sps = jnp.zeros((num_setpoints, q0.shape[0]))  # desired configurations
    phi_ss_sps = jnp.zeros((num_setpoints, phi0.shape[0]))  # steady-state control inputs
    for setpoint_idx in range(num_setpoints):
        rng, sample_key = random.split(rng)
        phi_ss = random.uniform(
            sample_key,
            phi0.shape,
            minval=phi_min,
            maxval=phi_max,
        )

        control_fn = partial(
            constant_control,
            phi_des=phi_ss,
        )

        sim_ts = simulate_closed_loop_system(
            dynamical_matrices_fn,
            params,
            q0=q0,
            q_d0=q_d0,
            phi0=phi0,
            sim_dt=sim_dt,
            control_dt=control_dt,
            duration=duration,
            control_fn=control_fn,
            ode_solver_class=Euler,
        )
        print("keys of sim_ts:", sim_ts.keys())
        # extract the last time step
        q_ss = sim_ts["q_ts"][-1]

        # run forward kinematics to get end-effector position
        chiee_ss = forward_kinematics_end_effector_fn(params, q_ss)
        pee_ss = chiee_ss[:2]

        print(
            "chiee_ss",
            chiee_ss,
            "q_ss:",
            q_ss,
            "phi_ss",
            phi_ss / jnp.pi * 180,
            "deg",
        )

        pee_ss_sps = pee_ss_sps.at[setpoint_idx].set(pee_ss)
        q_ss_sps = q_ss_sps.at[setpoint_idx].set(q_ss)
        phi_ss_sps = phi_ss_sps.at[setpoint_idx].set(phi_ss)

    print("pee_des_sps:\n", pee_ss_sps)
    print("q_des_sps:\n", q_ss_sps)
    print("phi_ss_sps:\n", phi_ss_sps)
