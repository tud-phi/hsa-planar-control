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

from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
    statically_invert_actuation_to_task_space_scipy_rootfinding,
)

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
params.update(
    {
        "phi_max": 210 / 180 * jnp.pi * jnp.ones_like(params["phi_max"]),
    }
)

# set seed
seed = 0
rng = random.PRNGKey(seed)

num_setpoints = 10
# set sampling range
# radial distance and polar angle of the end-effector position
pee_polar_min = jnp.array([0.110, -30 / 180 * jnp.pi])
pee_polar_max = jnp.array([0.202, 30 / 180 * jnp.pi])

# define initial configuration
q0 = jnp.array([0.0, 0.0, 0.0])
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
    residual_fn = jit(
        static_inversion_factory(
            params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
        )
    )
    planning_fn = partial(
        statically_invert_actuation_to_task_space_scipy_rootfinding,
        params,
        residual_fn=residual_fn,
        inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
        q0=q0,
        phi0=phi0,
    )

    pee_des_sps = jnp.zeros((num_setpoints, 2))  # desired end-effector positions
    q_des_sps = jnp.zeros((num_setpoints, q0.shape[0]))  # desired configurations
    phi_ss_sps = jnp.zeros(
        (num_setpoints, phi0.shape[0])
    )  # steady-state control ipnuts
    for setpoint_idx in range(num_setpoints):
        # resample if actuation is outside of constraints
        sample_valid = False
        while sample_valid is False:
            rng, sample_key = random.split(rng)
            pee_polar = random.uniform(
                sample_key,
                pee_polar_min.shape,
                minval=pee_polar_min,
                maxval=pee_polar_max,
            )
            pee_des = jnp.array(
                [
                    pee_polar[0] * jnp.sin(pee_polar[1]),
                    pee_polar[0] * jnp.cos(pee_polar[1]),
                ]
            )

            chiee_des, q_des, phi_ss, optimality_error = planning_fn(pee_des=pee_des)
            print(
                "chiee_des",
                chiee_des,
                "q_des:",
                q_des,
                "phi_ss",
                phi_ss / jnp.pi * 180,
                "deg",
            )

            if (phi_ss * params["h"].flatten() >= 0.0).all() and (
                phi_ss * params["h"].flatten() <= params["phi_max"].flatten()
            ).all():
                print("Sample accepted")
                sample_valid = True
                pee_des_sps = pee_des_sps.at[setpoint_idx].set(pee_des)
                q_des_sps = q_des_sps.at[setpoint_idx].set(q_des)
                phi_ss_sps = phi_ss_sps.at[setpoint_idx].set(phi_ss)

    print("pee_des_sps:\n", pee_des_sps)
    print("q_des_sps:\n", q_des_sps)
    print("phi_ss_sps:\n", phi_ss_sps)
