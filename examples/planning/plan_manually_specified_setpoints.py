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
from hsa_planar_control.planning.steady_state_rollout_planning import plan_with_rollout_to_steady_state

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

PLANNER_TYPE = "steady_state_rollout"  # "static_inversion", "steady_state_rollout"

# set parameters
params = PARAMS_FPU_CONTROL.copy()

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
    if PLANNER_TYPE == "static_inversion":
        residual_fn = jit(
            static_inversion_factory(
                params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
            )
        )
        planning_fn = partial(
            statically_invert_actuation_to_task_space_scipy_rootfinding,
            params=params,
            residual_fn=residual_fn,
            inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
            verbose=True,
        )
    elif PLANNER_TYPE == "steady_state_rollout":
        planning_fn = partial(
            plan_with_rollout_to_steady_state,
            params=params,
            forward_kinematics_end_effector_fn=forward_kinematics_end_effector_fn,
            dynamical_matrices_fn=dynamical_matrices_fn,
            q0=q0,
            phi0=phi0,
            solver="optimistix_levenberg_marquardt",
        )
    else:
        raise ValueError(f"Unknown PLANNER_TYPE: {PLANNER_TYPE}")

    # desired end-effector positions
    pee_des_sps = jnp.array(
        [
            [0.0, 0.120],
            [+0.00479247, 0.12781018],
            [-0.035, 0.122],
            [-0.00782133, 0.13024847],
            [0.00823294, 0.114643],
            [-0.01417039, 0.12388105],
            [0.0, 0.140],
            [0.02524261, 0.1304036],
            [-0.0059703, 0.13986947],
            [0.0073023, 0.11479653],
            [0.00567301, 0.1271345],
        ]
    )
    num_setpoints = pee_des_sps.shape[0]

    chiee_des_sps = jnp.zeros((num_setpoints, 3))  # poses
    q_des_sps = jnp.zeros((num_setpoints, 3))  # desired configurations
    phi_ss_sps = jnp.zeros((num_setpoints, 2))  # steady-state control inputs
    for setpoint_idx in range(num_setpoints):
        pee_des = pee_des_sps[setpoint_idx]
        chiee_des, q_des, phi_ss, optimality_error = planning_fn(
            pee_des=pee_des
        )

        chiee_des_sps = chiee_des_sps.at[setpoint_idx].set(chiee_des)
        q_des_sps = q_des_sps.at[setpoint_idx].set(q_des)
        phi_ss_sps = phi_ss_sps.at[setpoint_idx].set(phi_ss)

    print("chiee_des_sps:\n", chiee_des_sps)
    print("q_des_sps:\n", q_des_sps)
    print("phi_ss_sps:\n", phi_ss_sps)
