from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from functools import partial
from jax import Array, jit, random, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Callable, Dict, Tuple

from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
    statically_invert_actuation_to_task_space_scipy_rootfinding,
    statically_invert_actuation_to_task_space_projected_descent,
)
from hsa_planar_control.planning.task_space_trajectory_generation import (
    generate_task_space_trajectory_from_image_contour,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
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

IMAGE_TYPE = "star"

if IMAGE_TYPE == "star":
    pee_centroid = jnp.array([0.0, 0.127])
    max_radius = jnp.array(0.013)
elif IMAGE_TYPE == "tud-flame":
    pee_centroid = jnp.array([0.0, 0.1285])
    max_radius = jnp.array(0.013)
elif IMAGE_TYPE == "mit-csail":
    pee_centroid = jnp.array([0.0, 0.127])
    max_radius = jnp.array(0.015)
elif IMAGE_TYPE == "manta-ray":
    pee_centroid = jnp.array([0.0, 0.129])
    max_radius = jnp.array(0.027)
elif IMAGE_TYPE == "bat":
    pee_centroid = jnp.array([0.0, 0.1285])
    max_radius = jnp.array(0.030)
else:
    raise ValueError(f"Unknown image type: {IMAGE_TYPE}")


def main():
    pee_des_sps = generate_task_space_trajectory_from_image_contour(
        image_type=IMAGE_TYPE,
        pee_centroid=pee_centroid,
        max_radius=max_radius,
        verbose=True,
        show_images=True,
    )
    num_setpoints = pee_des_sps.shape[0]
    print("The trajectory has", num_setpoints, "setpoints.")

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
    planning_fn = jit(
        partial(
            statically_invert_actuation_to_task_space_projected_descent,
            params=params,
            residual_fn=residual_fn,
            inverse_kinematics_end_effector_fn=inverse_kinematics_end_effector_fn,
            maxiter=250,
            verbose=True,
        )
    )

    # set the first initial conditions
    q0 = jnp.zeros((3, ))
    phi0 = jnp.zeros((2, ))

    q_des_sps = jnp.zeros((num_setpoints, q0.shape[0]))  # desired configurations
    phi_ss_sps = jnp.zeros((num_setpoints, phi0.shape[0]))  # steady-state control inputs
    optimality_error_sps = jnp.zeros(num_setpoints)  # optimality errors
    for setpoint_idx in range(num_setpoints):
        # Start timer
        start_time = time.time()

        pee_des = pee_des_sps[setpoint_idx]
        chiee_des, q_des, phi_ss, optimality_error = planning_fn(pee_des=pee_des, q0=q0, phi0=phi0)

        # End timer
        end_time = time.time()
        print("Elapsed time for planning: ", end_time - start_time, "seconds")

        pee_des_sps = pee_des_sps.at[setpoint_idx].set(pee_des)
        q_des_sps = q_des_sps.at[setpoint_idx].set(q_des)
        phi_ss_sps = phi_ss_sps.at[setpoint_idx].set(phi_ss)
        optimality_error_sps = optimality_error_sps.at[setpoint_idx].set(
            optimality_error
        )

        # update initial conditions
        q0 = q_des
        phi0 = phi_ss

    print("pee_des_sps:\n", pee_des_sps)
    print("q_des_sps:\n", q_des_sps)
    print("phi_ss_sps:\n", phi_ss_sps)

    plt.figure(num="Desired end-effector positions")
    ax = plt.gca()
    plt.plot(pee_des_sps[:, 0], pee_des_sps[:, 1], "k--")
    plt.axis("equal")
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.xlabel("$x$ [m]")
    plt.ylabel("$y$ [m]")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
    plt.figure(num="Desired end-effector orientation")
    plt.plot(jnp.arange(num_setpoints), pee_des_sps[:, 0], "--", label="theta")
    plt.xlabel("Setpoint index")
    plt.ylabel(r"$\theta$ [rad]")
    plt.grid(True)
    plt.box(True)
    plt.show()
    plt.figure(num="Steady-state control inputs")
    plt.plot(jnp.arange(num_setpoints), phi_ss_sps[:, 0], label=r"$\phi_{ss,1}$")
    plt.plot(jnp.arange(num_setpoints), phi_ss_sps[:, 1], label=r"$\phi_{ss,2}$")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.xlabel("Setpoint index")
    plt.ylabel(r"$\phi_{ss}$ [rad]")
    plt.show()
    plt.figure(num="Optimality error")
    plt.plot(jnp.arange(num_setpoints), optimality_error_sps)
    plt.grid(True)
    plt.box(True)
    plt.xlabel("Setpoint index")
    plt.ylabel(r"Optimality error")
    plt.show()


if __name__ == "__main__":
    main()
