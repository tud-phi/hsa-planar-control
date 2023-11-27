from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
from pathlib import Path

from hsa_planar_control.collocated_form import (
    mapping_into_collocated_form_factory,
)
from hsa_planar_control.controllers.generalized_torques_to_actuation import linearize_actuation

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

num_segments = 1
num_rods_per_segment = 2
# set parameters
params = PARAMS_EPU_CONTROL.copy()

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

(
    forward_kinematics_virtual_backbone_fn,
    forward_kinematics_end_effector_fn,
    jacobian_end_effector_fn,
    inverse_kinematics_end_effector_fn,
    dynamical_matrices_fn,
    sys_helpers,
) = planar_hsa.factory(sym_exp_filepath)

# the current configuration of the robot
q = jnp.array([8.0, 0.02, 0.4])
q_d = jnp.array([0.0, 0.0, 0.0])
# steady state actuation
phi_ss = jnp.array([0.0, 0.0])


def alpha_fn(phi: Array) -> Array:
    B, C, G, K, D, alpha = dynamical_matrices_fn(params, q, q_d, phi)
    return alpha


def linearized_actuation_force_fn(phi: Array) -> Array:
    tau_eq, A = linearize_actuation(
        partial(dynamical_matrices_fn, params), q, phi_ss
    )
    tau_q = tau_eq + A @ phi
    return tau_q


alpha_fn_vmapped = vmap(alpha_fn)
linearized_actuation_force_fn_vmapped = vmap(linearized_actuation_force_fn)

figsize = (5, 2.0)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
x_samples = jnp.linspace(0.0, jnp.pi, 100)
phi1_samples = jnp.stack([x_samples, jnp.zeros_like(x_samples)], axis=1)
phi2_samples = jnp.stack([jnp.zeros_like(x_samples), x_samples], axis=1)

if __name__ == "__main__":
    # plot nonlinear alpha
    fig, axes = plt.subplots(1, 3, figsize=figsize, num="nonlinear_alpha")
    ylabels = [
        r"Torque on bend. strain: $\tau_\mathrm{be}$",
        r"Torque on shear strain: $\tau_\mathrm{sh}$",
        r"Torque on axial strain: $\tau_\mathrm{ax}$"
    ]
    alpha_phi1_samples = alpha_fn_vmapped(phi1_samples)
    alpha_phi2_samples = alpha_fn_vmapped(phi2_samples)
    for i, ax in enumerate(axes):
        ax.plot(x_samples, alpha_phi1_samples[:, i], color=colors[0], label=rf"$\phi_1$")
        ax.plot(x_samples, alpha_phi2_samples[:, i], color=colors[1], label=rf"$\phi_2$")
        ax.set_xlabel(r"Control input $\phi$")
        ax.set_ylabel(ylabels[i])
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # plot linearized alpha
    fig, axes = plt.subplots(1, 3, figsize=figsize, num="linearized_alpha")
    ylabels = [
        r"Torque on bend. strain: $\tau_\mathrm{be}$",
        r"Torque on shear strain: $\tau_\mathrm{sh}$",
        r"Torque on axial strain: $\tau_\mathrm{ax}$"
    ]
    linearized_alpha_phi1_samples = linearized_actuation_force_fn_vmapped(phi1_samples)
    linearized_alpha_phi2_samples = linearized_actuation_force_fn_vmapped(phi2_samples)
    for i, ax in enumerate(axes):
        ax.plot(x_samples, linearized_alpha_phi1_samples[:, i], color=colors[0], label=rf"$\phi_1$")
        ax.plot(x_samples, linearized_alpha_phi2_samples[:, i], color=colors[1], label=rf"$\phi_2$")
        ax.set_xlabel(r"Control input $\phi$")
        ax.set_ylabel(ylabels[i])
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # plot in actuation coordinates
    fig, axes = plt.subplots(1, 3, figsize=figsize, num="actuation_coordinates")
    ylabels = [
        r"Torque on 1st act. coord.: $\tau_{\varphi_1}$",
        r"Torque on 2nd act. coord.: $\tau_{\varphi_2}$",
        r"Torque on unact. coord.: $\tau_{\varphi_3}$"
    ]
    y_samples = x_samples
    for i, ax in enumerate(axes):
        if i < 2:
            ax.plot(x_samples, y_samples, color=colors[i], label=rf"$u_{i+1}$")
        ax.set_xlabel(r"Control input $u$")
        ax.set_ylabel(ylabels[i])
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        if i < 2:
            ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
