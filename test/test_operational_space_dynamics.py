from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, debug, jit, random, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path


num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / "planar_hsa_ns-1_nrs-2.dill"
)


def test_operational_space_dynamics():
    params = PARAMS_FPU_CONTROL.copy()
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    dynamical_matrices_fn = partial(dynamical_matrices_fn, params, eps=1e-1)
    operational_space_dynamical_matrices_fn = partial(
        sys_helpers["operational_space_dynamical_matrices_fn"], params, eps=1e-1
    )

    def compute_operational_space_dynamical_components(
        q: Array, q_d: Array, phi: Array
    ):
        B, C, G, K, D, alpha = dynamical_matrices_fn(q, q_d, phi=phi)
        Lambda, mu, Jee, Jee_d, JeeB_pinv = operational_space_dynamical_matrices_fn(
            q, q_d, B, C
        )

        print("Jee:\n", Jee)
        print("Jee_d:\n", Jee_d)
        print("JeeB_pinv:\n", JeeB_pinv)

        print("Lambda:\n", Lambda)

        print("C:\n", C)

        print("mu:\n", mu[:2, :])

        # coupling of residual null-space dynamics on the operational space dynamics
        mu_N = mu[:2, :] @ (jnp.eye(q.shape[0]) - JeeB_pinv[:, :2] @ Jee[:2, :])

        print("mu_N:\n", mu_N)

        return mu_N

    # test for different axial strains
    print("test for different bending strains")
    num_points = 20001
    sigma_b_ps = jnp.linspace(-0.5, 0.5, num_points)
    q_ps = jnp.stack(
        [sigma_b_ps, jnp.zeros((num_points,)), jnp.zeros((num_points,))], axis=1
    )
    q_d = jnp.array([0.0, 0.0, 0.1])
    phi = jnp.zeros((2,))
    mu_N_ps = vmap(
        compute_operational_space_dynamical_components, in_axes=(0, None, None)
    )(q_ps, q_d, phi)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sigma_b_ps, mu_N_ps[:, 0, 0], label=r"$\mu_{N,11}$")
    ax.plot(sigma_b_ps, mu_N_ps[:, 0, 1], label=r"$\mu_{N,12}$")
    ax.plot(sigma_b_ps, mu_N_ps[:, 0, 2], label=r"$\mu_{N,13}$")
    ax.plot(sigma_b_ps, mu_N_ps[:, 1, 0], label=r"$\mu_{N,21}$")
    ax.plot(sigma_b_ps, mu_N_ps[:, 1, 1], label=r"$\mu_{N,22}$")
    ax.plot(sigma_b_ps, mu_N_ps[:, 1, 2], label=r"$\mu_{N,23}$")
    ax.set_xlabel(r"$\sigma_b$")
    ax.set_ylabel(r"$\mu_N$")
    ax.legend()
    plt.tight_layout()
    plt.box(True)
    plt.grid(True)
    plt.show()

    mu_Nb = compute_operational_space_dynamical_components(q_ps[0], q_d, phi)
    mu_N0 = compute_operational_space_dynamical_components(jnp.zeros((3,)), q_d, phi)


if __name__ == "__main__":
    test_operational_space_dynamics()
