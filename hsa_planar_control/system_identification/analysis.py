from jax import Array, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict


def analyze_neutral_rod_length_model(
    params: Dict[str, Array],
    data_ts: Dict[str, Array],
):
    def compute_axial_neutral_strain(phi: Array) -> Array:
        varepsilon = phi * params["h"] / params["l"] * params["C_varepsilon"]
        sigma_a_rest = params["sigma_a_eq"] + varepsilon
        return sigma_a_rest

    sigma_a_neutral_per_rod_ts = vmap(compute_axial_neutral_strain)(data_ts["phi_ts"])
    # average over the segment and rod dimension
    sigma_a_neutral_ts = jnp.mean(sigma_a_neutral_per_rod_ts, axis=(1, 2))

    # warning: this hack is only valid for the pure elongation case (i.e. no bending)
    plt.figure(num="neutral strain model")
    plt.plot(data_ts["t_ts"], sigma_a_neutral_ts, label="neutral strain model")
    plt.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 2],
        label="actual axial strain (from data)",
    )
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("axial strain [-]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
