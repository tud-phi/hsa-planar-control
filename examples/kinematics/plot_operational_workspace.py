import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax.numpy as jnp
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

hsa_material = "fpu"

if __name__ == "__main__":
    with open(
        str(
            Path(__file__).parent.parent.parent
            / "data"
            / "kinematics"
            / f"operational_workspace_{hsa_material}.dill"
        ),
        "rb",
    ) as f:
        data = dill.load(f)

    max_actuation_samples = data["max_actuation"]
    min_actuation_samples = data["min_actuation"]
    random_samples = data["random"]

    fig = plt.figure(figsize=(4.5, 4.5), num=f"Operational workspace of {hsa_material} HSA")

    tricontour_x = random_samples["chiee_ss"][:, 0] * 1e3
    tricontour_y = random_samples["chiee_ss"][:, 1] * 1e3
    # effort (i.e. the color) is the mean actuation magnitude
    tricontour_z = jnp.mean(random_samples["phi_ss"], axis=-1)
    plt.tricontour(
        tricontour_x,
        tricontour_y,
        tricontour_z,
        1000,
        cmap="viridis",
    )
    plt.plot(
        tricontour_x,
        tricontour_y,
        linestyle="None",
        marker=".",
    )

    plt.plot(
        max_actuation_samples["chiee_ss"][:, 0] * 1e3,
        max_actuation_samples["chiee_ss"][:, 1] * 1e3,
        "k--",
        label="Max. actuation"
    )
    plt.plot(
        min_actuation_samples["chiee_ss"][:, 0] * 1e3,
        min_actuation_samples["chiee_ss"][:, 1] * 1e3,
        "k--",
        label="Min. actuation"
    )

    plt.xlabel(r"$p_{\mathrm{ee},x}$ [mm]")
    plt.ylabel(r"$p_{\mathrm{ee},y}$ [mm]")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
