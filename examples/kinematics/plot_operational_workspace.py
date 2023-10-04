import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax.numpy as jnp
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
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

    # fit polynomial to the max actuation samples
    xmin, xmax = jnp.min(max_actuation_samples["chiee_ss"][:, 0]), jnp.max(max_actuation_samples["chiee_ss"][:, 0])
    pmax = jnp.polyfit(max_actuation_samples["chiee_ss"][:, 0], max_actuation_samples["chiee_ss"][:, 1], 20)
    pmin = jnp.polyfit(min_actuation_samples["chiee_ss"][:, 0], min_actuation_samples["chiee_ss"][:, 1], 20)
    xp = jnp.linspace(xmin, xmax, 50)
    # plt.plot(
    #     xp,
    #     jnp.polyval(pmax, xp),
    #     label="Polyfit for max. actuation"
    # )
    # plt.plot(
    #     xp,
    #     jnp.polyval(pmin, xp),
    #     label="Polyfit for min. actuation"
    # )

    # tricontour plot of the actuation effort needed within the operational workspace
    # for masking of the concave area, we took inspiration from
    # https://stackoverflow.com/questions/42426095/matplotlib-contour-contourf-of-concave-non-gridded-data
    tricontour_x = random_samples["chiee_ss"][:, 0]
    tricontour_y = random_samples["chiee_ss"][:, 1]
    # effort (i.e. the color) is the mean actuation magnitude
    tricontour_z = jnp.mean(random_samples["phi_ss"], axis=-1)
    triang = tri.Triangulation(tricontour_x, tricontour_y)
    x2 = tricontour_x[triang.triangles].mean(axis=1)
    y2 = tricontour_y[triang.triangles].mean(axis=1)
    # note the very obscure mean command, which, if not present causes an error.
    # now we need some masking condition.
    # this is easy in this case where we generated the data according to the same condition
    condmax = y2 <= jnp.polyval(pmax, x2)
    condmin = y2 >= jnp.polyval(pmin, x2)
    mask = jnp.where(condmin & condmax, 0, 1)
    # apply masking
    triang.set_mask(mask)
    #  plot the contour
    plt.tricontour(
        triang,
        tricontour_z,
        1000,
        cmap="Oranges",
    )
    # plot the masked triangles
    # plt.plot(
    #     x2[mask == 1],
    #     y2[mask == 1],
    #     linestyle="None",
    #     marker=".",
    #     label="Masked triangles"
    # )

    plt.plot(
        max_actuation_samples["chiee_ss"][:, 0],
        max_actuation_samples["chiee_ss"][:, 1],
        "k--",
        label="Max. actuation"
    )
    plt.plot(
        min_actuation_samples["chiee_ss"][:, 0],
        min_actuation_samples["chiee_ss"][:, 1],
        "k--",
        label="Min. actuation"
    )

    plt.xlabel(r"$p_{\mathrm{ee},x}$ [m]")
    plt.ylabel(r"$p_{\mathrm{ee},y}$ [m]")
    plt.axis("equal")
    plt.colorbar(label=r"Mean steady-state actuation $\frac{\phi_1^\mathrm{ss}+\phi_2^\mathrm{ss}}{2}$ [rad]")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
