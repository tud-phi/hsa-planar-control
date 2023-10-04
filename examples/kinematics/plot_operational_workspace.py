import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
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


num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

HSA_MATERIAL = "fpu"
SHOW_ROBOT_CONFIGS = True

# set parameters
if HSA_MATERIAL == "fpu":
    params = PARAMS_FPU_CONTROL.copy()
elif HSA_MATERIAL == "epu":
    params = PARAMS_EPU_CONTROL.copy()
else:
    raise ValueError(f"Unknown hsa_material: {HSA_MATERIAL}")


if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    batched_forward_kinematics_virtual_backbone_fn = vmap(
        forward_kinematics_virtual_backbone_fn, in_axes=(None, None, 0), out_axes=-1
    )
    batched_forward_kinematics_rod_fn = vmap(
        sys_helpers["forward_kinematics_rod_fn"],
        in_axes=(None, None, 0, None),
        out_axes=-1,
    )
    batched_forward_kinematics_platform_fn = vmap(
        sys_helpers["forward_kinematics_platform_fn"],
        in_axes=(None, None, 0),
        out_axes=0,
    )

    with open(
        str(
            Path(__file__).parent.parent.parent
            / "data"
            / "kinematics"
            / f"operational_workspace_{HSA_MATERIAL}.dill"
        ),
        "rb",
    ) as f:
        data = dill.load(f)

    max_actuation_samples = data["max_actuation"]
    min_actuation_samples = data["min_actuation"]
    random_samples = data["random"]

    fig = plt.figure(figsize=(4.0, 3.0), num=f"Operational workspace of {HSA_MATERIAL} HSA")
    ax = fig.add_subplot(111)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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
        zorder=1,
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
        zorder=2
        # label="Max. actuation"
    )
    plt.plot(
        min_actuation_samples["chiee_ss"][:, 0],
        min_actuation_samples["chiee_ss"][:, 1],
        "k--",
        zorder=2
        # label="Min. actuation"
    )

    if SHOW_ROBOT_CONFIGS:
        def visualize_robot_configuration(idx: int, _q: Array, _ax: plt.Axes, _color: str = "CN"):
            print(f"Visualizing configuration q = {_q}")

            s_ps = jnp.linspace(0, jnp.sum(params["l"]), 100)
            lw = 8.0
            alpha = 0.25

            # derive the curve of the rods
            chiL_ps = batched_forward_kinematics_rod_fn(params, _q, s_ps, 0)  # poses of left rod
            chiR_ps = batched_forward_kinematics_rod_fn(params, _q, s_ps, 1)  # poses of right rod
            # add the first point of the proximal cap and the last point of the distal cap
            chiL_ps = jnp.concatenate(
                [
                    (chiL_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])).reshape(3, 1),
                    chiL_ps,
                    (
                            chiL_ps[:, -1]
                            + jnp.array(
                        [
                            -jnp.sin(chiL_ps[2, -1]) * params["ldc"][-1],
                            jnp.cos(chiL_ps[2, -1]) * params["ldc"][-1],
                            chiL_ps[2, -1],
                        ]
                    )
                    ).reshape(3, 1),
                ],
                axis=1,
            )
            # add the first point of the proximal cap and the last point of the distal cap
            chiR_ps = jnp.concatenate(
                [
                    (chiR_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])).reshape(3, 1),
                    chiR_ps,
                    (
                            chiR_ps[:, -1]
                            + jnp.array(
                        [
                            -jnp.sin(chiR_ps[2, -1]) * params["ldc"][-1],
                            jnp.cos(chiR_ps[2, -1]) * params["ldc"][-1],
                            chiR_ps[2, -1],
                        ]
                    )
                    ).reshape(3, 1),
                ],
                axis=1,
            )
            # plot both rods
            _ax.plot(
                chiL_ps[0, :],
                chiL_ps[1, :],
                linewidth=lw,
                color=_color,
                alpha=alpha,
                label=f"q = {_q}",
                zorder=(idx + 1) * 10
            )
            _ax.plot(
                chiR_ps[0, :],
                chiR_ps[1, :],
                linewidth=lw,
                color=_color,
                alpha=alpha,
                zorder=(idx + 1) * 10
            )

            # draw the platform
            # poses of the platforms
            chip_ps = batched_forward_kinematics_platform_fn(
                params, _q, jnp.arange(0, num_segments)
            )
            for i in range(chip_ps.shape[0]):
                # iterate over the platforms
                platform_R = jnp.array(
                    [
                        [jnp.cos(chip_ps[i, 2]), -jnp.sin(chip_ps[i, 2])],
                        [jnp.sin(chip_ps[i, 2]), jnp.cos(chip_ps[i, 2])],
                    ]
                )  # rotation matrix for the platform
                platform_llc = chip_ps[i, :2] + platform_R @ jnp.array(
                    [
                        -params["pcudim"][i, 0] / 2,  # go half the width to the left
                        -params["pcudim"][i, 1] / 2,  # go half the height down
                    ]
                )  # lower left corner of the platform
                platform_ulc = chip_ps[i, :2] + platform_R @ jnp.array(
                    [
                        -params["pcudim"][i, 0] / 2,  # go half the width to the left
                        +params["pcudim"][i, 1] / 2,  # go half the height down
                    ]
                )  # upper left corner of the platform
                platform_urc = chip_ps[i, :2] + platform_R @ jnp.array(
                    [
                        +params["pcudim"][i, 0] / 2,  # go half the width to the left
                        +params["pcudim"][i, 1] / 2,  # go half the height down
                    ]
                )  # upper right corner of the platform
                platform_lrc = chip_ps[i, :2] + platform_R @ jnp.array(
                    [
                        +params["pcudim"][i, 0] / 2,  # go half the width to the left
                        -params["pcudim"][i, 1] / 2,  # go half the height down
                    ]
                )  # lower right corner of the platform
                platform_curve = jnp.stack(
                    [platform_llc, platform_ulc, platform_urc, platform_lrc, platform_llc],
                    axis=1,
                )
                _ax.fill(
                    platform_curve[0],
                    platform_curve[1],
                    color=_color,
                    alpha=alpha,
                    zorder=(idx + 1) * 10 + 1
                )

            # draw the end-effector
            chiee = forward_kinematics_end_effector_fn(params, _q)
            _ax.plot(
                chiee[0],
                chiee[1],
                marker="o",
                color=_color,
                alpha=min(alpha + 0.2, 1.0),
                zorder=(idx + 1) * 10 + 2
            )

        # neutral configuration
        visualize_robot_configuration(
            0,
            min_actuation_samples["q_ss"][min_actuation_samples["q_ss"].shape[0] // 2],
            ax, colors[0]
        )
        # maximum bending left
        visualize_robot_configuration(1, max_actuation_samples["q_ss"][0], ax, colors[2])
        # maximum bending right
        visualize_robot_configuration(2, max_actuation_samples["q_ss"][-1], ax, colors[3])

    plt.xlabel(r"$p_{\mathrm{ee},x}$ [m]")
    plt.ylabel(r"$p_{\mathrm{ee},y}$ [m]")
    plt.axis("equal")
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.colorbar(label=r"Mean steady-state actuation $\frac{\phi_1^\mathrm{ss}+\phi_2^\mathrm{ss}}{2}$ [rad]")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
