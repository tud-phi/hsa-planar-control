import dill
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
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

EXPERIMENT_NAME = "20230925_113825"  # experiment name
TRAJECTORY_TYPE = "tud-star"  # trajectory type
PLOT_STEADY_STATE_ACTUATION = True

DURATION = None
if TRAJECTORY_TYPE == "manual":
    DURATION = 110.0  # duration of the experiment [s]


def main():
    experiment_folder = Path("data") / "experiments" / EXPERIMENT_NAME
    with open(
        str(experiment_folder / ("rosbag2_" + EXPERIMENT_NAME + "_0.dill")), "rb"
    ) as f:
        data_ts = dill.load(f)
    ci_ts = data_ts["controller_info_ts"]

    print("Available time series data:\n", data_ts.keys())
    print("Available controller info", ci_ts.keys())

    # absolute start time
    start_time = ci_ts["ts"][0]
    ts = ci_ts["ts"] - start_time
    print("Experiment duration:", ts[-1])

    # trim the time series data
    if DURATION is not None:
        end_time_idx = np.argmax(ts > DURATION)
        ts = ts[:end_time_idx]
        for key in ci_ts.keys():
            ci_ts[key] = ci_ts[key][:end_time_idx]

    figsize = (4.5, 3)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth_dotted = 2.25
    dashes = (1.2, 0.8)

    plt.figure(figsize=(4.5, 3), num="End-effector position")
    ax = plt.gca()
    ax.plot(
        ts,
        ci_ts["chiee_des"][:, 0],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$p_\mathrm{x}^\mathrm{d}$",
    )
    ax.plot(
        ts,
        ci_ts["chiee_des"][:, 1],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$p_\mathrm{y}^\mathrm{d}$",
    )
    ax.plot(
        ts,
        ci_ts["chiee"][:, 0],
        color=colors[0],
        label=r"$p_\mathrm{x}$",
    )
    ax.plot(
        ts,
        ci_ts["chiee"][:, 1],
        color=colors[1],
        label=r"$p_\mathrm{y}$",
    )
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"End-effector position $p_\mathrm{ee}$ [m]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_pee.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_pee.eps"))
    plt.show()

    plt.figure(figsize=figsize, num="Control input")
    ax = plt.gca()
    if PLOT_STEADY_STATE_ACTUATION:
        ax.plot(
            ts,
            ci_ts["phi_ss"][:, 0],
            color=colors[0],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\phi^\mathrm{ss}_1$",
        )
        ax.plot(
            ts,
            ci_ts["phi_ss"][:, 1],
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\phi^\mathrm{ss}_2$",
        )
    ax.plot(
        ts,
        ci_ts["phi_des_sat"][:, 0],
        color=colors[0],
        label=r"$\phi_1$",
    )
    ax.plot(
        ts,
        ci_ts["phi_des_sat"][:, 1],
        color=colors[1],
        label=r"$\phi_2$",
    )
    plt.xlabel(r"$t$ [s]")
    plt.ylabel(r"Control input $\phi$ [rad]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_phi.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_phi.eps"))
    plt.show()

    fig = plt.figure(figsize=figsize, num="Configuration")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(
        ts,
        ci_ts["q_des"][:, 0],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\kappa_\mathrm{b}^\mathrm{d}$",
    )
    ax1.plot(
        ts,
        ci_ts["q"][:, 0],
        color=colors[0],
        label=r"$\kappa_\mathrm{b}$",
    )
    ax2.plot(
        ts,
        ci_ts["q_des"][:, 1],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\sigma_\mathrm{sh}^\mathrm{d}$",
    )
    ax2.plot(
        ts,
        ci_ts["q_des"][:, 2],
        color=colors[2],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\sigma_\mathrm{ax}^\mathrm{d}$",
    )
    ax2.plot(
        ts,
        ci_ts["q"][:, 1],
        color=colors[1],
        label=r"$\sigma_\mathrm{sh}$",
    )
    ax2.plot(
        ts,
        ci_ts["q"][:, 2],
        color=colors[2],
        label=r"$\sigma_\mathrm{ax}$",
    )
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{b}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_q.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_q.eps"))
    plt.show()

    if "varphi" in ci_ts.keys() and ci_ts["varphi"].size > 0:
        plt.figure(figsize=figsize, num="Actuation coordinates")
        ax = plt.gca()
        ax.plot(
            ts,
            ci_ts["varphi_des"][:, 0],
            color=colors[0],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\varphi^\mathrm{d}_1$",
        )
        ax.plot(
            ts,
            ci_ts["varphi_des"][:, 1],
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\varphi^\mathrm{d}_2$",
        )
        ax.plot(
            ts,
            ci_ts["varphi"][:, 0],
            color=colors[0],
            label=r"$\varphi_1$",
        )
        ax.plot(
            ts,
            ci_ts["varphi"][:, 1],
            color=colors[1],
            label=r"$\varphi_2$",
        )
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"Actuation coordinates $\varphi_\mathrm{a}$ [rad]")
        plt.legend()
        plt.grid(True)
        plt.box(True)
        plt.tight_layout()
        plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_varphi.pdf"))
        plt.savefig(str(experiment_folder / f"{EXPERIMENT_NAME}_varphi.eps"))
        plt.show()


if __name__ == "__main__":
    main()
