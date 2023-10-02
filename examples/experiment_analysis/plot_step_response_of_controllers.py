import dill
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

START_TIME = 59.0
END_TIME = 67.0
experiments = {
    "20230925_092200": {"linestyle": "dotted", "label": "PID"},
    "20230925_093236": {"linestyle": "dashed", "label": "P-satI-D"},
    "20230925_094023": {"linestyle": "dashdot", "label": "P-satI-D+GC"},
}


experiment_folder = Path("data") / "experiments"


if __name__ == "__main__":
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        with open(
            str(experiment_folder / experiment_id / ("rosbag2_" + experiment_id + "_0.dill")), "rb"
        ) as f:
            data_ts = dill.load(f)
        ci_ts = data_ts["controller_info_ts"]

        # trim the time series data
        ts = ci_ts["ts"] - ci_ts["ts"][0]
        time_selector = ts >= START_TIME
        if END_TIME is not None:
            time_selector = time_selector & (ts <= END_TIME)
        ci_ts["ts"] = ts[time_selector]
        for key in ci_ts.keys():
            if key != "ts":
                ci_ts[key] = ci_ts[key][time_selector]

        experiments[experiment_id]["data_ts"] = data_ts
        experiments[experiment_id]["ci_ts"] = ci_ts

    figsize = (4.5, 3)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lw_ref = 1.9  # linewidth for reference trajectory
    lw = 2.1  # linewidth for the actual trajectory
    dashes = (1.2, 0.8)

    plt.figure(figsize=figsize, num="Step response: End-effector position")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        ci_ts = experiments[experiment_id]["ci_ts"]
        # plot the reference trajectory
        if experiment_idx == 0:
            ax1.plot(
                ci_ts["ts"],
                ci_ts["chiee_des"][:, 0]*1e3,
                color=colors[0],
                linestyle="solid",
                linewidth=lw_ref,
                # dashes=dashes,
                label=r"$p_\mathrm{x}^\mathrm{d}$",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["chiee_des"][:, 1]*1e3,
                color=colors[1],
                linestyle="solid",
                linewidth=lw_ref,
                # dashes=dashes,
                label=r"$p_\mathrm{y}^\mathrm{d}$",
            )
        ax1.plot(
            ci_ts["ts"],
            ci_ts["chiee"][:, 0]*1e3,
            color=colors[0],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            # label=r"$p_\mathrm{x}$",
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["chiee"][:, 1]*1e3,
            color=colors[1],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            # label=r"$p_\mathrm{y}$",
        )

    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"x-position $p_{\mathrm{ee},x}$ [mm]")
    ax2.set_ylabel(r"y-position $p_{\mathrm{ee},y}$ [mm]")
    ax1.legend(loc="lower center")
    ax2.legend(loc="lower right")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=figsize, num="Step response: Configuration")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        ci_ts = experiments[experiment_id]["ci_ts"]
        if experiment_idx == 0:
            # plot the reference trajectory
            ax1.plot(
                ci_ts["ts"],
                ci_ts["q_des"][:, 0],
                color=colors[0],
                linestyle="solid",
                linewidth=lw_ref,
                # dashes=dashes,
                label=r"$\kappa_\mathrm{b}^\mathrm{d}$",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["q_des"][:, 1],
                color=colors[1],
                linestyle="solid",
                linewidth=lw_ref,
                # dashes=dashes,
                label=r"$\sigma_\mathrm{sh}^\mathrm{d}$",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["q_des"][:, 2],
                color=colors[2],
                linestyle="solid",
                linewidth=lw_ref,
                # dashes=dashes,
                label=r"$\sigma_\mathrm{ax}^\mathrm{d}$",
            )
        ax1.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 0],
            color=colors[0],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            # label=r"$\kappa_\mathrm{b}$",
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 1],
            color=colors[1],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            # label=r"$\sigma_\mathrm{sh}$",
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 2],
            color=colors[2],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            # label=r"$\sigma_\mathrm{ax}$",
        )

    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{b}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
