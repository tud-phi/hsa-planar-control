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
    "20230925_092200": {"linestyle": "solid", "marker": "o", "label": "PID", "time_lag": 0.0},
    "20230925_093236": {"linestyle": "dashed", "marker": "^", "label": "P-satI-D", "time_lag": 0.159},
    "20230925_094023": {"linestyle": "dashdot", "marker": "s", "label": "P-satI-D+GC", "time_lag": 0.098},
}


experiment_folder = Path("data") / "experiments"


if __name__ == "__main__":
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        with open(
            str(
                experiment_folder
                / experiment_id
                / ("rosbag2_" + experiment_id + "_0.dill")
            ),
            "rb",
        ) as f:
            data_ts = dill.load(f)
        ci_ts = data_ts["controller_info_ts"]

        # trim the time series data
        ts = ci_ts["ts"] - ci_ts["ts"][0] - experiments[experiment_id].get("time_lag", 0.0)
        time_selector = ts >= START_TIME
        if END_TIME is not None:
            time_selector = time_selector & (ts <= END_TIME)
        ci_ts["ts"] = ts[time_selector]
        for key in ci_ts.keys():
            if key != "ts":
                ci_ts[key] = ci_ts[key][time_selector]

        experiments[experiment_id]["data_ts"] = data_ts
        experiments[experiment_id]["ci_ts"] = ci_ts

    # plot reference so that we can identify time lag between experiments
    fig = plt.figure(figsize=(5.0, 4.0), num="Step response: References for identifying time lag")
    ax = plt.gca()
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        ci_ts = experiments[experiment_id]["ci_ts"]
        ax.plot(ci_ts["ts"], ci_ts["chiee_des"][..., 0], label=experiment_id)
    plt.legend()
    plt.show()

    figsize = (5.0, 4.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lw_ref = 2.5  # linewidth for reference trajectory
    lw = 2.0  # linewidth for the actual trajectory
    dots = (1.2, 0.8)
    markevery = 10
    ms = 4.5  # marker size

    plt.figure(figsize=figsize, num="Step response: End-effector pose")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    for experiment_idx, experiment_id in enumerate(experiments.keys()):
        ci_ts = experiments[experiment_id]["ci_ts"]
        # plot the reference trajectory
        if experiment_idx == 0:
            ax1.plot(
                ci_ts["ts"],
                ci_ts["chiee_des"][:, 0] * 1e3,
                color=colors[0],
                linestyle="dotted",
                linewidth=lw_ref,
                dashes=dots,
                label=r"$p_\mathrm{x}^\mathrm{d}$ Reference",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["chiee_des"][:, 1] * 1e3,
                color=colors[1],
                linestyle="dotted",
                linewidth=lw_ref,
                dashes=dots,
                label=r"$p_\mathrm{y}^\mathrm{d}$ Reference",
            )
        ax1.plot(
            ci_ts["ts"],
            ci_ts["chiee"][:, 0] * 1e3,
            color=colors[0],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            marker=experiments[experiment_id]["marker"],
            markeredgecolor="black",
            markevery=markevery,
            markersize=ms,
            label=r"$p_\mathrm{x}$ " + experiments[experiment_id]["label"],
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["chiee"][:, 1] * 1e3,
            color=colors[1],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            marker=experiments[experiment_id]["marker"],
            markeredgecolor="black",
            markevery=markevery,
            markersize=ms,
            label=r"$p_\mathrm{y}$ " + experiments[experiment_id]["label"],
        )

    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"x-position $p_{\mathrm{ee},x}$ [mm]")
    ax2.set_ylabel(r"y-position $p_{\mathrm{ee},y}$ [mm]")
    ax1.set_ylim([-17.0, 18.0])
    ax2.set_ylim([100.0, 142.5])
    ax1.legend(loc=(0.57, 0.05), labelspacing=0.3)
    ax2.legend(loc=(0.57, 0.55), labelspacing=0.3)
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
                linestyle="dotted",
                linewidth=lw_ref,
                dashes=dots,
                label=r"$\kappa_\mathrm{be}^\mathrm{d}$ Ref",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["q_des"][:, 1],
                color=colors[1],
                linestyle="dotted",
                linewidth=lw_ref,
                dashes=dots,
                label=r"$\sigma_\mathrm{sh}^\mathrm{d}$ Ref",
            )
            ax2.plot(
                ci_ts["ts"],
                ci_ts["q_des"][:, 2],
                color=colors[2],
                linestyle="dotted",
                linewidth=lw_ref,
                dashes=dots,
                label=r"$\sigma_\mathrm{ax}^\mathrm{d}$ Ref",
            )
        ax1.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 0],
            color=colors[0],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            marker=experiments[experiment_id]["marker"],
            markeredgecolor="black",
            markevery=markevery,
            markersize=ms,
            label=r"$\kappa_\mathrm{be}$ " + experiments[experiment_id]["label"],
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 1],
            color=colors[1],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            marker=experiments[experiment_id]["marker"],
            markeredgecolor="black",
            markevery=markevery,
            markersize=ms,
            label=r"$\sigma_\mathrm{sh}$ " + experiments[experiment_id]["label"],
        )
        ax2.plot(
            ci_ts["ts"],
            ci_ts["q"][:, 2],
            color=colors[2],
            linestyle=experiments[experiment_id]["linestyle"],
            linewidth=lw,
            marker=experiments[experiment_id]["marker"],
            markeredgecolor="black",
            markevery=markevery,
            markersize=ms,
            label=r"$\sigma_\mathrm{ax}$ " + experiments[experiment_id]["label"],
        )

    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.set_ylim([-2.5, 10.5])
    ax2.set_ylim([-0.05, 0.52])
    ax1.legend(loc=(0.4, 0.6), fontsize="small", labelspacing=0.3)
    ax2.legend(
        loc=(0.4, 0.3), ncols=2, fontsize="small", columnspacing=0.5, labelspacing=0.3
    )
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.show()
