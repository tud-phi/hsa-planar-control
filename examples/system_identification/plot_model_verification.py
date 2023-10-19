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


# experiment_id = "20230621_165020"  # FPU staircase bending ccw
# experiment_id = "20230621_183620"  # FPU GBN bending combined 180 deg
experiment_id = "20230927_150452"  # EPU GBN bending combined 270 deg
START_TIME = 0.0
END_TIME = 4.0

if __name__ == "__main__":
    experiment_data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "system_identification"
        / experiment_id
    )

    with open(str(experiment_data_path / "preprocessed_data_history.dill"), "rb") as f:
        data_ts = dill.load(f)
    with open(str(experiment_data_path / "model_inference_history.dill"), "rb") as f:
        sim_ts = dill.load(f)

    # trim the time series data
    t_ts_data = data_ts["t_ts"] - data_ts["t_ts"][0]
    t_ts_sim = sim_ts["t_ts"] - sim_ts["t_ts"][0]
    data_time_selector = t_ts_data >= START_TIME
    sim_time_selector = t_ts_sim >= START_TIME
    if END_TIME is not None:
        data_time_selector = data_time_selector & (t_ts_data <= END_TIME)
        sim_time_selector = sim_time_selector & (t_ts_sim <= END_TIME)
    data_ts["t_ts"] = t_ts_data[data_time_selector]
    sim_ts["t_ts"] = t_ts_sim[sim_time_selector]
    for key in data_ts.keys():
        if key != "t_ts":
            data_ts[key] = data_ts[key][data_time_selector]
    for key in sim_ts.keys():
        if key not in ["t_ts", "controller_info_ts"]:
            sim_ts[key] = sim_ts[key][sim_time_selector]

    figsize = (5.0, 3)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lw_gt = 2.0
    lw_hat = 2.6
    dashes = (1.2, 0.8)

    plt.figure(figsize=figsize, num="Model-verification: End-effector pose")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(
        data_ts["t_ts"],
        data_ts["chiee_ts"][:, 0] * 1e3,
        color=colors[0],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$p_\mathrm{x}$",
    )
    ax1.plot(
        data_ts["t_ts"],
        data_ts["chiee_ts"][:, 1] * 1e3,
        color=colors[1],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$p_\mathrm{y}$",
    )
    ax2.plot(
        data_ts["t_ts"],
        data_ts["chiee_ts"][:, 2],
        color=colors[2],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$\theta_\mathrm{ee}$",
    )
    ax1.plot(
        sim_ts["t_ts"],
        sim_ts["chiee_ts"][:, 0] * 1e3,
        linestyle="--",
        linewidth=lw_hat,
        color=colors[0],
        label=r"$\hat{p}_\mathrm{x}$",
    )
    ax1.plot(
        sim_ts["t_ts"],
        sim_ts["chiee_ts"][:, 1] * 1e3,
        linestyle="--",
        linewidth=lw_hat,
        color=colors[1],
        label=r"$\hat{p}_\mathrm{y}$",
    )
    ax2.plot(
        sim_ts["t_ts"],
        sim_ts["chiee_ts"][:, 2],
        linestyle="--",
        linewidth=lw_hat,
        color=colors[2],
        label=r"$\hat{\theta}_\mathrm{ee}$",
    )
    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"End effector position $p_{\mathrm{ee}}$ [mm]")
    ax2.set_ylabel(r"End effector orientation $\theta_{\mathrm{ee}}$ [rad]")
    ax1.legend(loc="lower left", ncols=2, columnspacing=0.5, labelspacing=0.3)
    ax2.legend(loc="lower right", columnspacing=0.5, labelspacing=0.3)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(
        str(experiment_data_path / f"{experiment_id}_model_verification_chiee.pdf")
    )
    plt.savefig(
        str(experiment_data_path / f"{experiment_id}_model_verification_chiee.eps")
    )
    plt.show()

    fig = plt.figure(figsize=figsize, num="Model-verification: Strains")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 0],
        color=colors[0],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$\kappa_\mathrm{be}$",
    )
    ax2.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 1],
        color=colors[1],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$\sigma_\mathrm{sh}$",
    )
    ax2.plot(
        data_ts["t_ts"],
        data_ts["xi_ts"][:, 2],
        color=colors[2],
        linestyle="-",
        linewidth=lw_gt,
        label=r"$\sigma_\mathrm{ax}$",
    )
    ax1.plot(
        sim_ts["t_ts"],
        sim_ts["xi_ts"][:, 0],
        color=colors[0],
        linestyle="--",
        linewidth=lw_hat,
        label=r"$\hat{\kappa}_\mathrm{be}$",
    )
    ax2.plot(
        sim_ts["t_ts"],
        sim_ts["xi_ts"][:, 1],
        color=colors[1],
        linestyle="--",
        linewidth=lw_hat,
        label=r"$\hat{\sigma}_\mathrm{sh}$",
    )
    ax2.plot(
        sim_ts["t_ts"],
        sim_ts["xi_ts"][:, 2],
        color=colors[2],
        linestyle="--",
        linewidth=lw_hat,
        label=r"$\hat{\sigma}_\mathrm{ax}$",
    )
    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.legend(loc="upper left", columnspacing=0.5, labelspacing=0.3)
    ax2.legend(loc="upper right", ncols=2, columnspacing=0.5, labelspacing=0.3)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_data_path / f"{experiment_id}_model_verification_q.pdf"))
    plt.savefig(str(experiment_data_path / f"{experiment_id}_model_verification_q.eps"))
    plt.show()
