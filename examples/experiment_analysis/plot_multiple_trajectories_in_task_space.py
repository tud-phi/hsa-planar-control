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
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

TRAJ_TYPE = "bat"  # "star", "tud-flame", "mit-csail", "bat"

experiments = {
    "20231019_084914": {"color": colors[0], "label": "Small"},
    "20231019_084052": {"color": colors[1], "label": "Medium"},
    "20231019_083240": {"color": colors[2], "label": "Large"},
}

START_TIME = 1.0
END_TIME = 190.0

experiment_folder = Path("data") / "experiments"


if __name__ == "__main__":
    fig = plt.figure(
        figsize=(5.0, 2.7), num=f"End-effector position: {TRAJ_TYPE} trajectory"
    )
    ax = plt.gca()
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
        ts = ci_ts["ts"] - ci_ts["ts"][0]
        time_selector = ts >= START_TIME
        if END_TIME is not None:
            time_selector = time_selector & (ts <= END_TIME)
        ts = ts[time_selector]
        for key in ci_ts.keys():
            ci_ts[key] = ci_ts[key][time_selector]

        experiments[experiment_id]["data_ts"] = data_ts
        experiments[experiment_id]["ci_ts"] = ci_ts

        # plot the reference trajectory
        plt.plot(
            ci_ts["chiee_des"][:, 0] * 1e3,
            ci_ts["chiee_des"][:, 1] * 1e3,
            color=experiments[experiment_id]["color"],
            linestyle=":",
            dashes=(1.2, 0.8),
            linewidth=2.5,
        )

        # plot end-effector position trajectory
        plt.plot(
            ci_ts["chiee"][:, 0] * 1e3,
            ci_ts["chiee"][:, 1] * 1e3,
            color=experiments[experiment_id]["color"],
            linewidth=2.0,
            label=experiments[experiment_id]["label"],
        )

    plt.axis("equal")
    plt.xlabel(r"$x$ [mm]")
    plt.ylabel(r"$y$ [mm]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.grid(True)
    plt.box(True)
    plt.legend(loc="upper center", ncols=3, labelspacing=0.4, columnspacing=0.8)
    plt.tight_layout()
    plt.show()
