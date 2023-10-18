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

TRAJ_TYPE = "star"  # "star", "tud-flame", "mit-csail

START_TIME = 0.0
END_TIME = None
if TRAJ_TYPE == "star":
    START_TIME = 1.0
    END_TIME = 66.0
    experiments = {
        # "20230925_100308": {"linestyle": "-", "label": "PID"},
        "20230925_095416": {"linestyle": "-", "label": "P-satI-D"},
        # "20230925_095851": {"linestyle": "-", "label": "P-satI-D+GC"},
    }
elif TRAJ_TYPE == "tud-flame":
    START_TIME = 1.0
    END_TIME = 60.0
    experiments = {
        # "20230925_101931": {"linestyle": "-", "label": "PID"},
        "20230925_102428": {"linestyle": "-", "label": "P-satI-D"},
        # "20230925_102856": {"linestyle": "-", "label": "P-satI-D+GC"},
    }
elif TRAJ_TYPE == "mit-csail":
    START_TIME = 1.0
    END_TIME = 49.0
    experiments = {
        # "20230925_113430": {"linestyle": "-", "label": "PID"},
        "20230925_113825": {"linestyle": "-", "label": "P-satI-D"},
        # "20230925_114328": {"linestyle": "-", "label": "P-satI-D+GC"},
    }
else:
    raise ValueError(f"Unknown trajectory type: {TRAJ_TYPE}")


experiment_folder = Path("data") / "experiments"


if __name__ == "__main__":
    fig = plt.figure(
        figsize=(3.0, 3.0), num=f"End-effector position: {TRAJ_TYPE} trajectory"
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
        if experiment_idx == 0:
            plt.plot(
                ci_ts["chiee_des"][:, 0] * 1e3,
                ci_ts["chiee_des"][:, 1] * 1e3,
                color="black",
                linestyle=":",
                dashes=(1.2, 0.8),
                linewidth=2.5,
                label=r"Ref",
            )

        # plot end-effector position trajectory
        plt.plot(
            ci_ts["chiee"][:, 0] * 1e3,
            ci_ts["chiee"][:, 1] * 1e3,
            color="C" + str(experiment_idx),
            linestyle=experiments[experiment_id]["linestyle"],
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
    plt.legend(loc="upper center", ncols=2, labelspacing=0.4, columnspacing=0.8)
    plt.tight_layout()
    plt.show()
