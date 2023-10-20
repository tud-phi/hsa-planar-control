import dill
from functools import partial
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

EXPERIMENT_NAME = "20230925_093236"  # experiment name
DURATION = 110.0  # default duration for manual setpoints trajectory [s]


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

    # compute the task space trajectory tracking metrics
    e_pee_ts = jnp.linalg.norm(ci_ts["chiee_des"][..., :2] - ci_ts["chiee"][..., :2], axis=1)
    rmse_pee = np.sqrt(np.mean(np.square(e_pee_ts), axis=0))
    print(f"RMSE of end-effector position: {rmse_pee}")


if __name__ == "__main__":
    main()
