from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax.numpy as jnp
import matplotlib

matplotlib.use("Qt5Cairo")
from matplotlib import animation
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from hsa_planar_control.analysis.utils import trim_time_series_data

# latex text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

EXPERIMENT_NAME = "20230718_145921"  # experiment name
if EXPERIMENT_NAME == "20230718_145921":
    # manual setpoints trajectory
    DATA_REL_START_TIME = 35.0
    DURATION = 92.0
    SPEEDUP = 2.5
    SIGMA_A_EQ = 1.00834525
elif EXPERIMENT_NAME == "20230719_095505":
    # star trajectory
    DATA_REL_START_TIME = 48.2
    DURATION = 111.53
    SPEEDUP = 2.5
    SIGMA_A_EQ = 1.00352598
elif EXPERIMENT_NAME == "20230719_094822":
    # tud-flame trajectory
    DATA_REL_START_TIME = 48.7
    DURATION = 92.65
    SPEEDUP = 2.5
    SIGMA_A_EQ = 1.00233353
else:
    raise NotImplementedError("Please add the settings for the new experiment.")
STEP_SKIP = 2


def main():
    experiment_folder = Path("data") / "experiments" / EXPERIMENT_NAME
    data_ts = jnp.load(
        str(experiment_folder / ("rosbag2_" + EXPERIMENT_NAME + "_0.npz"))
    )
    data_ts = dict(data_ts)

    # absolute start time
    start_time = data_ts["ts_chiee_des"][0] + DATA_REL_START_TIME
    # trim the dictionary with the time series data
    data_ts = trim_time_series_data(data_ts, start_time, DURATION)

    assert (
        data_ts["ts_phi_ss"].shape[0] == data_ts["ts_phi_sat"].shape[0]
    ), "Currently we assume that the control input is computed at the same rate as the steady-state actuation."
    t_ts = data_ts["ts_phi_sat"]
    # frame rate
    frame_rate = SPEEDUP / STEP_SKIP * (1 / (t_ts[1:] - t_ts[:-1]).mean().item())
    print("Frame rate:", frame_rate)
    pbar = tqdm(total=t_ts.shape[0])

    fig = plt.figure(figsize=(5, 3.0), num="Control input", dpi=200)
    ax = plt.gca()
    phi_ss_lines = []
    for i in range(2):
        (line,) = ax.plot(
            [],
            [],
            color=colors[i],
            linestyle=":",
            linewidth=2.25,
            dashes=(1.2, 0.8),
            label=r"$\phi_{" + str(i + 1) + "}^\mathrm{ss}$",
        )
        phi_ss_lines.append(line)
    phi_sat_lines = []
    for i in range(2):
        (line,) = ax.plot(
            [],
            [],
            color=colors[i],
            label=r"$\phi_" + str(i + 1) + "$",
        )
        phi_sat_lines.append(line)
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylabel(r"Control input $\phi$ [rad]")
    ax.set_xlim(t_ts[0], t_ts[-1])
    ax.set_ylim(
        jnp.min(data_ts["phi_sat_ts"]) - jnp.pi / 16,
        jnp.max(data_ts["phi_sat_ts"]) + jnp.pi / 16,
    )
    ax.legend()
    ax.grid(True)
    plt.box(True)
    plt.tight_layout()

    def animate(time_idx):
        for _i, _line in enumerate(phi_ss_lines):
            _line.set_data(
                data_ts["ts_phi_ss"][:time_idx],
                data_ts["phi_ss_ts"][:time_idx, _i],
            )
        for _i, _line in enumerate(phi_sat_lines):
            _line.set_data(
                data_ts["ts_phi_sat"][:time_idx],
                data_ts["phi_sat_ts"][:time_idx, _i],
            )

        lines = phi_ss_lines + phi_sat_lines
        pbar.update(STEP_SKIP)
        return lines

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=jnp.arange(t_ts.shape[0], step=STEP_SKIP),
        interval=1000 / frame_rate,
        blit=True,
    )

    movie_writer = animation.FFMpegWriter(fps=frame_rate)
    ani.save(
        str(
            experiment_folder
            / f"{EXPERIMENT_NAME}_control_input_{SPEEDUP*100:.0f}x.mp4"
        ),
        writer=movie_writer,
    )

    plt.show()
    pbar.close()


if __name__ == "__main__":
    main()
