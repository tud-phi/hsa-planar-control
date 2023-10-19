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

    t_ts = data_ts["ts_q_des"]
    q_indices = jnp.linspace(0, data_ts["ts_q"].shape[0] - 1, t_ts.shape[0], dtype=int)
    # frame rate
    frame_rate = SPEEDUP / STEP_SKIP * (1 / (t_ts[1:] - t_ts[:-1]).mean().item())
    print("Frame rate:", frame_rate)
    pbar = tqdm(total=t_ts.shape[0])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth = 2.25
    dashes = (1.2, 0.8)

    fig = plt.figure(figsize=(5, 3.0), num="Configuration", dpi=200)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    (kappa_b_des_line,) = ax1.plot(
        [],
        [],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth,
        dashes=dashes,
        label=r"$\kappa_\mathrm{be}^\mathrm{d}$",
    )
    (sigma_sh_des_line,) = ax2.plot(
        [],
        [],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth,
        dashes=dashes,
        label=r"$\sigma_\mathrm{sh}^\mathrm{d}$",
    )
    (sigma_a_des_line,) = ax2.plot(
        [],
        [],
        color=colors[2],
        linestyle=":",
        linewidth=linewidth,
        dashes=dashes,
        label=r"$\sigma_\mathrm{ax}^\mathrm{d}$",
    )
    q_des_lines = [kappa_b_des_line, sigma_sh_des_line, sigma_a_des_line]
    (kappa_b_line,) = ax1.plot(
        [],
        [],
        color=colors[0],
        linewidth=linewidth,
        label=r"$\kappa_\mathrm{be}$",
    )
    (sigma_sh_line,) = ax2.plot(
        [],
        [],
        color=colors[1],
        linewidth=linewidth,
        label=r"$\sigma_\mathrm{sh}$",
    )
    (sigma_a_line,) = ax2.plot(
        [],
        [],
        color=colors[2],
        linewidth=linewidth,
        label=r"$\sigma_\mathrm{ax}$",
    )
    q_lines = [kappa_b_line, sigma_sh_line, sigma_a_line]
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.set_xlim(t_ts[0], t_ts[-1])
    ax1.set_ylim(
        jnp.min(data_ts["q_ts"][:, 0]) - jnp.pi / 4,
        jnp.max(data_ts["q_ts"][:, 0]) + jnp.pi / 4,
    )
    ax2.set_ylim(
        jnp.min(data_ts["q_ts"][:, 1:]) - 0.1, jnp.max(data_ts["q_ts"][:, 1:]) + 0.1
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True)
    plt.box(True)
    plt.tight_layout()

    def animate(time_idx):
        for _i, _line in enumerate(q_des_lines):
            _line.set_data(
                t_ts[:time_idx],
                data_ts["q_des_ts"][:time_idx, _i],
            )
        for _i, _line in enumerate(q_lines):
            _line.set_data(
                t_ts[:time_idx],
                data_ts["q_ts"][q_indices][:time_idx, _i],
            )

        lines = q_des_lines + q_lines
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
            / f"{EXPERIMENT_NAME}_configuration_{SPEEDUP*100:.0f}x.mp4"
        ),
        writer=movie_writer,
    )

    plt.show()
    pbar.close()


if __name__ == "__main__":
    main()
