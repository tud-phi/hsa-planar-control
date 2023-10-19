from jax import Array, jit, vmap
import jax.numpy as jnp
from functools import partial
from matplotlib import pyplot as plt
from os import PathLike
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import spcs_kinematics.jax_math as jmath
from typing import Callable, Dict
import warnings

from .utils import resample_trajectory


# vmap and jit some functions
vquat_SE3_to_se3 = jit(
    vmap(
        fun=jmath.quat_SE3_to_se3,
        in_axes=0,
        out_axes=0,
    ),
)
vinverse_transformation_matrix = jit(
    vmap(
        fun=jmath.inverse_transformation_matrix,
        in_axes=0,
        out_axes=0,
    )
)
vrotmat_to_euler_xyz = jit(
    vmap(
        jmath.rotmat_to_euler_xyz,
        in_axes=0,
        out_axes=0,
    )
)


def preprocess_data(
    inverse_kinematics_end_effector_fn: Callable,
    experimental_data_path: PathLike,
    known_params: Dict[str, Array],
    mocap_body_ids: Dict[str, int],
    use_raw_mocap_data: bool = True,
    resample: bool = True,
    resampling_dt: float = 0.01,  # 100 Hz
    filter: bool = False,
    filter_window_length: int = 5,
    derivative_method: str = "savgol_filter",
    plotting: bool = False,
) -> Dict[str, Array]:
    """
    Preprocesses the experimental data for system identification. This includes:
    - extracting the motor and mocap data
    - fitting cubic splines to the data
    - resampling both data sources to a common time base
    - running the inverse kinematics to obtain the configuration data
    - computing the velocities and accelerations of the configuration data using finite differences
    Arguments:
        inverse_kinematics_end_effector_fn: Inverse kinematics function that takes the end effector position and orientation as input and returns the configuration vector.
        experimental_data_path: Path to the experimental data folder.
        known_params: Dictionary with known robot parameters. This should include all geometric parameters
        mocap_body_ids: Dictionary with the body ids of the mocap markers. For example, {"base": 4, "platform": 5}.
        use_raw_mocap_data: Whether to use the raw mocap data or the data from the inverse kinematics.
        resample: Whether to resample the data to a common time base.
        resampling_dt: Time step for resampling.
        filter: Whether to filter the data. This applies a Savitzky-Golay filter to the data.
        filter_window_length: Window length for the Savitzky-Golay filter.
        derivative_method: Method for computing the derivatives. Either "finite_difference" or "savgol_filter".
        plotting: Whether to plot the data.
    Returns:
        data_filt_ts: Dictionary with the preprocessed data.
    """
    experiment_data_path = Path(experimental_data_path)
    experiment_id = experiment_data_path.name
    print(f"Preprocessing data for experiment {experiment_id}")

    data_raw_ts = {}
    data_raw_ts["t_ts"] = jnp.array(
        pd.read_csv(
            str(experiment_data_path / "experiment_time_history.csv"),
        ).to_numpy()
    ).squeeze()
    data_raw_ts["u_ts"] = jnp.array(
        pd.read_csv(
            str(experiment_data_path / "present_positions_history.csv"),
        ).to_numpy()
    )

    # read out mocap data
    if use_raw_mocap_data:
        assert resample is True, "Resampling must be enabled when using raw mocap data"
        df_mocap = pd.read_csv(
            str(Path(experimental_data_path) / f"take_{experiment_id}.csv"),
            skiprows=7,
            usecols=[i for i in range(16)],
        )
        df_mocap.columns = [
            "time_idx",
            "t_mts",
            "qb_x",
            "qb_y",
            "qb_z",
            "qb_w",
            "xb",
            "yb",
            "zb",
            "qp_x",
            "qp_y",
            "qp_z",
            "qp_w",
            "xp",
            "yp",
            "zp",
        ]
        print(df_mocap)
        # remove all lines with NaN values
        df_mocap = df_mocap.dropna()
        data_raw_ts["t_mts"] = jnp.array(df_mocap["t_mts"].to_numpy())
        data_raw_ts["chib3D_ts"] = jnp.array(
            df_mocap[["qb_x", "qb_y", "qb_z", "qb_w", "xb", "yb", "zb"]].to_numpy()
        )  # 3D pose of the robot base
        data_raw_ts["chip3D_ts"] = jnp.array(
            df_mocap[["qp_x", "qp_y", "qp_z", "qp_w", "xp", "yp", "zp"]].to_numpy()
        )  # 3D pose of the robot platform
    else:
        data_raw_ts["t_mts"] = data_raw_ts[
            "t_ts"
        ].copy()  # array with timing of mocap data
        df_mocap = pd.read_csv(
            str(Path(experimental_data_path) / "mocap_frame_data_history.csv"),
            names=["time_idx", "rigid_body_id", "qx", "qy", "qz", "qw", "x", "y", "z"],
        )
        print(df_mocap)
        # 3D pose of the robot base
        df_mocap_base = df_mocap[df_mocap["rigid_body_id"] == mocap_body_ids["base"]]
        df_mocap_platform = df_mocap[
            df_mocap["rigid_body_id"] == mocap_body_ids["platform"]
        ]
        data_raw_ts["chib3D_ts"] = jnp.array(
            df_mocap_base[["qx", "qy", "qz", "qw", "x", "y", "z"]].to_numpy()
        )
        data_raw_ts["chip3D_ts"] = jnp.array(
            df_mocap_platform[["qx", "qy", "qz", "qw", "x", "y", "z"]].to_numpy()
        )

    # make sure that the matlab data has all the same length
    matlab_data_keys = ["t_ts", "u_ts"]
    if use_raw_mocap_data is False:
        matlab_data_keys.extend(["t_mts", "chib3D_ts", "chip3D_ts"])
    num_matlab_samples_per_category = jnp.array(
        [data_raw_ts[key].shape[0] for key in matlab_data_keys]
    )
    num_matlab_samples = jnp.min(num_matlab_samples_per_category)
    for key in matlab_data_keys:
        data_raw_ts[key] = data_raw_ts[key][:num_matlab_samples, ...]

    # poses from SE(3) to se(3) (e.g. transformation matrices)
    Tb_ts = vquat_SE3_to_se3(data_raw_ts["chib3D_ts"])
    Tp_ts = vquat_SE3_to_se3(data_raw_ts["chip3D_ts"])
    # remove the translational offset of the markers
    # the MoCap markers of the base are 3.5 mm above the base frame (e.g. top surface of the top motor mounting plate)
    # the MoCap markers of the platform are roughly 7 mm above the end-effector frame (e.g. top surface of the platform)
    Tb_ts = Tb_ts.at[:, :3, 3].set(Tb_ts[:, :3, 3] - jnp.array([0.0, 3.5e-3, 0.0]))
    Tp_ts = Tp_ts.at[:, :3, 3].set(Tp_ts[:, :3, 3] - jnp.array([0.0, 7e-3, 0.0]))
    # transformation matrix from base frame to platform frame
    T_bp_ts = Tp_ts.at[:, :3, 3].set(Tp_ts[:, :3, 3] - Tb_ts[:, :3, 3])
    # project into the x-y plane
    euler_xyz_ts = vrotmat_to_euler_xyz(T_bp_ts[:, :3, :3])
    data_raw_ts["chiee_ts"] = jnp.zeros((data_raw_ts["chib3D_ts"].shape[0], 3))
    # z-axis of world becomes the x-axis
    data_raw_ts["chiee_ts"] = data_raw_ts["chiee_ts"].at[:, 0].set(T_bp_ts[:, 2, 3])
    # y-axis of world becomes the y-axis
    data_raw_ts["chiee_ts"] = data_raw_ts["chiee_ts"].at[:, 1].set(T_bp_ts[:, 1, 3])
    # angle around x-axis becomes negative theta
    data_raw_ts["chiee_ts"] = data_raw_ts["chiee_ts"].at[:, 2].set(-euler_xyz_ts[:, 0])

    # map motor positions to motor angles
    motor_neutral_positions = data_raw_ts["u_ts"][0, ...]
    data_raw_ts["phi_ts"] = (
        (data_raw_ts["u_ts"] - motor_neutral_positions) * 2 * jnp.pi / 4096
    )

    if plotting:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax1 = plt.subplots(num="Time-alignment of motor with MoCap data")
        ax2 = ax1.twinx()
        ax1.plot(
            data_raw_ts["t_mts"],
            data_raw_ts["chiee_ts"][:, 0],
            marker=".",
            color=colors[0],
            label="$p_\mathrm{x}$",
        )
        ax1.plot(
            data_raw_ts["t_mts"],
            data_raw_ts["chiee_ts"][:, 1],
            marker=".",
            color=colors[1],
            label="$p_\mathrm{y}$",
        )
        ax2.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 0],
            marker=".",
            color=colors[2],
            label="$\phi_{21}$",
        )
        ax2.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 1],
            marker=".",
            color=colors[3],
            label="$\phi_{22}$",
        )
        ax2.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 2],
            marker=".",
            color=colors[4],
            label="$\phi_{23}$",
        )
        ax2.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 3],
            marker=".",
            color=colors[5],
            label="$\phi_{24}$",
        )
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.set_xlabel("Time $t$ [s]")
        ax1.set_ylabel("End-effector position [m]")
        ax2.set_ylabel("Motor angle [rad]")
        ax1.grid(True)
        ax1.set_frame_on(True)
        plt.show()

    # resample
    if resample:
        if use_raw_mocap_data is True:
            resampling_dt_user_defined = resampling_dt
            resampling_dt = jnp.mean(jnp.diff(data_raw_ts["t_mts"]))
            warnings.warn(
                f"Ignoring the set resampling_dt = {resampling_dt_user_defined} s and "
                f"using the time-step of the MoCap data instead, which is {resampling_dt} s."
            )

        start_time = max(data_raw_ts["t_ts"][0].item(), data_raw_ts["t_mts"][0].item())
        end_time = min(data_raw_ts["t_ts"][-1].item(), data_raw_ts["t_mts"][-1].item())
        t_ts = jnp.arange(start_time, end_time, step=resampling_dt)

        phi_spline_order = 2
        chiee_spline_order = 3

        data_res_ts = {
            "t_ts": t_ts,
            "t_mts": t_ts,
            "phi_ts": resample_trajectory(
                data_raw_ts["t_ts"], data_raw_ts["phi_ts"], t_ts, k=phi_spline_order
            ),
            "chiee_ts": resample_trajectory(
                data_raw_ts["t_mts"],
                data_raw_ts["chiee_ts"],
                t_ts,
                k=chiee_spline_order,
            ),
        }

        if plotting:
            # data for plotting
            t_pts = jnp.arange(
                data_res_ts["t_ts"][0].item(),
                data_res_ts["t_ts"][-1].item(),
                step=0.001,
            )
            data_pts = {
                "t_ts": t_pts,
                "phi_ts": resample_trajectory(
                    data_raw_ts["t_ts"],
                    data_raw_ts["phi_ts"],
                    t_pts,
                    k=phi_spline_order,
                ),
                "chiee_ts": resample_trajectory(
                    data_raw_ts["t_mts"],
                    data_raw_ts["chiee_ts"],
                    t_pts,
                    k=chiee_spline_order,
                ),
            }
    else:
        data_res_ts = data_raw_ts

    if plotting:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax1 = plt.subplots(num="End-effector pose vs. time")
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.plot(
            data_raw_ts["t_mts"],
            data_raw_ts["chiee_ts"][:, 0],
            linestyle="None",
            marker=".",
            color=colors[0],
            label=r"$p_\mathrm{x}$",
        )
        ax1.plot(
            data_raw_ts["t_mts"],
            data_raw_ts["chiee_ts"][:, 1],
            linestyle="None",
            marker=".",
            color=colors[1],
            label=r"$p_\mathrm{y}$",
        )
        ax2.plot(
            data_raw_ts["t_mts"],
            data_raw_ts["chiee_ts"][:, 2],
            linestyle="None",
            marker=".",
            color=colors[2],
            label=r"$\theta$",
        )
        if resample:
            ax1.plot(
                data_pts["t_ts"],
                data_pts["chiee_ts"][:, 0],
                linestyle="-",
                color=colors[0],
                label=r"$\hat{p}_\mathrm{x}$",
            )
            ax1.plot(
                data_pts["t_ts"],
                data_pts["chiee_ts"][:, 1],
                linestyle="-",
                color=colors[1],
                label=r"$\hat{p}_\mathrm{y}$",
            )
            ax2.plot(
                data_pts["t_ts"],
                data_pts["chiee_ts"][:, 2],
                linestyle="-",
                color=colors[2],
                label=r"$\hat{\theta}$",
            )
            ax1.plot(
                data_res_ts["t_ts"],
                data_res_ts["chiee_ts"][:, 0],
                linestyle="None",
                marker="x",
                color=colors[0],
                label=r"$\hat{p}_\mathrm{x}$",
            )
            ax1.plot(
                data_res_ts["t_ts"],
                data_res_ts["chiee_ts"][:, 1],
                linestyle="None",
                marker="x",
                color=colors[1],
                label=r"$\hat{p}_\mathrm{y}$",
            )
            ax2.plot(
                data_res_ts["t_ts"],
                data_res_ts["chiee_ts"][:, 2],
                linestyle="None",
                marker="x",
                color=colors[2],
                label=r"$\hat{\theta}$",
            )
        plt.xlabel("Time [s]")
        ax1.set_ylabel("Position [m]")
        ax2.set_ylabel("Orientation [rad]")
        plt.grid(True)
        plt.box(True)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        plt.figure(num="Current motor angles vs. time")
        plt.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 0],
            linestyle="None",
            marker=".",
            color=colors[0],
            label="$\phi_{21}$",
        )
        plt.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 1],
            linestyle="None",
            marker=".",
            color=colors[1],
            label="$\phi_{22}$",
        )
        plt.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 2],
            linestyle="None",
            marker=".",
            color=colors[2],
            label="$\phi_{23}$",
        )
        plt.plot(
            data_raw_ts["t_ts"],
            data_raw_ts["phi_ts"][:, 3],
            linestyle="None",
            marker=".",
            color=colors[3],
            label="$\phi_{24}$",
        )
        if resample:
            plt.plot(
                data_pts["t_ts"],
                data_pts["phi_ts"][:, 0],
                linestyle="-",
                color=colors[0],
                label="$\hat{\phi}_{21}$",
            )
            plt.plot(
                data_pts["t_ts"],
                data_pts["phi_ts"][:, 1],
                linestyle="-",
                color=colors[1],
                label="$\hat{\phi}_{22}$",
            )
            plt.plot(
                data_pts["t_ts"],
                data_pts["phi_ts"][:, 2],
                linestyle="-",
                color=colors[2],
                label="$\hat{\phi}_{23}$",
            )
            plt.plot(
                data_pts["t_ts"],
                data_pts["phi_ts"][:, 3],
                linestyle="-",
                color=colors[3],
                label="$\hat{\phi}_{24}$",
            )
            plt.plot(
                data_res_ts["t_ts"],
                data_res_ts["phi_ts"][:, 0],
                linestyle="None",
                marker="x",
                color=colors[0],
                label="$\hat{\phi}_{21}$",
            )
            plt.plot(
                data_res_ts["t_ts"],
                data_res_ts["phi_ts"][:, 1],
                linestyle="None",
                marker="x",
                color=colors[1],
                label="$\hat{\phi}_{22}$",
            )
            plt.plot(
                data_res_ts["t_ts"],
                data_res_ts["phi_ts"][:, 2],
                linestyle="None",
                marker="x",
                color=colors[2],
                label="$\hat{\phi}_{23}$",
            )
            plt.plot(
                data_res_ts["t_ts"],
                data_res_ts["phi_ts"][:, 3],
                linestyle="None",
                marker="x",
                color=colors[3],
                label="$\hat{\phi}_{24}$",
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Motor position / rod twist angle [rad]")
        plt.grid(True)
        plt.box(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # run inverse kinematics
    # set dummy values for the rest strains
    known_params = known_params.copy()
    known_params["kappa_b_eq"] = jnp.zeros_like(known_params["roff"])
    known_params["sigma_sh_eq"] = jnp.zeros_like(known_params["roff"])
    known_params["sigma_a_eq"] = jnp.zeros_like(known_params["roff"])
    data_res_ts["xi_ts"] = vmap(
        partial(inverse_kinematics_end_effector_fn, known_params),
        in_axes=0,
        out_axes=0,
    )(data_res_ts["chiee_ts"])

    data_filt_ts = data_res_ts.copy()
    if filter:
        data_filt_ts = data_res_ts.copy()
        # filter using savgol filter
        data_filt_ts["xi_ts"] = jnp.array(
            savgol_filter(
                data_res_ts["xi_ts"],
                window_length=filter_window_length,
                polyorder=3,
                axis=0,
            )
        )

    if derivative_method == "finite_difference":
        # differentiate the configuration using finite differences
        # we receive the configuration velocity
        data_filt_ts["xi_d_ts"] = jnp.gradient(
            data_res_ts["xi_ts"], resampling_dt, axis=0
        )

        # differentiate the configuration velocity using finite differences
        # we receive the configuration velocity
        data_filt_ts["xi_dd_ts"] = jnp.gradient(
            data_filt_ts["xi_d_ts"], resampling_dt, axis=0
        )
    elif derivative_method == "savgol_filter":
        data_filt_ts["xi_d_ts"] = jnp.array(
            savgol_filter(
                data_res_ts["xi_ts"],
                window_length=filter_window_length,
                polyorder=3,
                deriv=1,
                delta=resampling_dt,
                axis=0,
            )
        )
        data_filt_ts["xi_dd_ts"] = jnp.array(
            savgol_filter(
                data_res_ts["xi_ts"],
                window_length=filter_window_length,
                polyorder=3,
                deriv=2,
                delta=resampling_dt,
                axis=0,
            )
        )
    else:
        raise NotImplementedError

    if plotting:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, (ax1a, ax2a, ax3a) = plt.subplots(
            num=r"$q$ and $\dot{q}$ vs. time", nrows=3, figsize=(10, 10)
        )

        ax1b = ax1a.twinx()  # instantiate a second axes that shares the same x-axis
        # filtered data
        ax1a.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_ts"][:, 0],
            color=colors[0],
            label=r"$\hat{\kappa}_\mathrm{be}$",
        )
        ax1b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_ts"][:, 1],
            color=colors[1],
            label=r"$\hat{\sigma}_\mathrm{sh}$",
        )
        ax1b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_ts"][:, 2],
            color=colors[2],
            label=r"$\hat{\sigma}_\mathrm{a}$",
        )
        if filter:
            # plot the unfiltered data points
            ax1a.plot(
                data_res_ts["t_ts"],
                data_res_ts["xi_ts"][:, 0],
                linestyle="None",
                marker=".",
                color=colors[0],
                label=r"$\kappa_\mathrm{be}$",
            )
            ax1b.plot(
                data_res_ts["t_ts"],
                data_res_ts["xi_ts"][:, 1],
                linestyle="None",
                marker=".",
                color=colors[1],
                label=r"$\sigma_\mathrm{sh}$",
            )
            ax1b.plot(
                data_res_ts["t_ts"],
                data_res_ts["xi_ts"][:, 2],
                linestyle="None",
                marker=".",
                color=colors[2],
                label=r"$\sigma_\mathrm{a}$",
            )
        plt.xlabel("Time [s]")
        ax1a.set_ylabel("Rotational strain [rad/m]")
        ax1b.set_ylabel("Linear strain [-]")
        ax1a.legend(loc="upper left")
        ax1b.legend(loc="upper right")
        ax1a.grid(True)
        ax1a.set_frame_on(True)

        ax2b = ax2a.twinx()  # instantiate a second axes that shares the same x-axis
        ax2a.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_d_ts"][:, 0],
            color=colors[0],
            label=r"$\dot{\kappa}_\mathrm{be}$",
        )
        ax2b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_d_ts"][:, 1],
            color=colors[1],
            label=r"$\dot{\sigma}_\mathrm{sh}$",
        )
        ax2b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_d_ts"][:, 2],
            color=colors[2],
            label=r"$\dot{\sigma}_\mathrm{a}$",
        )
        ax2a.set_xlabel("Time [s]")
        ax2a.set_ylabel("Rotational strain velocity [rad/(m s)]")
        ax2b.set_ylabel("Linear strain velocity [1/s]")
        ax2a.legend(loc="upper left")
        ax2b.legend(loc="upper right")
        ax2a.grid(True)
        ax2a.set_frame_on(True)

        ax3b = ax3a.twinx()  # instantiate a second axes that shares the same x-axis
        ax3a.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_dd_ts"][:, 0],
            color=colors[0],
            label=r"$\ddot{\kappa}_\mathrm{be}$",
        )
        ax3b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_dd_ts"][:, 1],
            color=colors[1],
            label=r"$\ddot{\sigma}_\mathrm{sh}$",
        )
        ax3b.plot(
            data_filt_ts["t_ts"],
            data_filt_ts["xi_dd_ts"][:, 2],
            color=colors[2],
            label=r"$\ddot{\sigma}_\mathrm{a}$",
        )
        ax3a.set_xlabel("Time [s]")
        ax3a.set_ylabel("Rotational strain acceleration [rad/(m s^2)]")
        ax3b.set_ylabel("Linear strain acceleration [1/s^2]")
        ax3a.legend(loc="upper left")
        ax3b.legend(loc="upper right")
        ax3a.grid(True)
        ax3a.set_frame_on(True)

        plt.tight_layout()
        plt.show()

    return data_filt_ts
