from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
from jax import Array, jit, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_CONTROL
from jsrm.systems import planar_hsa
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
from pathlib import Path

from hsa_planar_control.collocated_form import mapping_into_collocated_form_factory
from hsa_planar_control.analysis.utils import trim_time_series_data

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
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
PLOT_COLLOCATED_COORDINATES = True


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

    if PLOT_COLLOCATED_COORDINATES:
        num_segments = 1
        num_rods_per_segment = 2
        # filepath to symbolic expressions
        sym_exp_filepath = (
            Path(jsrm.__file__).parent
            / "symbolic_expressions"
            / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
        )
        (
            forward_kinematics_virtual_backbone_fn,
            forward_kinematics_end_effector_fn,
            jacobian_end_effector_fn,
            inverse_kinematics_end_effector_fn,
            dynamical_matrices_fn,
            sys_helpers,
        ) = planar_hsa.factory(sym_exp_filepath)
        map_into_collocated_form_fn, _ = mapping_into_collocated_form_factory(
            sym_exp_filepath, sys_helpers
        )
        # params
        params = PARAMS_CONTROL.copy()
        params["sigma_a_eq"] = SIGMA_A_EQ * jnp.ones_like(params["sigma_a_eq"])

    figsize = (4.5, 3)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth_dotted = 2.25
    dashes = (1.2, 0.8)

    plt.figure(figsize=(4.5, 3), num="End-effector position")
    ax = plt.gca()
    ax.plot(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 0],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$p_\mathrm{x}^\mathrm{d}$",
    )
    ax.plot(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 1],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$p_\mathrm{y}^\mathrm{d}$",
    )
    ax.plot(
        data_ts["ts_chiee"],
        data_ts["chiee_ts"][:, 0],
        color=colors[0],
        label=r"$p_\mathrm{x}$",
    )
    ax.plot(
        data_ts["ts_chiee"],
        data_ts["chiee_ts"][:, 1],
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
    ax.plot(
        data_ts["ts_phi_ss"],
        data_ts["phi_ss_ts"][:, 0],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\phi^\mathrm{ss}_1$",
    )
    ax.plot(
        data_ts["ts_phi_ss"],
        data_ts["phi_ss_ts"][:, 1],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\phi^\mathrm{ss}_2$",
    )
    ax.plot(
        data_ts["ts_phi_sat"],
        data_ts["phi_sat_ts"][:, 0],
        color=colors[0],
        label=r"$\phi_1$",
    )
    ax.plot(
        data_ts["ts_phi_sat"],
        data_ts["phi_sat_ts"][:, 1],
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
        data_ts["ts_q_des"],
        data_ts["q_des_ts"][:, 0],
        color=colors[0],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\kappa_\mathrm{b}^\mathrm{d}$",
    )
    ax1.plot(
        data_ts["ts_q"],
        data_ts["q_ts"][:, 0],
        color=colors[0],
        label=r"$\kappa_\mathrm{b}$",
    )
    ax2.plot(
        data_ts["ts_q_des"],
        data_ts["q_des_ts"][:, 1],
        color=colors[1],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\sigma_\mathrm{sh}^\mathrm{d}$",
    )
    ax2.plot(
        data_ts["ts_q_des"],
        data_ts["q_des_ts"][:, 2],
        color=colors[2],
        linestyle=":",
        linewidth=linewidth_dotted,
        dashes=dashes,
        label=r"$\sigma_\mathrm{ax}^\mathrm{d}$",
    )
    ax2.plot(
        data_ts["ts_q"],
        data_ts["q_ts"][:, 1],
        color=colors[1],
        label=r"$\sigma_\mathrm{sh}$",
    )
    ax2.plot(
        data_ts["ts_q"],
        data_ts["q_ts"][:, 2],
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

    if PLOT_COLLOCATED_COORDINATES:
        t_ts = data_ts["ts_phi_ss"]
        q_indices = jnp.linspace(
            0, data_ts["ts_q"].shape[0] - 1, t_ts.shape[0], dtype=int
        )
        q_des_indices = jnp.linspace(
            0, data_ts["ts_q_des"].shape[0] - 1, t_ts.shape[0], dtype=int
        )
        varphi_ts, _ = vmap(
            partial(map_into_collocated_form_fn, params),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(data_ts["q_ts"][q_indices, ...], data_ts["phi_ss_ts"])
        varphi_des_ts, _ = vmap(
            partial(map_into_collocated_form_fn, params),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(data_ts["q_des_ts"][q_des_indices, ...], data_ts["phi_ss_ts"])

        plt.figure(figsize=figsize, num="Actuation coordinates")
        ax = plt.gca()
        ax.plot(
            t_ts,
            varphi_des_ts[:, 0],
            color=colors[0],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\varphi^\mathrm{d}_1$",
        )
        ax.plot(
            t_ts,
            varphi_des_ts[:, 1],
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dashes,
            label=r"$\varphi^\mathrm{d}_2$",
        )
        ax.plot(
            t_ts,
            varphi_ts[:, 0],
            color=colors[0],
            label=r"$\varphi_1$",
        )
        ax.plot(
            t_ts,
            varphi_ts[:, 1],
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
