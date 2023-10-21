import dill
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cv2
from jax import Array, jit, vmap
import jax.numpy as jnp
import numpy as onp
from pathlib import Path


EXPERIMENT_NAME = "20231019_083240"  # experiment name

# SHOW additional plots for calibration purposes
CALIBRATE = False

HSA_MATERIAL = "fpu"  # "fpu" or "epu"
SOURCE_RES = 2160  # 2160 (for 4k) or 1080 (for 180p), the default is 2160p
if EXPERIMENT_NAME == "20230925_094023":
    # FPU manual setpoints trajectory with baseline PID controller
    DATA_REL_START_TIME = 0.0
    VIDEO_REL_START_TIME = 1.75 + DATA_REL_START_TIME
    DURATION = 110.0
    SPEEDUP = 4.0
    COMMIT_EVERY_N_FRAMES = 4
    ORIGIN_UV = jnp.array([556, 305], dtype=jnp.uint32)  # uv coordinates of the origin
    EE_UV = jnp.array(
        [556, 1041], dtype=jnp.uint32
    )  # uv coordinates of the end-effector
    OVERLAY_CURRENT_SETPOINT = True
    OVERLAY_END_EFFECTOR_POSITION = True
    OVERLAY_EE_HISTORY = False
    OVERLAY_EE_DES_HISTORY = False
    OVERLAY_VIRTUAL_BACKBONE = True
elif EXPERIMENT_NAME == "20230925_093236":
    # FPU manual setpoints trajectory with P-satI-D controller
    DATA_REL_START_TIME = 0.0
    VIDEO_REL_START_TIME = 2.0 + DATA_REL_START_TIME
    DURATION = 110.0
    SPEEDUP = 4.0
    COMMIT_EVERY_N_FRAMES = 4
    ORIGIN_UV = jnp.array([550, 300], dtype=jnp.uint32)  # uv coordinates of the origin
    EE_UV = jnp.array(
        [550, 1045], dtype=jnp.uint32
    )  # uv coordinates of the end-effector
    OVERLAY_CURRENT_SETPOINT = True
    OVERLAY_END_EFFECTOR_POSITION = True
    OVERLAY_EE_HISTORY = False
    OVERLAY_EE_DES_HISTORY = False
    OVERLAY_VIRTUAL_BACKBONE = True
elif EXPERIMENT_NAME == "20231019_081703":
    # FPU tud-flame trajectory with P-satI-D controller
    SOURCE_RES = 1080
    DATA_REL_START_TIME = 1.0
    VIDEO_REL_START_TIME = 1.703 + DATA_REL_START_TIME
    DURATION = 85.0 - DATA_REL_START_TIME
    SPEEDUP = 3.0
    COMMIT_EVERY_N_FRAMES = 2
    ORIGIN_UV = jnp.array([317, 174], dtype=jnp.uint32)  # uv coordinates of the origin
    EE_UV = jnp.array(
        [317, 588], dtype=jnp.uint32
    )  # uv coordinates of the end-effector
    OVERLAY_CURRENT_SETPOINT = True
    OVERLAY_END_EFFECTOR_POSITION = True
    OVERLAY_EE_HISTORY = True
    OVERLAY_EE_DES_HISTORY = True
    OVERLAY_VIRTUAL_BACKBONE = True
elif EXPERIMENT_NAME == "20231019_083240":
    # FPU large bat trajectory with P-satI-D controller
    # 1080p at 30 fps
    SOURCE_RES = 1080
    DATA_REL_START_TIME = 1.0
    VIDEO_REL_START_TIME = 4.80 + DATA_REL_START_TIME
    DURATION = 189.0 - DATA_REL_START_TIME
    SPEEDUP = 6.0
    COMMIT_EVERY_N_FRAMES = 3
    ORIGIN_UV = jnp.array([310, 178], dtype=jnp.uint32)  # uv coordinates of the origin
    EE_UV = jnp.array(
        [310, 589], dtype=jnp.uint32
    )  # uv coordinates of the end-effector
    OVERLAY_CURRENT_SETPOINT = True
    OVERLAY_END_EFFECTOR_POSITION = True
    OVERLAY_EE_HISTORY = True
    OVERLAY_EE_DES_HISTORY = True
    OVERLAY_VIRTUAL_BACKBONE = True
else:
    raise NotImplementedError("Please add the settings for the new experiment.")

# all sizes are defined for 4k resolution
res_mult = SOURCE_RES / 2160


@jit
def position_to_uv(origin: Array, res: Array, position: Array):
    """
    Convert a position in the world frame to uv coordinates
    Args:
        origin: origin in uv coordinates [pixel]
        res: resolution of the video [pixel/m]
        position: position in the world frame [m]
    Returns:
        uv: position in uv coordinates [pixel]
    """
    # flip the x-axis
    position = jnp.array([-position[0], position[1]])

    # convert to uv coordinates
    uv = (origin + res * position).astype(jnp.uint32)
    return uv


def main():
    num_segments = 1
    num_rods_per_segment = 2
    # filepath to symbolic expressions
    sym_exp_filepath = (
        Path(jsrm.__file__).parent
        / "symbolic_expressions"
        / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
    )

    # load robot parameters
    if HSA_MATERIAL == "fpu":
        params = PARAMS_FPU_CONTROL.copy()
    elif HSA_MATERIAL == "epu":
        params = PARAMS_EPU_CONTROL.copy()
    else:
        raise ValueError(f"Unknown HSA material: {HSA_MATERIAL}")
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)

    batched_position_to_uv = jit(vmap(position_to_uv, in_axes=(None, None, 0)))
    # we use for plotting N points along the length of the robot
    s_ps = jnp.linspace(0, jnp.sum(params["l"]), 50)
    batched_forward_kinematics_virtual_backbone_fn = jit(
        vmap(
            partial(forward_kinematics_virtual_backbone_fn, params),
            in_axes=(None, 0),
            out_axes=-1,
        )
    )

    # load processed ROS bag data
    experiment_folder = Path("data") / "experiments" / EXPERIMENT_NAME
    with open(
        str(experiment_folder / ("rosbag2_" + EXPERIMENT_NAME + "_0.dill")), "rb"
    ) as f:
        data_ts = dill.load(f)
    ci_ts = data_ts["controller_info_ts"]

    # the video should be already cropped and have aspect ratio 4:5
    video_path = (
        Path("data") / "experiments" / EXPERIMENT_NAME / (EXPERIMENT_NAME + ".mov")
    )
    overlayed_video_path = video_path.with_name(
        video_path.stem + f"_overlayed_{SPEEDUP*100:.0f}x.mp4"
    )

    if CALIBRATE:
        plt.plot(
            ci_ts["ts"] - ci_ts["ts"][0],
            ci_ts["chiee"][:, 1],
            label="y",
        )
        plt.xlabel("t [s]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.box(True)
        plt.show()

    # absolute start time
    data_start_time = ci_ts["ts"][0] + DATA_REL_START_TIME
    ci_ts["ts"] = ci_ts["ts"] - data_start_time
    ci_start_time_idx = jnp.argmin(jnp.abs(ci_ts["ts"]))  # find the data point closest to zero
    print("Experiment full duration:", ci_ts["ts"][-1])

    # trim the time series data
    if DURATION is not None:
        end_time_idx = jnp.argmax(ci_ts["ts"] > DURATION)
        for key in ci_ts.keys():
            ci_ts[key] = ci_ts[key][:end_time_idx]

    cap = cv2.VideoCapture(str(video_path))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        raise RuntimeError("Error opening video stream or file")

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    print("Frame size = ", frame_width, "x", frame_height, "pixels")

    # compute the resolution of the video
    res = (EE_UV[1] - ORIGIN_UV[1]).astype(jnp.float64) / ci_ts["chiee"][
        0, 1
    ]
    print("Identified resolution = ", res, "pixel/m")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    # compute the output fps
    out_fps = fps_in * SPEEDUP / COMMIT_EVERY_N_FRAMES
    print("Incoming fps = ", fps_in, "Hz,", "outgoing fps = ", out_fps, "Hz")

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal. Also frame size is passed.
    out_writer = cv2.VideoWriter(
        str(overlayed_video_path),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=out_fps,
        frameSize=(frame_width, frame_height),
    )

    # define colors
    colors_plt = {
        "red": mcolors.to_rgb(mcolors.TABLEAU_COLORS["tab:red"]),
        "gray": mcolors.to_rgb(mcolors.TABLEAU_COLORS["tab:gray"]),
        "green": mcolors.to_rgb(mcolors.TABLEAU_COLORS["tab:green"]),
        "blue": mcolors.to_rgb(mcolors.TABLEAU_COLORS["tab:blue"]),
    }
    colors_bgr = {}
    for key, value in colors_plt.items():
        color_rgb = (onp.array(mcolors.to_rgb(value)) * 255).astype(onp.uint8)
        colors_bgr[key] = color_rgb[::-1].tolist()

    # initialize some variables
    frame_idx_in = -1
    frame_idx_out = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret is True:
            frame_idx_in += 1
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3
            time = video_time - VIDEO_REL_START_TIME
            if (
                time < 0.0
                or (DURATION is not None and time > DURATION)
                or time > ci_ts["ts"][-1]
            ):
                continue

            if CALIBRATE and frame_idx_out == 0:
                plt.figure(num="First frame")
                plt.imshow(frame)
                plt.show()
                break

            # skip frames
            if frame_idx_in % COMMIT_EVERY_N_FRAMES != 0:
                continue

            # write current time to frame
            frame = cv2.putText(
                frame,
                f"t = {time:.2f} s",
                org=(20, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5 * res_mult,
                color=(255, 255, 255),
                thickness=int(4 * res_mult),
            )
            if SPEEDUP != 1.0:
                # write speedup to frame
                frame = cv2.putText(
                    frame,
                    f"{SPEEDUP:.1f} x",
                    org=(frame_width - 100, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5 * res_mult,
                    color=(255, 255, 255),
                    thickness=int(4 * res_mult),
                )

            # find the closest data points to the current time
            ci_time_idx = jnp.argmin(jnp.abs(ci_ts["ts"] - time))

            if OVERLAY_CURRENT_SETPOINT:
                # plot current setpoint
                pee_des_uv = onp.array(
                    position_to_uv(ORIGIN_UV, res, ci_ts["chiee_des"][ci_time_idx, :2])
                )
                frame = cv2.circle(
                    frame,
                    center=(pee_des_uv[0], pee_des_uv[1]),
                    radius=int(34 * res_mult),
                    color=colors_bgr["red"],
                    thickness=-1,
                )
            if OVERLAY_EE_DES_HISTORY and ci_time_idx > 0:
                pee_des_uv_hs = onp.array(
                    batched_position_to_uv(
                        ORIGIN_UV, res, ci_ts["chiee_des"][ci_start_time_idx:ci_time_idx:1, :2]
                    )
                )
                frame = cv2.polylines(
                    frame,
                    pts=[pee_des_uv_hs.astype(onp.int32)],
                    isClosed=False,
                    color=colors_bgr["red"],
                    thickness=int(14 * res_mult),
                )
            if OVERLAY_END_EFFECTOR_POSITION:
                pee_uv = onp.array(
                    position_to_uv(ORIGIN_UV, res, ci_ts["chiee"][ci_time_idx, :2])
                )
                frame = cv2.circle(
                    frame,
                    center=(pee_uv[0], pee_uv[1]),
                    radius=int(28 * res_mult),
                    color=(0, 0, 0),
                    thickness=-1,
                )
            if OVERLAY_EE_HISTORY and ci_time_idx > 0:
                # only plot every 16th data point to reduce the number of points
                pee_uv_hs = onp.array(
                    batched_position_to_uv(
                        ORIGIN_UV, res, ci_ts["chiee"][ci_start_time_idx:ci_time_idx:16, :2]
                    )
                )
                frame = cv2.polylines(
                    frame,
                    pts=[pee_uv_hs.astype(onp.int32)],
                    isClosed=False,
                    color=(0, 0, 0),
                    thickness=int(10 * res_mult),
                )
            if OVERLAY_VIRTUAL_BACKBONE:
                # poses along the robot of shape (3, N)
                chiv_ps = batched_forward_kinematics_virtual_backbone_fn(
                    ci_ts["q"][ci_time_idx], s_ps
                )  # poses of virtual backbone
                # draw the virtual backbone
                # add the first point of the proximal cap and the last point of the distal cap
                chiv_ps = jnp.concatenate(
                    [
                        # add poses of the proximal cap
                        (
                            chiv_ps[:, 0] - jnp.array([0.0, params["lpc"][0], 0.0])
                        ).reshape(3, 1),
                        # add the poses of the HSA rod
                        chiv_ps,
                        # add the poses of the distal cap
                        (
                            chiv_ps[:, -1]
                            + jnp.array(
                                [
                                    -jnp.sin(chiv_ps[2, -1]) * params["ldc"][-1],
                                    jnp.cos(chiv_ps[2, -1]) * params["ldc"][-1],
                                    chiv_ps[2, -1],
                                ]
                            )
                        ).reshape(3, 1),
                    ],
                    axis=1,
                )
                # map poses to uv coordinates
                chiv_uv_ps = onp.array(
                    batched_position_to_uv(ORIGIN_UV, res, chiv_ps[:2, :].T)
                )
                frame = cv2.polylines(
                    frame,
                    pts=[chiv_uv_ps.astype(onp.int32)],
                    isClosed=False,
                    color=colors_bgr["blue"],
                    thickness=int(10 * res_mult),
                )

            # Display the resulting frame
            cv2.imshow("Frame", frame)
            # use cv2 writer
            out_writer.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            frame_idx_out += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
