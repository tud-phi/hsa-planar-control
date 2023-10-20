import dill
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
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


EXPERIMENT_NAME = "20230925_093236"  # experiment name

# SHOW additional plots for calibration purposes
CALIBRATE = True

# Attention: in this script we are expecting 4K footage
if EXPERIMENT_NAME == "20230925_093236":
    # manual setpoints trajectory
    VIDEO_REL_START_TIME = 0.0
    DURATION = 10.0
    SPEEDUP = 4.0
    COMMIT_EVERY_N_FRAMES = 4
    ORIGIN_UV = jnp.array([580, 355], dtype=jnp.uint32)  # uv coordinates of the origin
    EE_UV = jnp.array(
        [565, 1096], dtype=jnp.uint32
    )  # uv coordinates of the end-effector
    OVERLAY_CURRENT_SETPOINT = True
    OVERLAY_END_EFFECTOR_POSITION = True
    OVERLAY_EE_HISTORY = False
    OVERLAY_EE_DES_HISTORY = False
    OVERLAY_VIRTUAL_BACKBONE = True
else:
    raise NotImplementedError("Please add the settings for the new experiment.")


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
    params = PARAMS_FPU_CONTROL.copy()
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
    start_time = ci_ts["ts"][0]
    ci_ts["ts"] = ci_ts["ts"] - start_time
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

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    # compute the output fps
    out_fps = fps_in * SPEEDUP / COMMIT_EVERY_N_FRAMES
    print("Incoming fps = ", fps_in, "fps", "outgoing fps = ", out_fps, "fps")

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
    res = None

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

            # skip frames
            if frame_idx_in % COMMIT_EVERY_N_FRAMES != 0:
                continue

            if CALIBRATE and frame_idx_out == 0:
                plt.figure(num="First frame")
                plt.imshow(frame)
                plt.show()
                break

            # write current time to frame
            frame = cv2.putText(
                frame,
                f"t = {time:.2f} s",
                org=(20, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=4,
            )
            if SPEEDUP != 1.0:
                # write speedup to frame
                frame = cv2.putText(
                    frame,
                    f"{SPEEDUP:.1f} x",
                    org=(frame_width - 100, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 255, 255),
                    thickness=4,
                )

            # find the closest data points to the current time
            ci_time_idx = jnp.argmin(jnp.abs(ci_ts["ts"] - time))

            if res is None:
                # compute resolution of the video
                res = (EE_UV[1] - ORIGIN_UV[1]).astype(jnp.float64) / ci_ts["chiee"][
                    ci_time_idx, 1
                ]
                print("Identified resolution = ", res, "pixel/m")

            if OVERLAY_CURRENT_SETPOINT:
                # plot current setpoint
                pee_des_uv = onp.array(
                    position_to_uv(ORIGIN_UV, res, ci_ts["chiee_des"][ci_time_idx, :2])
                )
                frame = cv2.circle(
                    frame,
                    center=(pee_des_uv[0], pee_des_uv[1]),
                    radius=22,
                    color=colors_bgr["red"],
                    thickness=-1,
                )
            if OVERLAY_EE_DES_HISTORY and ci_time_idx > 0:
                pee_des_uv_hs = onp.array(
                    batched_position_to_uv(
                        ORIGIN_UV, res, ci_ts["chiee_des_hs"][:ci_time_idx:1, :2]
                    )
                )
                frame = cv2.polylines(
                    frame,
                    pts=[pee_des_uv_hs.astype(onp.int32)],
                    isClosed=False,
                    color=colors_bgr["red"],
                    thickness=6,
                )
            if OVERLAY_END_EFFECTOR_POSITION:
                pee_uv = onp.array(
                    position_to_uv(ORIGIN_UV, res, ci_ts["chiee"][ci_time_idx, :2])
                )
                frame = cv2.circle(
                    frame,
                    center=(pee_uv[0], pee_uv[1]),
                    radius=20,
                    color=(0, 0, 0),
                    thickness=-1,
                )
            if OVERLAY_EE_HISTORY and ci_time_idx > 0:
                # only plot every 16th data point to reduce the number of points
                pee_uv_hs = onp.array(
                    batched_position_to_uv(
                        ORIGIN_UV, res, ci_ts["chiee_hs"][:ci_time_idx:16, :2]
                    )
                )
                frame = cv2.polylines(
                    frame,
                    pts=[pee_uv_hs.astype(onp.int32)],
                    isClosed=False,
                    color=(0, 0, 0),
                    thickness=4,
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
                    thickness=8,
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
