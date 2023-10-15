import cv2
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from os import PathLike

from hsa_planar_control.operational_workspace import (
    get_operational_workspace_boundaries
)


def generate_task_space_trajectory_from_image_contour(
    image_type: str,
    image_path: PathLike = None,
    hsa_material: str = "fpu",
    pee_centroid: Array = jnp.array([0.0, 0.13]),
    max_radius: Array = jnp.array(0.01),
    verbose: bool = True,
    show_images: bool = True,
):
    """
    Generate a task space trajectory from an image.
    Args:
        image_type: either "star" or "tud-flame"
        image_path: path to the image with the contour
        hsa_material: material of the HSA (either "fpu" or "epu")
        pee_centroid: the end effector position matching the centroid of the contour [m]
        max_radius: maximum radius of the contour (i.e. largest distance from the centroid) [m]
        verbose: if True, print debug information
        show_images: if True, show the images for debugging purposes

    Returns:
        pee_des_sps: the task space trajectory as array of shape (N, 2)
    """
    if image_type == "star":
        if image_path is None:
            image_path = Path(__file__).parent.parent.parent / "assets" / "star.png"

        img = cv2.imread(str(image_path))

        # perform cropping of the width to the height
        w = img.shape[0]
        center = img.shape
        x = center[1] / 2 - w / 2
        img = img[:, int(x) : int(x + w)]

        # set the threshold for the binary image
        threshold = 128
        threshold_mode = cv2.THRESH_BINARY

    elif image_type == "tud-flame":
        if image_path is None:
            image_path = (
                Path(__file__).parent.parent.parent / "assets" / "tud_flame.jpeg"
            )

        img = cv2.imread(str(image_path))

        # set the threshold for the binary image
        threshold = 140
        threshold_mode = cv2.THRESH_BINARY_INV
    elif image_type == "mit-csail":
        if image_path is None:
            image_path = (
                Path(__file__).parent.parent.parent / "assets" / "mit_csail.png"
            )

        img = cv2.imread(str(image_path))

        # set the threshold for the binary image
        threshold = 140
        threshold_mode = cv2.THRESH_BINARY_INV
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    if verbose:
        print("Image size: ", img.shape)

    # convert the input image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding to convert grayscale to binary image
    ret, img_thresh = cv2.threshold(img_gray, threshold, 255, threshold_mode)

    if show_images:
        # Display the Grayscale Image
        cv2.imshow("Gray Image", img_gray)
        cv2.waitKey(0)

        # Display the Binary Image
        cv2.imshow("Binary Image", img_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(
        image=img_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )
    contour = contours[0]
    if verbose:
        print("contour shape", contour.shape)

    if show_images:
        # draw contours on the original image
        img_copy = img.copy()
        cv2.drawContours(
            image=img_copy,
            contours=contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # see the results
        cv2.imshow("None approximation", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # find centroid of the contour
    M = cv2.moments(contour)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = jnp.array((cX, cY))
    if show_images:
        # put text and highlight the center
        img_copy = img.copy()
        cv2.circle(img_copy, (cX, cY), 5, (0, 0, 0), -1)
        cv2.putText(
            img_copy,
            "centroid",
            (cX - 25, cY - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

        # display the image
        cv2.imshow("Image with centroid of contour", img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # subtract the centroid from the contour
    contour = contour - centroid
    # normalize to range [-1, 1]
    pee_sps_norm = (contour / jnp.max(jnp.abs(contour)))[:, 0, :]
    # flip the y axis to get a right handed coordinate system
    pee_sps_norm = pee_sps_norm.at[..., 1].set(-pee_sps_norm[..., 1])
    if show_images:
        plt.figure(num="Normalized contour")
        plt.plot(pee_sps_norm[:, 0], pee_sps_norm[:, 1])
        plt.axis("equal")
        plt.grid(True)
        plt.box(True)
        plt.show()

    # reduce the number of points
    if image_type == "star":
        # pee_sps_norm = jnp.array([
        #     [0.0, -1.0],
        #     [-0.225, -0.310],
        #     [-0.950, -0.310],
        #     [-0.365, 0.125],
        #     [-0.585, 0.805],
        #     [0.0, 0.385],
        #     [0.585, 0.805],
        #     [0.365, 0.125],
        #     [0.950, -0.310],
        #     [0.225, -0.310],
        #     [0.0, -1.0],
        # ])
        sample_step = 5  # only take every 5th point
        pee_sps_norm = pee_sps_norm[::sample_step, :]
    elif image_type == "tud-flame":
        sample_step = 5  # only take every 5th point
        pee_sps_norm = pee_sps_norm[::sample_step, :]
        # as the robot is facing upside-down, we need to flip the x-axis and y-axis
        pee_sps_norm = -pee_sps_norm
    elif image_type == "mit-csail":
        sample_step = 8  # only take every 8th point
        pee_sps_norm = pee_sps_norm[::sample_step, :]
        # as the robot is facing upside-down, we need to flip the x-axis and y-axis
        pee_sps_norm = -pee_sps_norm

    pee_des_sps = pee_centroid + pee_sps_norm * max_radius
    if show_images:
        # evaluate the operational workspace boundaries
        pee_min_ps, pee_max_ps = get_operational_workspace_boundaries(
            hsa_material=hsa_material
        )

        plt.figure(num="Final trajectory")
        ax = plt.gca()
        plt.plot(pee_des_sps[:, 0], pee_des_sps[:, 1], label="planned end-effector trajectory")
        plt.plot(pee_min_ps[:, 0], pee_min_ps[:, 1], "k--", label="operational workspace boundary")
        plt.plot(pee_max_ps[:, 0], pee_max_ps[:, 1], "k--")
        plt.axis("equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.box(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("pee_des_sps:\n", pee_des_sps)

    return pee_des_sps
