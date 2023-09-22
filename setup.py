from glob import glob
import os
from setuptools import setup

package_name = "hsa_planar_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=[
        package_name,
        package_name + ".analysis",
        package_name + ".controllers",
        package_name + ".planning",
        package_name + ".rendering",
        package_name + ".ros_nodes",
        package_name + ".system_identification",
        package_name + ".system_identification.optimization",
    ],
    data_files=[
        (os.path.join("share", package_name), glob("launch/*.py")),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "assets"), glob("assets/*")),
    ],
    install_requires=[
        "derivative",
        "diffrax",
        "dill",
        "jax",
        "jaxopt",
        "jax-cosmo",
        "jax-spcs-kinematics",
        "jsrm",
        "numpy",
        "opencv-python",
        "pandas",
        "setuptools",
        "sympy>=1.11",
        "tornado",
        "tqdm",
    ],
    zip_safe=True,
    author="Maximilian Stölzle",
    author_email="maximilian@stoelzle.ch",
    maintainer="Maximilian Stölzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="Model-based control of Planar HSA robots. "
    "This package contains the simulation, system identification, planning and control algorithms."
    "Furthermore, we also provide ROS2 nodes for performing closed-loop control on the real robot.",
    license="MIT",
    tests_require=["codecov", "coverage", "pytest", "pytest-cov", "pytest-html", "tox"],
    entry_points={
        "console_scripts": [
            "model_based_control_node = hsa_planar_control.ros_nodes.model_based_control_node:main",
            "static_inversion_planning_node = hsa_planar_control.ros_nodes.static_inversion_planning_node:main",
            "calibration_node = hsa_planar_control.ros_nodes.calibration_node:main",
        ],
    },
)
