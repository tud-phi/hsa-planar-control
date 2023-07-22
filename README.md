# HSA planar control

## Installation

1. Please follow the JAX installation instructions on [GitHub](https://github.com/google/jax).
2. Then, you can install the requirements with `pip install -r requirements.txt`. 
3. Finally, if you have locally cloned the `jax-soft-robot-modelling` repository, you can install it with `pip install -e .` from the root directory of the repository.

## ROS2

This repository also contains various ROS2 nodes. 
As usual, clone this repository into your ROS2 workspace and build it with `colcon build`.
You can launch the nodes with `ros2 launch hsa_planar_control ./launch/hsa_planar_cl_control.py`.
