# HSA planar control

## Installation

1. Please follow the JAX installation instructions on [GitHub](https://github.com/google/jax).
2. Then, you can install the requirements with `pip install -r requirements.txt`. 
3. Finally, if you have locally cloned the `jax-soft-robot-modelling` repository, you can install it with `pip install -e .` from the root directory of the repository.

## ROS2

This repository also contains various ROS2 nodes. 
As usual, clone this repository into your ROS2 workspace and build it with `colcon build`.
You can launch the nodes with `ros2 launch hsa_planar_control ./launch/hsa_planar_cl_control.py`.

## System identification

### 1. Identify the static elongation parameters

Use the script `identify_system_params_statically_elongation.py` to identify with linear least-squares

1. The axial rest strain `sigma_a_eq` and the axial, nominal stiffness `S_a_hat` for varying payloads.
2. The rest length elongation factor `C_varepsilon` for zero payload while assuming the axial stiffness to remain constant (i.e. `C_S_a=0`).
3. The change of the axial stiffness `C_S_a` for varying payload.

### 2. Identify the static bending parameters

Use the script `identify_system_params_statically_bending.py` to identify with linear least-squares in one go
`S_b_hat`, `S_sh_hat`, `S_b_sh`, `C_S_b`, and `C_S_sh` on a dataset with varying payload for a bending staircase sequence. 
Do not forget to first identify `sigma_a_eq` for this dataset with the script `identify_system_params_statically_elongation.py`.

### 3. Identify the dynamic parameters

Use the script `identify_system_params_dynamically.py` to identify with nonlinear least-squares in one go
all the damping coefficients. **Attention:** this doesn't give good results yet.
