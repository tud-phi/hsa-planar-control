# HSA planar control

This repository provides implementation and closed-loop simulation of various control strategies for planar HSA robots.
Furthermore, it contains ROS2 nodes for planning and control of the planar HSA robots.

## Citation

This simulator is part of the publication **An Experimental Study of Model-based Control
for Planar Handed Shearing Auxetics Robots** presented at the _18th International Symposium on Experimental Robotics_. 
You can find the publication ~~online~~ in the Springer Proceedings on Advanced Robotics (SPAR).

Please use the following citation if you use our software in your (scientific) work:

```bibtex
@inproceedings{stolzle2023experimental,
  title={An Experimental Study of Model-based Control for Planar Handed Shearing Auxetics Robots},
  author={St{\"o}lzle, Maximilian and Rus, Daniela and Della Santina, Cosimo},
  booktitle={Experimental Robotics: The 18th International Symposium},
  year={2023},
  organization={Springer}
}
```

## Installation

1. Please follow the JAX installation instructions on [GitHub](https://github.com/google/jax).
2. Then, you can install the requirements with `pip install -r requirements.txt`. 
3. Finally, if you have locally cloned the `jax-soft-robot-modelling` repository, you can install it with `pip install -e .` from the root directory of the repository.

## ROS2

This repository also contains various ROS2 nodes for planning and control. 
As usual, clone this repository into your ROS2 workspace and build it with `colcon build`.
Furthermore, we rely on the ROS2 packages in the [ros2-hsa](https://github.com/tud-cor-sr/ros2-hsa) repository for the communication with the hardware (both actuation and motion capture), inverse kinematics and visualization.
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
