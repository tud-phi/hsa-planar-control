# HSA planar control

This repository provides implementation and closed-loop simulation of various control strategies for planar HSA robots.
Furthermore, it contains ROS2 nodes for planning and control of the planar HSA robots.

[![An Experimental Study of Model-based Control for Planar Handed Shearing Auxetics Robots - Video](https://img.youtube.com/vi/7PgKnE_MOsY/0.jpg)](https://www.youtube.com/watch?v=7PgKnE_MOsY)

**Abstract:**
Parallel robots based on Handed Shearing Auxetics (HSAs) can implement complex motions using standard electric motors while maintaining the complete softness of the structure, thanks to specifically designed architected metamaterials.
However, their control is especially challenging due to varying and coupled stiffness, shearing, non-affine terms in the actuation model, and underactuation. In this paper, we present a model-based control strategy for planar HSA robots enabling regulation in task space. We formulate equations of motion, show that they admit a collocated form, and design a P-satI-D feedback controller with compensation for elastic and gravitational forces. We experimentally identify and verify the proposed control strategy in closed loop.

<img src="assets/20231019_081703_overlayed_300x.gif" width="250"/>
<img src="assets/20231019_083240_overlayed_600x.gif" width="250"/>

## Citation

This simulator is part of the publication **An Experimental Study of Model-based Control
for Planar Handed Shearing Auxetics Robots** presented at the _18th International Symposium on Experimental Robotics_. 
You can find the publication online in the Springer Proceedings on Advanced Robotics (SPAR): https://doi.org/10.1007/978-3-031-63596-0_14

Please use the following citation if you use our software in your (scientific) work:

```bibtex
@inproceedings{stolzle2023experimental,
  title={An experimental study of model-based control for planar handed shearing auxetics robots},
  author={St{\"o}lzle, Maximilian and Rus, Daniela and Della Santina, Cosimo},
  booktitle={International Symposium on Experimental Robotics},
  pages={153--167},
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
As usual, clone this repository into your ROS2 workspace and build it with `colon build.`
Furthermore, we rely on the ROS2 packages in the [ros2-hsa](https://github.com/tud-phi/ros2-hsa) repository for the communication with the hardware (both actuation and motion capture), inverse kinematics and visualization.
You can launch the nodes with `ros2 launch hsa_planar_control ./launch/hsa_planar_cl_control.py`.

## System identification

You can use the provided script `examples/system_identification/epu_identify_system_params_statically` to identify the nominal stiffness `S_a_hat`, the rest length elongation factor `C_varepsilon`, change of the axial stiffness `C_S_a`, and the bending parameters `S_b_hat`, `S_sh_hat`, `S_b_sh`, `C_S_b`, and `C_S_sh` using linear least-squares. Please modify the `SYSTTEMID_STEP` variable accordingly.

## See also

You might also be interested in the following repositories:
 - The [`jax-soft-robot-modelling`](https://github.com/tud-phi/jax-soft-robot-modelling) repository contains a JAX implementation 
 of various soft robot models, which can be, for example, used to simulate the robot's forward dynamics.
 - The [`jax-spcs-kinematics`](https://github.com/tud-phi/jax-spcs-kinematics) repository contains an implementation
 of the Selective Piecewise Constant Strain (SPCS) kinematics in JAX. Our paper shows that this kinematic 
model is suitable for representing the shape of HSA rods.
 - The [`HSA-PyElastica`](https://github.com/tud-phi/HSA-PyElastica) repository contains a plugin for PyElastica
for the simulation of HSA robots.
