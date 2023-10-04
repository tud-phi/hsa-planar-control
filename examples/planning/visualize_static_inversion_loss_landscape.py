from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from functools import partial
from jax import Array, jit, random, vmap
from jax import numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL, PARAMS_EPU_CONTROL
from jsrm.systems import planar_hsa
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Callable, Dict, Tuple

from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

pee_des = jnp.array([0.0195418, 0.13252665])
hsa_material = "epu"

# set parameters
if hsa_material == "fpu":
    params = PARAMS_FPU_CONTROL.copy()
elif hsa_material == "epu":
    params = PARAMS_EPU_CONTROL.copy()
else:
    raise ValueError(f"Unknown hsa_material: {hsa_material}")


def main():
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    residual_fn = jit(
        static_inversion_factory(
            params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
        )
    )
    batched_residual_fn = jit(vmap(residual_fn))


if __name__ == "__main__":
    main()
