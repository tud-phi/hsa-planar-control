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
import plotly.graph_objects as go
import time
from typing import Callable, Dict, Tuple

from hsa_planar_control.planning.static_planning import (
    static_inversion_factory,
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
# these are the values we should get for the setpoint pee_des = (0.0195418, 0.13252665) and the EPU material
thee_sol = jnp.array(-0.28864607)  # [rad]
# q_ss_sol = (-4.89230524e+00, 2.28753141e-03, 4.20002605e-01)
phi_ss_sol = jnp.array([3.96219014, 2.11047887])

hsa_material = "epu"
# set parameters
if hsa_material == "fpu":
    params = PARAMS_FPU_CONTROL.copy()
elif hsa_material == "epu":
    params = PARAMS_EPU_CONTROL.copy()
else:
    raise ValueError(f"Unknown hsa_material: {hsa_material}")

phi_range = (-jnp.pi, 2 * jnp.pi)
th_range = (-jnp.pi / 4, jnp.pi / 4)


def main():
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    residual_fn = partial(
        static_inversion_factory(
            params, inverse_kinematics_end_effector_fn, dynamical_matrices_fn
        ),
        pee_des=pee_des,
    )
    batched_residual_fn = jit(vmap(residual_fn))

    # residual at the solution
    res_sol = residual_fn(jnp.array([thee_sol, *phi_ss_sol]))
    print("Residual at the solution:", res_sol, "norm:", jnp.linalg.norm(res_sol))

    th_ss = jnp.linspace(th_range[0], th_range[1], 25)
    phi1_ss = jnp.linspace(phi_range[0], phi_range[1], 25)
    phi2_ss = jnp.linspace(phi_range[0], phi_range[1], 25)
    th_mesh, phi1_mesh, phi2_mesh = jnp.meshgrid(th_ss, phi1_ss, phi2_ss)
    thv, phi1v, phi2v = th_mesh.flatten(), phi1_mesh.flatten(), phi2_mesh.flatten()
    xv = jnp.stack([thv, phi1v, phi2v], axis=-1)

    residualv = batched_residual_fn(xv)
    residual_normv = jnp.linalg.norm(residualv, axis=-1)

    fig = go.Figure(data=go.Volume(
        x=phi1v,
        y=phi2v,
        z=thv,
        value=residual_normv,
        isomax=1.0,
        opacity=0.5,  # needs to be small to see through all surfaces
        surface_count=21,  # needs to be a large number for good volume rendering
        surface=dict(fill=0.7, pattern='odd'),
        caps=dict(x_show=False, y_show=False, z_show=False),  # no caps
        slices_z=dict(show=True, locations=[thee_sol]),
    ))
    fig.show()


if __name__ == "__main__":
    main()
