import dill
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, jit, random, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import (
    generate_base_params_for_fpu,
    generate_base_params_for_epu,
)
from jsrm.systems import planar_hsa
from pathlib import Path

from hsa_planar_control.simulation import simulate_steady_state

num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent
    / "symbolic_expressions"
    / f"planar_hsa_ns-{num_segments}_nrs-{num_rods_per_segment}.dill"
)

# seed random number generator
seed = 0
rng = random.PRNGKey(seed)

# set parameters
HSA_MATERIAL = "fpu"
END_EFFECTOR_ATTACHED = True
if HSA_MATERIAL == "fpu":
    params = generate_base_params_for_fpu(
        num_segments=num_segments,
        num_rods_per_segment=num_rods_per_segment,
        rod_multiplier=2,
        end_effector_attached=END_EFFECTOR_ATTACHED,
    )
elif HSA_MATERIAL == "epu":
    params = generate_base_params_for_epu(
        num_segments=num_segments,
        num_rods_per_segment=num_rods_per_segment,
        rod_multiplier=2,
        end_effector_attached=END_EFFECTOR_ATTACHED,
    )
else:
    raise ValueError(f"Unknown hsa_material: {HSA_MATERIAL}")
num_segments = params["l"].shape[0]

# slightly increase damping
params["zetab"] = 5 * params["zetab"]
params["zetash"] = 5 * params["zetash"]
params["zetaa"] = 5 * params["zetaa"]

sim_dt = 7e-5  # time step for simulation [s]
duration = 2.0  # duration of simulation [s]
q0 = jnp.zeros((3 * num_segments,))

if __name__ == "__main__":
    (
        forward_kinematics_virtual_backbone_fn,
        forward_kinematics_end_effector_fn,
        jacobian_end_effector_fn,
        inverse_kinematics_end_effector_fn,
        dynamical_matrices_fn,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    simulate_steady_state_fn = partial(
        simulate_steady_state,
        dynamical_matrices_fn,
        params,
        q0,
        sim_dt=sim_dt,
        duration=duration,
    )
    batched_simulate_steady_state_fn = jit(vmap(simulate_steady_state_fn))
    batched_forward_kinematics_end_effector_fn = jit(
        vmap(partial(forward_kinematics_end_effector_fn, params))
    )

    phi_max = params["phi_max"].flatten()

    # generate max actuation samples
    u = jnp.linspace(-phi_max.mean(), phi_max.mean(), 101)
    phi_ss = jnp.clip(
        jnp.stack([phi_max[0] + u, phi_max[1] - u], axis=1), None, phi_max
    )
    max_actuation_samples = {"phi_ss": phi_ss}
    (
        max_actuation_samples["q_ss"],
        max_actuation_samples["q_d_ss"],
    ) = batched_simulate_steady_state_fn(max_actuation_samples["phi_ss"])
    max_actuation_samples["chiee_ss"] = batched_forward_kinematics_end_effector_fn(
        max_actuation_samples["q_ss"]
    )
    if jnp.isnan(max_actuation_samples["q_ss"]).sum() > 0:
        raise ValueError("max actuation samples contain nans")
    print(
        "max actuation samples: q_d at steady state averaged over samples",
        max_actuation_samples["q_d_ss"].mean(axis=0),
    )

    # generate min actuation samples
    u = jnp.linspace(-phi_max.mean(), phi_max.mean(), 101)
    phi_ss = jnp.clip(jnp.stack([u, -u], axis=1), 0.0, None)
    min_actuation_samples = {"phi_ss": phi_ss}
    (
        min_actuation_samples["q_ss"],
        min_actuation_samples["q_d_ss"],
    ) = batched_simulate_steady_state_fn(min_actuation_samples["phi_ss"])
    min_actuation_samples["chiee_ss"] = batched_forward_kinematics_end_effector_fn(
        min_actuation_samples["q_ss"]
    )
    if jnp.isnan(min_actuation_samples["q_ss"]).sum() > 0:
        raise ValueError("min actuation samples contain nans")
    print(
        "min actuation samples: q_d at steady state averaged over samples",
        max_actuation_samples["q_d_ss"].mean(axis=0),
    )

    # generate random samples for plotting colors of axial strain
    rng, sample_key = random.split(rng)
    phi_ss = random.uniform(
        sample_key,
        (5000, phi_max.shape[0]),
        minval=0.0,
        maxval=phi_max,
    )
    random_samples = {"phi_ss": phi_ss}
    random_samples["q_ss"], random_samples["q_d_ss"] = batched_simulate_steady_state_fn(
        random_samples["phi_ss"],
    )
    random_samples["chiee_ss"] = batched_forward_kinematics_end_effector_fn(
        random_samples["q_ss"]
    )

    operational_workspace_samples = {
        "max_actuation": max_actuation_samples,
        "min_actuation": min_actuation_samples,
        "random": random_samples,
    }

    data_folder = Path(__file__).parent.parent.parent / "data" / "kinematics"
    data_folder.mkdir(parents=True, exist_ok=True)
    if END_EFFECTOR_ATTACHED:
        sample_filepath = data_folder / f"operational_workspace_{HSA_MATERIAL}_ee.dill"
    else:
        sample_filepath = data_folder / f"operational_workspace_{HSA_MATERIAL}.dill"
    with open(str(sample_filepath), "wb") as f:
        dill.dump(operational_workspace_samples, f)
