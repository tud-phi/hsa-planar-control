import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
from jax import Array, debug, jit, random, vmap
import jax.numpy as jnp
import jsrm
from jsrm.parameters.hsa_params import PARAMS_FPU_CONTROL
from jsrm.systems import planar_hsa
from jsrm.systems.utils import substitute_params_into_single_symbolic_expression
from pathlib import Path
import sympy as sp

from hsa_planar_control.collocated_form import (
    check_integrability_assumption,
    symbolically_derive_collocated_form,
    mapping_into_collocated_form_factory,
)


num_segments = 1
num_rods_per_segment = 2

# filepath to symbolic expressions
sym_exp_filepath = (
    Path(jsrm.__file__).parent / "symbolic_expressions" / "planar_hsa_ns-1_nrs-2.dill"
)


def test_collocated_form(seed: int = 0):
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    assert check_integrability_assumption(sym_exp_filepath)

    (
        _,
        _,
        _,
        _,
        _,
        sys_helpers,
    ) = planar_hsa.factory(sym_exp_filepath)
    map_into_collocated_form_fn, collocated_exps = mapping_into_collocated_form_factory(
        sym_exp_filepath, sys_helpers=sys_helpers, run_tests=True
    )

    phi_exp = sp.Matrix(sym_exps["state_syms"]["phi"])
    A_exp = sym_exps["exps"]["alpha"].jacobian(phi_exp)

    # subsitute in the parameters
    A_exp = substitute_params_into_single_symbolic_expression(
        A_exp, sym_exps["params_syms"], PARAMS_FPU_CONTROL
    )

    A_lambda = sp.lambdify(
        sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
        A_exp,
        "jax",
    )

    @jit
    def compute_A_varphi(q: Array, phi: Array) -> Array:
        # map the configuration to the strains
        xi = sys_helpers["configuration_to_strains_fn"](PARAMS_FPU_CONTROL, q)

        varphi, Jh = map_into_collocated_form_fn(PARAMS_FPU_CONTROL, q, phi)

        # compute the A matrix in q-space
        A_q = A_lambda(*xi, *phi)

        A_varphi = jnp.linalg.inv(Jh).T @ A_q
        return A_varphi

    A_varphi_target = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    for i in range(30):
        rng = random.PRNGKey(seed)
        rng, subrng1, subrng2, subrng3, subrng4, subrng5 = random.split(rng, 6)
        kappa_b = random.uniform(
            subrng1,
            (num_segments,),
            minval=-jnp.pi / jnp.mean(PARAMS_FPU_CONTROL["l"]),
            maxval=jnp.pi / jnp.mean(PARAMS_FPU_CONTROL["l"]),
        )
        sigma_sh = random.uniform(subrng2, (num_segments,), minval=-0.2, maxval=0.2)
        sigma_a = random.uniform(subrng3, (num_segments,), minval=0.0, maxval=0.5)
        q = jnp.concatenate((kappa_b, sigma_sh, sigma_a), axis=0)
        phi = PARAMS_FPU_CONTROL["h"].flatten() * random.uniform(
            subrng5, PARAMS_FPU_CONTROL["h"].flatten().shape, minval=0.0, maxval=jnp.pi
        )

        A_varphi = compute_A_varphi(q, phi)
        print("A_varphi:\n", A_varphi)

        assert jnp.allclose(
            A_varphi, A_varphi_target, atol=1e-6
        ), f"A_varphi is wrong for configuration q: {q}, phi: {phi}."

        return True


if __name__ == "__main__":
    test_collocated_form()
    print("All tests passed!")
