import dill
from jax import Array
import jax.numpy as jnp
from os import PathLike
import sympy as sp
from typing import Callable, Dict, Tuple

from jsrm.systems.utils import concatenate_params_syms


def check_integrability_assumption(sym_exp_filepath: PathLike) -> bool:
    """
    Check the integrability assumption for bringing the system into collocated form.
    Equation (4) of Pustina, Pietro, et al. "On the Collocated Form with Input Decoupling of Lagrangian Systems."
    arXiv preprint arXiv:2306.07258 (2023).
    Args:
        sym_exp_filepath: Path to the saved symbolic expressions.

    Returns:

    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    xi = sp.Matrix(sym_exps["state_syms"]["xi"])
    phi = sp.Matrix(sym_exps["state_syms"]["phi"])
    alpha = sym_exps["exps"]["alpha"]
    A = alpha.jacobian(phi)

    print("A =\n", A)

    for i in range(A.shape[1]):
        # iterate over columns
        for j in range(A.shape[0]):
            for k in range(A.shape[0]):
                lhs = A[j, i].diff(xi[k])
                rhs = A[k, i].diff(xi[j])
                if lhs != rhs:
                    print("Not integrable")
                    print("i = ", i)
                    print("j = ", j)
                    print("k = ", k)
                    print("lhs = ", lhs)
                    print("rhs = ", rhs)
                    return False

    print(
        "The integrability assumption is satisfied and we can bring the sytem into collocated form."
    )
    return True


def symbolically_derive_collocated_form(sym_exp_filepath: PathLike):
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    xi = sp.Matrix(sym_exps["state_syms"]["xi"])
    xi_d = sp.Matrix(sym_exps["state_syms"]["xi_d"])
    phi = sp.Matrix(sym_exps["state_syms"]["phi"])

    # derivation linearized actuation matrix
    alpha = sym_exps["exps"]["alpha"]
    A = alpha.jacobian(phi)

    # derive collocated form
    dy = A.T @ xi_d

    print("dy[0] =\n", dy[0])
    print("dy[1] =\n", dy[1])

    # define symbolic functions
    t = sp.Symbol("t", real=True, nonnegative=True)
    # xi_fn = sp.Function("xi")(t)
    # xi_fn = list(sp.symbols(f"xi_fn1:{xi.shape[0] + 1}", cls=sp.Function))
    xi_fn = [sp.Function(f"xi_fn{i + 1}", real=True)(t) for i in range(len(xi))]
    xi_d_fn = [xi_fn[i].diff(t) for i in range(len(xi_fn))]

    # replace the variables with the symbolic functions
    # dy_fn = dy.subs(sym_exps["state_syms"]["xi"], xi_fn)
    dy_fn = dy.copy()
    for i in range(len(xi_fn)):
        dy_fn = dy_fn.subs(sym_exps["state_syms"]["xi"][i], xi_fn[i])
        dy_fn = dy_fn.subs(sym_exps["state_syms"]["xi_d"][i], xi_d_fn[i])

    # replace the time variable for integration
    tau = sp.Symbol("tau", real=True, nonnegative=True)
    dy_fn = dy_fn.subs(t, tau)
    # perform integration
    # y_fn = sp.integrate(dy_fn, (tau, 0.0, t))
    y_fn = sp.integrate(dy_fn, tau)

    # replace the symbolic functions with the variables
    y = y_fn.copy()
    for i in range(len(xi_fn)):
        y = y.subs(xi_fn[i], sym_exps["state_syms"]["xi"][i])
        y = y.subs(xi_d_fn[i], sym_exps["state_syms"]["xi_d"][i])
    y = sp.simplify(y)
    print("y[0] =\n", y[0])
    print("y[1] =\n", y[1])


def mapping_into_collocated_form_factory(
    sym_exp_filepath: PathLike, sys_helpers: Dict, run_tests: bool = False
) -> Tuple[Callable, Dict[str, sp.Expr]]:
    """
    Create a function provides the mapping from the original system into the collocated form.
    Args:
        sym_exp_filepath: Path to the saved symbolic expressions.
        sys_helpers: Dictionary of helper functions for the system.
        run_tests: Whether to run tests to check the correctness of the mapping.
    Returns:
        map_into_collocated_form_fn: Function that maps the original system into the collocated form.
        collocated_map_exps: Dictionary of symbolic expressions for the mapping into the collocated variables.
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    assert (
        len(sym_exps["state_syms"]["phi"]) == 2
    ), "Mapping into collocated form was only implemented for two rods."

    xi_exp = sp.Matrix(sym_exps["state_syms"]["xi"])
    xi1, xi2, xi3 = tuple(sym_exps["state_syms"]["xi"])
    xi_d1, xi_d2, xi_d3 = tuple(sym_exps["state_syms"]["xi_d"])
    phi_exp = sp.Matrix(sym_exps["state_syms"]["phi"])
    phi1, phi2 = tuple(sym_exps["state_syms"]["phi"])

    l1 = sym_exps["params_syms"]["l"][0]
    sigma_a_eq1, sigma_a_eq2 = tuple(sym_exps["params_syms"]["sigma_a_eq"])
    C_varepsilon1, C_varepsilon2 = tuple(sym_exps["params_syms"]["C_varepsilon"])
    rout1, rout2 = tuple(sym_exps["params_syms"]["rout"])
    rin1, rin2 = tuple(sym_exps["params_syms"]["rin"])
    h1, h2 = tuple(sym_exps["params_syms"]["h"])
    roff1, roff2 = tuple(sym_exps["params_syms"]["roff"])
    S_b_hat1, S_b_hat2 = tuple(sym_exps["params_syms"]["S_b_hat"])
    S_sh_hat1, S_sh_hat2 = tuple(sym_exps["params_syms"]["S_sh_hat"])
    S_a_hat1, S_a_hat2 = tuple(sym_exps["params_syms"]["S_a_hat"])
    C_S_b1, C_S_b2 = tuple(sym_exps["params_syms"]["C_S_b"])
    C_S_sh1, C_S_sh2 = tuple(sym_exps["params_syms"]["C_S_sh"])
    C_S_a1, C_S_a2 = tuple(sym_exps["params_syms"]["C_S_a"])

    A_exp = sym_exps["exps"]["alpha"].jacobian(phi_exp)

    varphi_exp = sp.Matrix(
        [
            [
                h1
                * (
                    2 * C_S_a1 * C_varepsilon1 * h1 * phi1 * roff1 * xi1
                    + 2 * C_S_a1 * C_varepsilon1 * h1 * phi1 * xi3
                    - C_S_a1 * l1 * roff1**2 * xi1**2 / 2
                    + C_S_a1 * l1 * roff1 * sigma_a_eq1 * xi1
                    - C_S_a1 * l1 * roff1 * xi1 * xi3
                    + C_S_a1 * l1 * sigma_a_eq1 * xi3
                    - C_S_a1 * l1 * xi3**2 / 2
                    - C_S_b1 * l1 * xi1**2 / 2
                    - C_S_sh1 * l1 * xi2**2 / 2
                    + C_varepsilon1 * S_a_hat1 * l1 * roff1 * xi1
                    + C_varepsilon1 * S_a_hat1 * l1 * xi3
                )
                / l1**2
            ],
            [
                h2
                * (
                    2 * C_S_a2 * C_varepsilon2 * h2 * phi2 * roff2 * xi1
                    + 2 * C_S_a2 * C_varepsilon2 * h2 * phi2 * xi3
                    - C_S_a2 * l1 * roff2**2 * xi1**2 / 2
                    + C_S_a2 * l1 * roff2 * sigma_a_eq2 * xi1
                    - C_S_a2 * l1 * roff2 * xi1 * xi3
                    + C_S_a2 * l1 * sigma_a_eq2 * xi3
                    - C_S_a2 * l1 * xi3**2 / 2
                    - C_S_b2 * l1 * xi1**2 / 2
                    - C_S_sh2 * l1 * xi2**2 / 2
                    + C_varepsilon2 * S_a_hat2 * l1 * roff2 * xi1
                    + C_varepsilon2 * S_a_hat2 * l1 * xi3
                )
                / l1**2
            ],
            [xi2],  # use the shear strain as the unactuated variable
        ]
    )

    if run_tests:
        print(
            "Checking that the derived Jacobian of the h(q) corresponds to A.T for the actuated variables."
        )
        Jh_derived = varphi_exp.jacobian(xi_exp)
        Jh_sol = A_exp.T.row_insert(2, sp.Matrix([[0, 1, 0]]))
        assert Jh_derived.shape == Jh_sol.shape, "Jacobian of mapping has wrong shape."
        print("Jvarphi_derived =\n", Jh_derived)
        print("Jvarphi_sol =\n", Jh_sol)
        for i in range(Jh_sol.shape[0]):
            for j in range(Jh_sol.shape[1]):
                diff_ij = sp.simplify(Jh_derived[i, j] - Jh_sol[i, j])
                print(f"diff at i={i}, j={j}:\n", diff_ij)
                assert (
                    diff_ij == 0
                ), f"Jacobian of mapping is wrong at index i={i}, j={j}."

    # Jacobian of the mapping
    Jh_exp = varphi_exp.jacobian(xi_exp)

    # concatenate the robot params symbols
    params_syms_cat = concatenate_params_syms(sym_exps["params_syms"])

    h_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
        varphi_exp,
        "jax",
    )
    Jh_lambda = sp.lambdify(
        params_syms_cat + sym_exps["state_syms"]["xi"] + sym_exps["state_syms"]["phi"],
        Jh_exp,
        "jax",
    )

    def map_into_collocated_form_fn(
        params: Dict[str, Array], q: Array, phi: Array
    ) -> Tuple[Array, Array]:
        # map the configuration to the strains
        xi = sys_helpers["configuration_to_strains_fn"](params, q)

        # convert the dictionary of parameters to a list, which we can pass to the lambda function
        params_for_lambdify = sys_helpers["select_params_for_lambdify_fn"](params)

        varphi = h_lambda(*params_for_lambdify, *xi, *phi).squeeze()
        Jh = Jh_lambda(*params_for_lambdify, *xi, *phi)

        return varphi, Jh

    collocated_map_exps = {
        "varphi": varphi_exp,
        "Jh": Jh_exp,
    }

    return map_into_collocated_form_fn, collocated_map_exps
