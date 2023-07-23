import dill
from functools import partial
from jax import Array, debug, jit, vmap
import jax.numpy as jnp
from jsrm.systems.utils import substitute_params_into_single_symbolic_expression
from os import PathLike
import sympy as sp
from typing import Callable, Dict, List, Tuple
import warnings

from .utils import isolateVariablesToLeftHandSide


def linear_lq_optim_problem_factory(
    sym_exp_filepath: PathLike,
    dynamical_matrices_fn: Callable,
    sys_helpers: Dict,
    known_params: Dict[str, Array],
    params_to_be_idd_names: List[str],
    mode: str = "dynamic",
) -> Tuple[List[sp.Symbol], Callable, Callable]:
    """
    Factory function for designing the least-squares optimization problem for the system identification.
    Arguments:
        sym_exp_filepath: path to the file with saved symbolic expressions
        dynamical_matrices_fn: function for evaluating the dynamical matrices
        sys_helpers: dictionary with helper entries for the HSA system
        known_params: dictionary with known robot parameters
        params_to_be_idd_names: list with the names of the parameters to be identified. Needs to match
            the keys in the params_syms dictionary in the saved symbolic expressions.
        mode: "dynamic" or "static". Default: "dynamic".
    Returns:
        Pi_syms: list with symbols for the parameters to be identified
        cal_a_fn: function for evaluating the unknown parts of the equations of motion to be multiplied by the
            parameters to be identified
        cal_b_fn: function for evaluating the known parts of the equations of motion

    where cal_A @ Pi = cal_B
    """
    # load saved symbolic data
    sym_exps = dill.load(open(str(sym_exp_filepath), "rb"))

    state_syms = sym_exps["state_syms"]
    exps = sym_exps["exps"]
    xi_d = sp.Matrix(state_syms["xi_d"])
    xi_dd = sp.Matrix(state_syms["xi_dd"])

    rhs = exps["alpha"]
    lhs = exps["G"] + exps["K"]
    if mode == "dynamic":
        lhs = lhs + exps["B"] @ xi_dd + exps["C"] @ xi_d + exps["D"] @ xi_d

    # substitute the parameters of each rod with jointly-shared parameters
    # for example S_a_hat1 to S_a_hat4 are all replaced with S_a_hat
    Pi_syms = []
    for param_name in params_to_be_idd_names:
        shared_param = sp.Symbol(param_name)
        for rod_param in sym_exps["params_syms"][param_name]:
            lhs = lhs.subs(rod_param, shared_param)
            rhs = rhs.subs(rod_param, shared_param)
        Pi_syms.append(shared_param)

    # substitute the parameters with the known values
    lhs = substitute_params_into_single_symbolic_expression(
        lhs, sym_exps["params_syms"], known_params
    )
    rhs = substitute_params_into_single_symbolic_expression(
        rhs, sym_exps["params_syms"], known_params
    )

    # apply tricks where we find a multiplication of two parameters
    for param_idx, (param_name, param_sym) in enumerate(zip(params_to_be_idd_names, Pi_syms)):
        if param_name == "sigma_a_eq" and "S_a_hat" in params_to_be_idd_names:
            # we have a multiplication of S_a_hat with sigma_a_eq
            # therefore, we substitute sigma_a_eq with sigma_a_eq_times_S_a_hat / S_a_hat
            new_param_sym = sp.Symbol("S_a_hat_times_sigma_a_eq")
            lhs = lhs.subs(param_sym, new_param_sym / sp.Symbol("S_a_hat"))
            rhs = rhs.subs(param_sym, new_param_sym / sp.Symbol("S_a_hat"))
            warnings.warn("Substituted sigma_a_eq with S_a_hat_times_sigma_a_eq / S_a_hat")
        else:
            continue
        Pi_syms[param_idx] = new_param_sym\

    # define equation
    eq = sp.Eq(lhs, rhs)

    # isolate the parameters to be identified to the left-hand side
    eq = isolateVariablesToLeftHandSide(eq, Pi_syms)
    # the parameters are now isolated in lhs
    lhs, rhs = eq.args[0], eq.args[1]

    # symbolic expression
    cal_a = lhs.jacobian(Pi_syms)

    cal_a_lambda = sp.lambdify(
        (state_syms["xi"] + state_syms["xi_d"] + state_syms["phi"]),
        cal_a,
        "jax",
    )
    cal_b_lambda = sp.lambdify(
        (
            state_syms["xi"]
            + state_syms["xi_d"]
            + state_syms["xi_dd"]
            + state_syms["phi"]
        ),
        rhs,
        "jax",
    )

    @jit
    def cal_a_fn(xi: Array, xi_d: Array, phi: Array) -> Array:
        cal_a = cal_a_lambda(*xi, *xi_d, *phi)

        return cal_a

    # symbolic position of the payload mass
    # we assume that payload mass is attached to bottom end of the platform
    # orientation of end-effector with respect to the base
    thee = exps["chiee"][2, 0]
    R_exp = sp.Matrix([[sp.cos(thee), -sp.sin(thee)], [sp.sin(thee), sp.cos(thee)]])
    # distance along local y-axis from the end-effector to the payload CoM
    dy_pl_ee = known_params["pcudim"][-1, 1] + known_params.get("lpl", 0.0) / 2
    # subtract distance from the end-effector position
    ppl_exp = exps["chiee"][:2, 0] - R_exp @ sp.Matrix([0, dy_pl_ee.item()])
    # positional Jacobian of the payload mass
    Jpl_exp = ppl_exp.jacobian(state_syms["xi"])
    # substitute the parameters
    Jpl_exp = substitute_params_into_single_symbolic_expression(
        Jpl_exp, sym_exps["params_syms"], known_params
    )
    Jpl_lambda = sp.lambdify(
        sym_exps["state_syms"]["xi"],
        Jpl_exp,
        "jax",
    )

    @jit
    def cal_b_fn(
        xi: Array, xi_d: Array, xi_dd: Array, phi: Array, mpl: float = 0.0
    ) -> Array:
        """
        Evaluate the right side of the system identification problem
        Arguments:
        q: strains of the virtual backbone
        q_d: strain velocities of the virtual backbone
        q_dd: strain accelerations of the virtual backbone
        phi: platform orientation
        mpl: payload mass

        Returns:
            cal_b: right side of the system identification problem
        """
        # add a small number to the bending strain to avoid singularities
        xi_epsed = sys_helpers["apply_eps_to_bend_strains_fn"](
            xi, 1e4 * sys_helpers["eps"]
        )

        # evaluate the expression for the right side of the system identification problem
        cal_b = cal_b_lambda(*xi_epsed, *xi_d, *xi_dd, *phi).squeeze()

        # evaluate the positional Jacobian of the payload mass
        Jpl = Jpl_lambda(*xi_epsed).squeeze()
        # define gravity vector for payload mass
        Gpl = -mpl * Jpl.T @ known_params["g"]
        # add the gravity vector to the right side of the system identification problem
        cal_b = cal_b - Gpl

        return cal_b

    return Pi_syms, cal_a_fn, cal_b_fn


def optimize_with_closed_form_linear_lq(
    cal_a_fn: Callable,
    cal_b_fn: Callable,
    data_ts: Dict[str, Array],
) -> Array:
    """
    Optimize the parameters of the robot model using the closed-form least squares solution.
    I.e. apply the Moore-Penrose pseudoinverse to the linearized system of equations.
    """

    if "mpl_ts" in data_ts:
        mpl_ts = data_ts["mpl_ts"]
    else:
        mpl_ts = jnp.zeros_like(data_ts["t_ts"])

    cal_A = vmap(cal_a_fn, in_axes=(0, 0, 0), out_axes=0)(
        data_ts["xi_ts"], data_ts["xi_d_ts"], data_ts["phi_ts"]
    )

    # investigate the rank of the cal_A matrices
    # if the column rank is not full, the system parameters are not identifiable
    ranks_cal_a = vmap(
        lambda _cal_a: jnp.linalg.matrix_rank(_cal_a), in_axes=0, out_axes=0
    )(cal_A)
    print(
        f"Checking the rank of the cal_a matrices, each of shape {cal_A.shape[1:]}: "
        f"min = {ranks_cal_a.min()}, max = {ranks_cal_a.max()}, mean = {ranks_cal_a.mean()}"
    )

    cal_A = cal_A.reshape(-1, cal_A.shape[-1])
    print(
        f"The entire cal_A matrix of shape {cal_A.shape} has rank: {jnp.linalg.matrix_rank(cal_A)}"
    )

    cal_B = vmap(
        cal_b_fn,
        in_axes=(0, 0, 0, 0),
        axis_name="mpl",
        out_axes=0,
    )(
        data_ts["xi_ts"],
        data_ts["xi_d_ts"],
        data_ts["xi_dd_ts"],
        data_ts["phi_ts"],
        mpl=mpl_ts,
    )
    cal_B = cal_B.reshape(-1)

    cal_A_pinv = jnp.linalg.pinv(cal_A)
    Pi_est = cal_A_pinv @ cal_B  # Least-squares estimate of the parameters
    print("Pi_est", Pi_est)

    return Pi_est
