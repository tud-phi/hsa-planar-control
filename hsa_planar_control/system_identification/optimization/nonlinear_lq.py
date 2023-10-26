import dill
from functools import partial
from jax import Array, debug, jit, vmap
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt
from jsrm.systems.utils import substitute_params_into_single_symbolic_expression
from os import PathLike
import sympy as sp
from typing import Callable, Dict, List, Tuple


def nonlinear_lq_optim_problem_factory(
    dynamical_matrices_fn: Callable,
    sys_helpers: Dict,
    known_params: Dict[str, Array],
    params_to_be_idd_names: List[str],
    mode: str = "dynamic",
) -> Callable:
    """
    Factory function for designing the least-squares optimization problem for the system identification.
    Arguments:
        dynamical_matrices_fn: function for evaluating the dynamical matrices
        sys_helpers: dictionary with helper entries for the HSA system
        known_params: dictionary with known robot parameters
        params_to_be_idd_names: list with the names of the parameters to be identified. Needs to match
            the keys in the params_syms dictionary in the saved symbolic expressions.
        mode: "dynamic" or "static". Default: "dynamic".
    Returns:
        residual_fn: function for evaluating the residual of the optimization problem

    """
    # these are just dummy values
    # they will not be used during the actual identification procedure
    # they are just used to determine the data shapes and structure
    ones_rod = jnp.ones_like(known_params["h"])
    dummy_to_be_identified_params = {
        "kappa_b_eq": 0 * ones_rod,  # bending rest strains of each rod
        "sigma_sh_eq": 0 * ones_rod,  # shear rest strains of each rod
        "sigma_a_eq": 0 * ones_rod,  # axial rest strains of each rod
        # scale factor for the rest length as a function of the twist strain [1/(rad/m) = m / rad]
        "C_varepsilon": 0 * ones_rod,
        "S_b_hat": 0 * ones_rod,  # Nominal bending stiffness of each rod [Nm^2]
        "S_sh_hat": 0 * ones_rod,  # Nominal shear stiffness of each rod [N]
        "S_a_hat": 0 * ones_rod,  # Nominal axial stiffness of each rod [N]
        "S_b_sh": 0 * ones_rod,  # Elastic coupling between bending and shear [Nm/rad]
        # Scaling of bending stiffness with twist strain [Nm^3/rad]
        "C_S_b": 0 * ones_rod,
        "C_S_sh": 0 * ones_rod,  # Scaling of shear stiffness with twist strain [Nm/rad]
        "C_S_a": 0 * ones_rod,  # Scaling of axial stiffness with twist strain [Nm/rad]
        # damping coefficient for bending of shape (num_segments, rods_per_segment)
        "zetab": 0.0 * ones_rod,
        # damping coefficient for shear of shape (num_segments, rods_per_segment)
        "zetash": 0.0 * ones_rod,
        # damping coefficient for axial elongation of shape (num_segments, rods_per_segment)
        "zetaa": 0.0 * ones_rod,
    }
    params = dummy_to_be_identified_params | known_params

    @jit
    def eom_residual_fn(
        Pi: Array, xi: Array, xi_d: Array, xi_dd: Array, phi: Array, mpl: float = 0.0
    ) -> Array:
        """
        Args:
            Pi: array with unknown parameters
            xi: strains of the virtual backbone
            xi_d: strain velocities of the virtual backbone
            xi_dd: strain accelerations of the virtual backbone
            phi: array with motor angles
            mpl: payload mass at the bottom of the platform

        Returns:
            residual: array with the residual of the optimization problem
        """
        _params = params.copy()
        for param_idx, param_name in enumerate(params_to_be_idd_names):
            _params[param_name] = Pi[param_idx] * jnp.ones_like(
                dummy_to_be_identified_params[param_name]
            )

        # update the payload mass in the params dictionary
        _params["mpl"] = mpl

        # add a small number to the bending strain to avoid singularities
        xi_epsed = sys_helpers["apply_eps_to_bend_strains_fn"](
            xi, 1e4 * sys_helpers["eps"]
        )

        # compute the rest strain based on the current estimate of the parameters
        xi_eq = sys_helpers["rest_strains_fn"](_params)
        # compute the configuration based on the current estimate of the rest strain
        q = xi_epsed - xi_eq
        q_d = xi_d

        # evaluate the dynamical matrices
        B, C, G, K, D, alpha = dynamical_matrices_fn(_params, q, q_d, phi)

        # rhs of the equations of motion
        rhs = alpha
        # left-hand side of the equations of motion
        lhs = G + K
        if mode == "dynamic":
            lhs = lhs + B @ xi_dd + C @ xi_d + D @ xi_d

        residual = lhs - rhs
        return residual

    return eom_residual_fn


def optimize_with_nonlinear_lq(
    eom_residual_fn: Callable,
    data_ts: Dict[str, Array],
    Pi_init: Array,
) -> Array:
    """
    Optimize the parameters of the robot model using nonlinear least squares.
    """

    if "mpl_ts" in data_ts:
        mpl_ts = data_ts["mpl_ts"]
    else:
        mpl_ts = jnp.zeros_like(data_ts["t_ts"])

    @jit
    def optimality_fn(_Pi: Array) -> Array:
        """
        Args:
            _Pi: array with unknown parameters

        Returns:
            residual: array of residuals of the optimization problem
        """
        debug.print("Pi = {_Pi}", _Pi=_Pi)
        residual = vmap(eom_residual_fn, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0,)(
            _Pi,
            data_ts["xi_ts"],
            data_ts["xi_d_ts"],
            data_ts["xi_dd_ts"],
            data_ts["phi_ts"],
            mpl_ts,
        )

        residual = residual.reshape((-1,))
        return residual

    solver = LevenbergMarquardt(residual_fun=optimality_fn, jit=False)

    print("Initial optimality error:", solver.l2_optimality_error(Pi_init))

    Pi_est, info = solver.run(Pi_init)
    print("Optimization info:\n", info)

    optimality_error = solver.l2_optimality_error(Pi_est)
    print(f"Optimality error of nonlinear least squares: {optimality_error}")

    return Pi_est
