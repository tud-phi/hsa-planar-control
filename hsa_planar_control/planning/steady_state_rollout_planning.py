import diffrax as dfx
from functools import partial
from jax import Array, jacfwd, jacrev, jit, random, vmap
import jax.numpy as jnp
import jaxopt as jo
import optimistix as optx
from scipy.optimize import least_squares
from typing import Callable, Dict, Tuple

from hsa_planar_control.simulation import simulate_steady_state


def steady_state_rollout_planning_factory(
    params: Dict[str, Array],
    forward_kinematics_end_effector_fn: Callable,
    dynamical_matrices_fn: Callable,
    sim_dt: float = 5e-4,
    duration: float = 5.0,
    ode_solver_class=dfx.Tsit5,
) -> Tuple[Callable, Callable, Callable]:
    """
    Factory function for planning with steady-state rollout.
    Args:
        params: a dictionary of robot parameters
        forward_kinematics_end_effector_fn: a function that computes the end effector pose from the configuration
        dynamical_matrices_fn: a function that computes the dynamical matrices
        sim_dt: the time step for the simulation
        duration: the duration of the simulation (i.e. the time we expect the system to take to reach steady-state)
        ode_solver_class: ODE solver class
    Returns:
        rollout_fn: Callable that returns the steady-state end effector pose, configuration, and configuration velocity
        residual_fn: Callable that returns the residual vector given the steady-state actuation.
        jac_residual_fn: Callable that returns the Jacobian of the residual vector with respect to the steady-state actuation.
    """
    @jit
    def rollout_fn(phi_ss: Array, q0: Array):
        q_ss, q_d_ss = simulate_steady_state(
            dynamical_matrices_fn=dynamical_matrices_fn,
            params=params,
            q0=q0,
            phi_ss=phi_ss,
            sim_dt=sim_dt,
            duration=duration,
            ode_solver_class=ode_solver_class,
            allow_forward_autodiff=True
        )

        chiee_ss = forward_kinematics_end_effector_fn(params, q_ss)

        return chiee_ss, q_ss, q_d_ss

    @jit
    def residual_fn(phi_ss: Array, *args, pee_des: Array, q0: Array) -> Array:
        chiee_ss, q_ss, q_d_ss = rollout_fn(phi_ss, q0=q0)

        residual = chiee_ss[:2] - pee_des
        return residual

    jac_residual_fn = jit(jacfwd(residual_fn))

    return rollout_fn, residual_fn, jac_residual_fn


def plan_with_rollout_to_steady_state(
        params: Dict[str, Array],
        rollout_fn: Callable,
        residual_fn: Callable,
        jac_residual_fn: Callable,
        pee_des: Array,
        q0: Array,
        phi0: Array,
        solver_type="scipy_least_squares",
        verbose: bool = False
) -> Tuple[Array, Array, Array, Array]:
    """
    Plan the steady-state actuation and configuration for a given desired end effector position.
    Args:
        params: a dictionary of robot parameters
        rollout_fn: Callable that returns the steady-state end effector pose, configuration, and configuration velocity
        residual_fn: Callable that returns the residual vector given the end-effector orientation and the motor positions.
        jac_residual_fn: Callable that returns the Jacobian of the residual vector with respect to the steady-state actuation.
        pee_des: the desired end effector position
        q0: the initial configuration used for the rollout
        phi0: the initial guess for the steady-state actuation
        solver_type: the type of solver to use for solving the nonlinear least squares problem
        verbose: whether to print the result of the optimization
    Returns:
        chiee_ss: the steady-state end effector pose
        q_ss: the steady-state configuration
        phi_ss: the steady-state actuation
        optimality_error: the optimality error of the steady-state actuation
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    # set the bounds (not used by all solvers)
    lb = jnp.min(
        jnp.stack(
            [(params["h"] * params["phi_max"]).flatten(), jnp.zeros((num_rods,))]
        ),
        axis=0,
    )
    ub = jnp.max(
        jnp.stack(
            [(params["h"] * params["phi_max"]).flatten(), jnp.zeros((num_rods,))]
        ),
        axis=0,
    )

    # pass the desired end effector position and the initial configuration to the rollout and residual functions
    rollout_fn = partial(rollout_fn, q0=q0)
    residual_fn = partial(residual_fn, pee_des=pee_des, q0=q0)
    jac_residual_fn = partial(jac_residual_fn, pee_des=pee_des, q0=q0)

    # solve the nonlinear least squares problem
    optimality_error = None
    if solver_type == "scipy_least_squares":
        sol = least_squares(
            residual_fn,
            phi0,
            jac=jac_residual_fn,
            method="lm",
            verbose=1 if verbose else 0,
        )
        # optimal steady-state phi
        phi_ss = sol.x
        # compute the L2 optimality
        optimality_error = jnp.linalg.norm(sol.fun)
    elif solver_type == "jaxopt_levenberg_marquardt":
        # solve the nonlinear least squares problem
        lm = jo.LevenbergMarquardt(residual_fun=residual_fn)
        sol = lm.run(phi0)
        # optimal steady-state phi
        phi_ss = sol.params
        # compute the L2 optimality
        optimality_error = lm.l2_optimality_error(sol.params)
    elif solver_type == "jaxopt_projected_gradient":
        # set the lower and upper bounds for the optimization problem
        pg = jo.ProjectedGradient(
            fun=lambda x: 0.5 * jnp.mean(residual_fn(x) ** 2),
            maxiter=10,
            projection=jo.projection.projection_box,
            # decrease_factor=0.8,
            # tol=1e-5,
        )
        phi_ss, info = pg.run(phi0, hyperparams_proj=(lb, ub))
        optimality_error = pg.l2_optimality_error(phi_ss, hyperparams_proj=(lb, ub))
    elif solver_type == "optimistix_levenberg_marquardt":
        lm = optx.LevenbergMarquardt(
            rtol=1e-10,
            atol=1e-10,
            verbose=frozenset({"step", "accepted", "loss", "step_size"}) if verbose else None,
        )
        sol = optx.least_squares(residual_fn, lm, phi0, max_steps=10)
        phi_ss = sol.value
    else:
        raise ValueError(f"Unknown solver: {solver_type}")

    # compute the L2 optimality if it has not been computed yet
    if optimality_error is None:
        optimality_error = jnp.linalg.norm(residual_fn(phi_ss))

    # compute the steady-state configuration and end effector pose
    chiee_ss, q_ss, q_d_ss = rollout_fn(phi_ss)

    if verbose:
        print(
            "phi_ss", phi_ss,
            "chiee_ss", chiee_ss, "pee_des", pee_des, "e_pee", chiee_ss[:2] - pee_des,
            "q_ss", q_ss, "q_d_ss", q_d_ss,
            "optimality_error", optimality_error
        )

    return chiee_ss, q_ss, phi_ss, optimality_error
