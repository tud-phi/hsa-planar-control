from diffrax import Dopri5, Euler
import diffrax as dfx
from jax import Array, jacfwd, jacrev, jit, random, vmap
import jax.numpy as jnp
import jaxopt as jo
import optimistix as optx
from scipy.optimize import least_squares
from typing import Callable, Dict, Tuple

from hsa_planar_control.simulation import simulate_steady_state


def plan_with_rollout_to_steady_state(
        params: Dict[str, Array],
        forward_kinematics_end_effector_fn: Callable,
        dynamical_matrices_fn: Callable,
        pee_des: Array,
        q0: Array,
        phi0: Array,
        sim_dt: float = 1e-4,
        duration: float = 10.0,
        solver="scipy_least_squares",
) -> Tuple[Array, Array, Array, Array]:
    """
    Plan the steady-state actuation and configuration for a given desired end effector position.
    Args:
        params: a dictionary of robot parameters
        forward_kinematics_end_effector_fn: a function that computes the end effector pose from the configuration
        dynamical_matrices_fn: a function that computes the dynamical matrices
        pee_des: the desired end effector position
        q0: the initial configuration used for the rollout
        phi0: the initial guess for the steady-state actuation
        sim_dt: the time step for the simulation
        duration: the duration of the simulation (i.e. the time we expect the system to take to reach steady-state)

    Returns:

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

    @jit
    def residual_fn(_phi_ss: Array, *args) -> Array:
        _q_ss, _q_d_ss = simulate_steady_state(
            dynamical_matrices_fn=dynamical_matrices_fn,
            params=params,
            q0=q0,
            phi_ss=_phi_ss,
            sim_dt=sim_dt,
            duration=duration,
            allow_forward_autodiff=True
        )

        _chiee_ss = forward_kinematics_end_effector_fn(params, _q_ss)
        _residual = _chiee_ss[:2] - pee_des
        return _residual

    # solve the nonlinear least squares problem
    optimality_error = None
    if solver == "scipy_least_squares":
        jac_fn = jit(jacfwd(residual_fn))
        sol = least_squares(residual_fn, phi0, jac=jac_fn, method="lm", verbose=2)
        # optimal steady-state phi
        phi_ss = sol.x
        # compute the L2 optimality
        optimality_error = jnp.linalg.norm(sol.fun)
    elif solver == "jaxopt_levenberg_marquardt":
        # solve the nonlinear least squares problem
        lm = jo.LevenbergMarquardt(residual_fun=residual_fn)
        sol = lm.run(phi0)
        # optimal steady-state phi
        phi_ss = sol.params
        # compute the L2 optimality
        optimality_error = lm.l2_optimality_error(sol.params)
    elif solver == "jaxopt_projected_gradient":
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
    elif solver == "optimistix_levenberg_marquardt":
        lm = optx.LevenbergMarquardt(
            rtol=1e-10,
            atol=1e-10,
            verbose=frozenset({"step", "accepted", "loss", "step_size"})
        )
        sol = optx.least_squares(residual_fn, lm, phi0)
        phi_ss = sol.value
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # compute the L2 optimality if it has not been computed yet
    if optimality_error is None:
        optimality_error = jnp.linalg.norm(residual_fn(phi_ss))

    # compute the steady-state configuration and end effector pose
    q_ss, q_d_ss = simulate_steady_state(
        dynamical_matrices_fn=dynamical_matrices_fn,
        params=params,
        q0=q0,
        phi_ss=phi_ss,
        sim_dt=sim_dt,
        duration=duration
    )
    chiee_ss = forward_kinematics_end_effector_fn(params, q_ss)

    return chiee_ss, q_ss, phi_ss, optimality_error
