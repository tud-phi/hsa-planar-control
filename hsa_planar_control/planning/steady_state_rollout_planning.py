from diffrax import Dopri5, Euler
from jax import Array, jit, random, vmap
import jax.numpy as jnp
import jaxopt as jo
from typing import Callable, Dict, Tuple

from hsa_planar_control.simulation import simulate_steady_state


def plan_with_rollout_to_steady_state(
        params: Dict[str, Array],
        forward_kinematics_end_effector_fn: Callable,
        dynamical_matrices_fn: Callable,
        pee_des: Array,
        phi0: Array,
        sim_dt: float = 1e-4,
        duration: float = 10.0,
) -> Tuple[Array, Array, Array, Array]:
    """
    Plan the steady-state actuation and configuration for a given desired end effector position.
    Args:
        params: a dictionary of robot parameters
        forward_kinematics_end_effector_fn: a function that computes the end effector pose from the configuration
        dynamical_matrices_fn: a function that computes the dynamical matrices
        pee_des: the desired end effector position
        phi0: the initial guess for the steady-state actuation
        sim_dt: the time step for the simulation
        duration: the duration of the simulation (i.e. the time we expect the system to take to reach steady-state)

    Returns:

    """
    @jit
    def residual_fn(_phi_ss: Array) -> Array:
        # TODO: add q0 as an argument to this function
        _q_ss, _q_d_ss = simulate_steady_state(
            dynamical_matrices_fn=dynamical_matrices_fn,
            params=params,
            phi_ss=_phi_ss,
            sim_dt=sim_dt,
            duration=duration
        )
        _chiee_ss = forward_kinematics_end_effector_fn(params, _q_ss)
        _residual = _chiee_ss[:2] - pee_des
        return _residual

    # # solve the nonlinear least squares problem
    # lm = jo.LevenbergMarquardt(residual_fun=residual_fn)
    # sol = lm.run(phi0)
    # # optimal steady-state phi
    # phi_ss = sol.params
    # # compute the L2 optimality
    # optimality_error = lm.l2_optimality_error(sol.params)

    # set the lower and upper bounds for the optimization problem
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]
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
    pg = jo.ProjectedGradient(
        fun=lambda x: 0.5 * jnp.mean(residual_fn(x) ** 2),
        maxiter=1000,
        projection=jo.projection.projection_box,
        decrease_factor=0.8,
        tol=1e-5,
    )
    phi_ss, info = pg.run(phi0, hyperparams_proj=(lb, ub))
    optimality_error = pg.l2_optimality_error(phi_ss, hyperparams_proj=(lb, ub))
    print("phi_ss", phi_ss, "info", info, "optimality_error", optimality_error)

    # compute the steady-state configuration and end effector pose
    q_ss, q_d_ss = simulate_steady_state(
        dynamical_matrices_fn=dynamical_matrices_fn,
        params=params,
        phi_ss=phi_ss,
        sim_dt=1e-4,
        duration=10.0,
        ode_solver_class=Euler
    )
    chiee_ss = forward_kinematics_end_effector_fn(params, q_ss)

    return chiee_ss, q_ss, phi_ss, optimality_error
