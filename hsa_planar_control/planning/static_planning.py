from functools import partial
from jax import Array, debug, jacfwd, jit
import jax.numpy as jnp
import jaxopt as jo
from typing import Callable, Tuple


def static_inversion_factory(
    params: Array,
    inverse_kinematics_end_effector_fn: Callable,
    dynamical_matrices_fn: Callable,
) -> Callable:
    """
    Create the residual function for static inversion.
    Args:
        params: Dictionary of robot parameters
        pee_des: Desired end-effector position of shape (2, )
        inverse_kinematics_end_effector_fn: Callable that returns the configuration vector given the end-effector pose.
        dynamical_matrices_fn: Callable that returns the B, C, G, K, D, and alpha vectors.
    Returns:
        residual_fn: Callable that returns the residual vector given the end-effector orientation and the motor positions.
    """

    @jit
    def residual_fn(_x: Array, pee_des: Array) -> Array:
        _theta = _x[0]
        _phi = _x[1:]

        # end-effector pose
        _chiee = jnp.concatenate((pee_des, _theta.reshape(1)))

        # compute the configuration using inverse kinematics
        _q = inverse_kinematics_end_effector_fn(params, _chiee)

        # compute the dynamical matrices at the current configuration and phi
        _, _, _G, _K, _, _alpha = dynamical_matrices_fn(
            params, _q, jnp.zeros_like(_q), _phi
        )

        lhs = _G + _K
        rhs = _alpha
        residual = lhs - rhs

        return residual

    return residual_fn


def statically_invert_actuation_to_task_space_scipy_rootfinding(
    params: Array,
    pee_des: Array,
    residual_fn: Callable,
    inverse_kinematics_end_effector_fn: Callable,
    maxiter: int = 1000,
    verbose: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute the desired configuration from an operational space target position by applying static inversion.
        Use scipy root finding to find the desired configuration that minimizes the residual between the forces
        applied by the actuator and the steady-state forces (gravity + elasticity).
    Args:
        params: Dictionary of robot parameters
        pee_des: Desired end-effector position of shape (2, )
        residual_fn: Callable that returns the residual vector given the end-effector orientation and the motor positions.
        inverse_kinematics_end_effector_fn: Callable that returns the configuration vector given the end-effector pose.
        maxiter: Maximum number of iterations for the optimization algorithm.
        verbose: If True, print the optimization result.
    Returns:
        chiee_des: Desired end-effector pose vector of shape (3, )
        q_des: Desired configuration vector of shape (n_q, ), which should be feasible for the given actuator
            characteristics.
        phi_ss: Steady-state motor positions vector of shape (n_phi, )
        optimality_error: L2 norm of the optimality residual.
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    # initial guess for [theta, phi1, ..., phin]
    x0 = jnp.zeros((1 + num_rods,))
    x0 = x0.at[1:].set(jnp.pi)  # initialize phi to 180 deg

    # partial off pee_des from residual_fn
    residual_fn = partial(residual_fn, pee_des=pee_des)

    # solve the optimization problem
    solver = jo.ScipyRootFinding(
        optimality_fun=residual_fn, method="lm", options={"maxiter": maxiter}, jit=True
    )
    x_best, info = solver.run(x0)
    optimality_error = solver.l2_optimality_error(x_best)
    if verbose:
        debug.print(
            "ScipyRootFinding finished with x_best = {x_best}, optimality_error = {optimality_error} and info:\n{info}",
            x_best=x_best,
            optimality_error=optimality_error,
            info=info,
        )

    th_des = x_best[0]
    phi_ss = x_best[1:]

    # desired end-effector pose
    chiee_des = jnp.concatenate((pee_des, th_des.reshape(1)))

    # desired configuration
    q_des = inverse_kinematics_end_effector_fn(params, chiee_des)

    return chiee_des, q_des, phi_ss, optimality_error


def statically_invert_actuation_to_task_space_projected_descent(
    params: Array,
    pee_des: Array,
    residual_fn: Callable,
    inverse_kinematics_end_effector_fn: Callable,
    maxiter: int = 2500,
    verbose: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute the desired configuration from an operational space target position by applying static inversion.
        Use projected gradient descent to find the desired configuration that minimizes the residual between the forces
        applied by the actuator and the steady-state forces (gravity + elasticity).
    Args:
        params: Dictionary of robot parameters
        pee_des: Desired end-effector position of shape (2, )
        residual_fn: Callable that returns the residual vector given the end-effector orientation and the motor positions.
        inverse_kinematics_end_effector_fn: Callable that returns the configuration vector given the end-effector pose.
        maxiter: Maximum number of iterations for the optimization algorithm.
        verbose: If True, print the optimization result.
    Returns:
        chiee_des: Desired end-effector pose vector of shape (3, )
        q_des: Desired configuration vector of shape (n_q, ), which should be feasible for the given actuator
            characteristics.
        phi_ss: Steady-state motor positions vector of shape (n_phi, )
        optimality_error: L2 norm of the optimality residual.
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    # initial guess for [theta, phi1, ..., phin]
    x0 = jnp.zeros((1 + num_rods,))
    # TODO: in-place assignment probably slows us down a lot
    x0 = x0.at[1:].set(jnp.pi / 2)  # initialize phi to 90 deg

    # partial off pee_des from residual_fn
    residual_fn = partial(residual_fn, pee_des=pee_des)

    # set the lower and upper bounds for the optimization problem
    lb = jnp.zeros((1 + num_rods,))
    ub = jnp.zeros((1 + num_rods,))
    # TODO: in-place assignment probably slows us down a lot
    lb = lb.at[0].set(-jnp.pi)
    ub = ub.at[0].set(jnp.pi)
    lb = lb.at[1:].set(
        jnp.min(
            jnp.stack(
                [(params["h"] * params["phi_max"]).flatten(), jnp.zeros((num_rods,))]
            ),
            axis=0,
        )
    )
    ub = ub.at[1:].set(
        jnp.max(
            jnp.stack(
                [(params["h"] * params["phi_max"]).flatten(), jnp.zeros((num_rods,))]
            ),
            axis=0,
        )
    )

    solver = jo.ProjectedGradient(
        fun=lambda x: 0.5 * jnp.mean(residual_fn(x) ** 2),
        maxiter=maxiter,
        projection=jo.projection.projection_box,
        decrease_factor=0.8,
        tol=1e-5,
    )
    x_best, info = solver.run(x0, hyperparams_proj=(lb, ub))
    optimality_error = solver.l2_optimality_error(x_best, hyperparams_proj=(lb, ub))
    if verbose:
        debug.print(
            "Nonlinear Least squares finished with x_best = {x_best}, optimality_error = {optimality_error}, "
            "and info:\n{info}",
            x_best=x_best,
            optimality_error=optimality_error,
            info=info,
        )

    th_des = x_best[0]
    phi_ss = x_best[1:]

    # desired end-effector pose
    chiee_des = jnp.concatenate((pee_des, th_des.reshape(1)))

    # desired configuration
    q_des = inverse_kinematics_end_effector_fn(params, chiee_des)

    return chiee_des, q_des, phi_ss, optimality_error


def statically_invert_actuation_to_task_space_projected_descent_straight_config(
    params: Array,
    pee_des: Array,
    residual_fn: Callable,
    inverse_kinematics_end_effector_fn: Callable,
    maxiter: int = 2500,
    verbose: bool = True,
) -> Tuple[Array, Array, Array, Array]:
    """
    The optimization problem near straight configurations is ill-conditioned (for example because of eps sign switching).
    This function assumes that both actuators are symmetrically (i.e. equally actuated) for "almost-straight" configurations.
    Attention: only works if the robot has two rods and both have positive handedness and have the same properties
    (i.e. same length, same stiffness, same radius, etc.)
    Args:
        params: Dictionary of robot parameters
        pee_des: Desired end-effector position of shape (2, )
        residual_fn: Callable that returns the residual vector given the end-effector orientation and the motor positions.
        inverse_kinematics_end_effector_fn: Callable that returns the configuration vector given the end-effector pose.
        maxiter: Maximum number of iterations for the optimization algorithm.
        verbose: If True, print the optimization result.
    Returns:
        chiee_des: Desired end-effector pose vector of shape (3, )
        q_des: Desired configuration vector of shape (n_q, ), which should be feasible for the given actuator
            characteristics.
        phi_ss: Steady-state motor positions vector of shape (n_phi, )
        optimality_error: L2 norm of the optimality residual.
    """
    num_rods = params["rout"].shape[0] * params["rout"].shape[1]

    # initial guess for [phi]
    x0 = jnp.array([jnp.pi / 2])  # initialize phi to 90 deg

    # partial off pee_des from residual_fn
    residual_fn = partial(residual_fn, pee_des=pee_des)

    # set the lower and upper bounds for the optimization problem
    lb = jnp.array([0.0])
    ub = jnp.array([jnp.mean(params["phi_max"])])

    def cost_fn(x: Array):
        # set theta to 0
        # expand phi to phi1 and phi2
        x = jnp.concatenate((jnp.array([0.0]), jnp.repeat(x, num_rods)))
        return 0.5 * jnp.mean(residual_fn(x) ** 2)

    solver = jo.ProjectedGradient(
        fun=cost_fn,
        maxiter=maxiter,
        projection=jo.projection.projection_box,
        decrease_factor=0.8,
        tol=1e-5,
    )
    x_best, info = solver.run(x0, hyperparams_proj=(lb, ub))
    optimality_error = solver.l2_optimality_error(x_best, hyperparams_proj=(lb, ub))
    if verbose:
        debug.print(
            "Nonlinear Least squares finished with x_best = {x_best}, optimality_error = {optimality_error}, "
            "and info:\n{info}",
            x_best=x_best,
            optimality_error=optimality_error,
            info=info,
        )

    # theta is 0.0 for almost straight configurations
    th_des = jnp.array([0.0])
    # phi is the same for all rods
    phi_ss = jnp.repeat(x_best, num_rods)
    debug.print("phi_ss = {phi_ss}", phi_ss=phi_ss)

    # desired end-effector pose
    chiee_des = jnp.concatenate((pee_des, th_des.reshape(1)))

    # desired configuration
    q_des = inverse_kinematics_end_effector_fn(params, chiee_des)

    return chiee_des, q_des, phi_ss, optimality_error
