from diffrax import diffeqsolve, Dopri5, Euler, ODETerm, SaveAt
from jax import Array, debug, jit, lax, vmap
from jax import numpy as jnp
from functools import partial
from jsrm.systems import planar_hsa
from typing import Callable, Dict, Optional, Tuple


def simulate_closed_loop_system(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    q0: Array,
    q_d0: Array,
    phi0: Array,
    sim_dt: float,
    control_dt: float,
    duration: float,
    control_fn: Callable = None,
    controller_state_init: Optional[Dict[str, Array]] = None,
    ode_solver_class=Euler,
    consider_underactuation_model: bool = True,
) -> Dict[str, Array]:
    num_segments = params["l"].shape[0]
    num_rods_per_segment = params["rout"].shape[1]
    n_q = q0.shape[0]

    ode_fn = planar_hsa.ode_factory(
        dynamical_matrices_fn,
        params=params,
        consider_underactuation_model=consider_underactuation_model,
    )
    ode_solver = ode_solver_class()

    control_ts = jnp.arange(0.0, duration, control_dt)
    x0 = jnp.concatenate((q0, q_d0))

    @jit
    def scan_fn(carry, t):
        x, phi = carry["x_next"], carry["phi_next"]
        q, q_d = x[:n_q], x[n_q:]

        if control_fn is None:
            if consider_underactuation_model:
                u = jnp.zeros((num_segments * num_rods_per_segment,))
            else:
                u = jnp.zeros((n_q,))
        else:
            if controller_state_init is None:
                u, controller_info = control_fn(t, q, q_d, phi)
            else:
                u, controller_state, controller_info = control_fn(
                    t, q, q_d, phi, controller_state=carry["controller_state"]
                )
                carry["controller_state"] = controller_state

        ode_term = ODETerm(ode_fn)

        sol = diffeqsolve(
            ode_term,
            solver=ode_solver,
            t0=t,
            t1=t + control_dt,
            dt0=sim_dt,
            y0=x,
            args=u,
            max_steps=None,
        )

        x_next = sol.ys[-1]

        carry["x_next"] = x_next
        if consider_underactuation_model:
            carry["phi_next"] = u

        output = {
            "t_ts": t,
            "q_ts": q,
            "q_d_ts": q_d,
            "x_ts": x,
            "u_ts": u,
            "phi_ts": carry["phi_next"],
        }

        if control_fn is not None:
            output["controller_info_ts"] = controller_info

        return carry, output

    init_carry = dict(x_next=x0, phi_next=phi0, controller_state=controller_state_init)
    last_carry, sim_ts = lax.scan(scan_fn, init_carry, control_ts)

    return sim_ts


def simulate_steady_state(
    dynamical_matrices_fn: Callable,
    params: Dict[str, Array],
    phi_ss: Array,
    sim_dt: float = 1e-3,
    duration: float = 5.0,
    ode_solver_class=Dopri5,
) -> Tuple[Array, Array]:
    """
    Simulate the steady-state response of the system.
    Args:
        dynamical_matrices_fn: function that returns the dynamical matrices
        params: dictionary of system parameters
        phi_ss: steady-state control inputs
        sim_dt: time step for simulation
        duration: duration of simulation. This is the time we expect the system to take to reach steady-state.
        ode_solver_class: ODE solver class

    Returns:
        q_ss: steady-state configuration
        q_d_ss: steady-state configuration velocity. Should be almost zero, otherwise duration is too short.
    """
    num_segments = params["l"].shape[0]
    n_q = 3 * num_segments  # number of generalized coordinates
    # initial condition
    q0 = jnp.zeros((n_q,))  # initial configuration
    q_d0 = jnp.zeros((n_q,))  # initial velocity
    x0 = jnp.concatenate((q0, q_d0))

    ode_fn = planar_hsa.ode_factory(dynamical_matrices_fn, params=params)
    ode_term = ODETerm(ode_fn)
    ode_solver = ode_solver_class()

    sol = diffeqsolve(
        ode_term,
        solver=ode_solver,
        t0=0.0,
        t1=duration,
        dt0=sim_dt,
        y0=x0,
        args=phi_ss,
        max_steps=int(duration // sim_dt) + 10,
    )

    # last (i.e. steady-state) configuration
    q_ss = sol.ys[-1][:n_q]
    # also return the last configuration velocity so that we can check that it is (almost) zero
    q_d_ss = sol.ys[-1][n_q:]

    return q_ss, q_d_ss
