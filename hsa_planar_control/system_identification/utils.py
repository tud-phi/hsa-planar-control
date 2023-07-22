from jax import Array, vmap
import jax.numpy as jnp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import (
    InterpolatedUnivariateSpline as InterpolatedUnivariateSpline_scipy,
)


def resample_trajectory(
    t_ts: Array, y_ts: Array, that_ts: Array, k: int = 3, use_jax: bool = False
) -> Array:
    """
    Resample the trajectory to a fixed time step
    Use the jax-cosmo implementation of scipy.interpolate.InterpolatedUnivariateSpline
    https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/interpolate.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html
    Arguments:
        t_ts: time stamps of the trajectory of shape (N,)
        y_ts: trajectory data of shape (N, n_y)
        that_ts: time stamps of the resampled trajectory pf shape (Nhat,)
        k: degree of the spline
        use_jax: whether to use the jax-cosmo implementation of scipy.interpolate.InterpolatedUnivariateSpline
    Returns:
        yhat_ts: resampled trajectory data of shape (Nhat, n_y)
    """
    if use_jax:
        yhat_ts = vmap(
            lambda _y_ts: InterpolatedUnivariateSpline(t_ts, _y_ts, k=k)(that_ts),
            in_axes=-1,
            out_axes=-1,
        )(y_ts)
    else:
        yhat_ts = jnp.zeros((that_ts.shape[0], y_ts.shape[-1]))
        for i in range(y_ts.shape[-1]):
            spline = InterpolatedUnivariateSpline_scipy(t_ts, y_ts[:, i], k=k)
            yhat_ts = yhat_ts.at[:, i].set(jnp.array(spline(that_ts)))

    return yhat_ts
