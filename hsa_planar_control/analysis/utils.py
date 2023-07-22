from jax import Array
from typing import Dict


def trim_time_series_data(
    data_ts: Dict[str, Array], start_time: Array, duration: Array = None
) -> Dict[str, Array]:
    for key, item in data_ts.items():
        if key[:2] == "ts":
            time_selector = item >= start_time
            if duration is not None:
                time_selector = time_selector & (item <= (start_time + duration))
            data_ts[key] = item[time_selector] - start_time
            data_ts[key[3:] + "_ts"] = data_ts[key[3:] + "_ts"][time_selector]
    return data_ts
