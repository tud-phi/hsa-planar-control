from jax import Array
from typing import Dict


def trim_time_series_data(
    data_ts: Dict[str, Array], start_time: Array, duration: Array = None
) -> Dict[str, Array]:
    for key, item in data_ts.items():
        if key[:2] == "ts" and key != "controller_info_ts":
            time_selector = item >= start_time
            if duration is not None:
                time_selector = time_selector & (item <= (start_time + duration))
            data_ts[key] = item[time_selector] - start_time
            data_ts[key[3:] + "_ts"] = data_ts[key[3:] + "_ts"][time_selector]
    if "controller_info_ts" in data_ts.keys():
        ci_ts = data_ts["controller_info_ts"]
        time_selector = ci_ts["ts"] >= start_time
        if duration is not None:
            time_selector = time_selector & (ci_ts["ts"] <= (start_time + duration))
        for key in ci_ts.keys():
            ci_ts[key] = ci_ts[key][time_selector]
        # calibrate the time to start at 0.0
        ci_ts["ts"] = ci_ts["ts"] - start_time
        data_ts["controller_info_ts"] = ci_ts
    return data_ts
