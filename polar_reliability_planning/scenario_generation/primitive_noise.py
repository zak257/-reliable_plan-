"""Module/mechanism keyed Philox streams; no draw-order dependent coupling.

Measured hours and negative warmup hours use separate stable streams. Increasing
capacity, horizon or warmup does not alter any shared device/hour random number.
The arrays belong exclusively to the simulator, never to its Observation.
"""
import numpy as np

NOISE_VERSION = "philox-device-mechanism-hour-v1"
MECHANISMS = {"diesel_run": 1, "diesel_start": 2, "diesel_repair": 3,
              "wind_run": 4, "wind_repair": 5, "pv_run": 6,
              "pv_repair": 7, "pcs_run": 8, "pcs_repair": 9}


class PrimitiveNoise:
    def __init__(self, seed, scenario, hours, max_units, warmup=0):
        self.seed, self.scenario, self.warmup = int(seed), int(scenario), warmup
        self.arrays = {}
        for mechanism, code in MECHANISMS.items():
            count = max_units[mechanism.split("_")[0]]
            arr = np.empty((hours + warmup, count))
            for device in range(count):
                rng = np.random.Generator(np.random.Philox([seed, scenario, code, device, 0]))
                arr[warmup:, device] = rng.random(hours)
                if warmup:
                    rng = np.random.Generator(np.random.Philox([seed, scenario, code, device, 1]))
                    arr[:warmup, device] = rng.random(warmup)[::-1]
            self.arrays[mechanism] = arr

    def uniform(self, mechanism, hour, device):
        return float(self.arrays[mechanism][hour + self.warmup, device])


def repair_duration(uniform, mean, distribution="geometric"):
    import math
    if distribution == "fixed":
        if mean != int(mean):
            raise ValueError("Fixed hourly repair requires an integer duration")
        return int(mean)
    if distribution != "geometric" or mean < 1:
        raise ValueError("Repair requires geometric mean >= 1 or fixed integer duration")
    if mean == 1:
        return 1
    return 1 + int(math.floor(math.log1p(-min(uniform, 1 - 1e-15)) / math.log1p(-1 / mean)))
