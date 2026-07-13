import numpy as np
import pandas as pd

from predweem_optimizer import (
    default_parameters,
    optimize_parameters,
    prepare_field,
    simulate_emergence,
    validate_independently,
)


class DummyANN:
    def predict(self, X):
        jd = X[:, 0]
        y = 0.05 + 0.9 * np.exp(-0.5 * ((jd - 105.0) / 18.0) ** 2)
        return y, np.cumsum(y)


def make_weather():
    dates = pd.date_range("2026-01-01", periods=190, freq="D")
    jd = dates.dayofyear.to_numpy()
    return pd.DataFrame({
        "Fecha": dates,
        "TMAX": 25 + 8 * np.sin(2 * np.pi * jd / 365),
        "TMIN": 12 + 6 * np.sin(2 * np.pi * jd / 365),
        "Prec": np.where(jd % 13 == 0, 18.0, 0.0),
    })


def make_field(weather, params):
    sim = simulate_emergence(weather, DummyANN(), params)
    dates = pd.date_range("2026-02-01", "2026-06-20", freq="14D")
    flows = []
    previous = dates[0]
    flows.append(0.0)
    for current in dates[1:]:
        flows.append(sim.loc[(sim.Fecha > previous) & (sim.Fecha <= current), "EMERREL"].sum())
        previous = current
    return pd.DataFrame({"FECHA": dates, "PLM2": flows})


def test_pipeline_smoke():
    weather = make_weather()
    params = default_parameters()
    params.update({"lag_dias": 8, "umbral_termoinhibicion": 27.0, "recarga_relativa": 0.25})
    cal = prepare_field(make_field(weather, params), value_mode="interval")
    val_raw = make_field(weather, params)
    val_raw["PLM2"] *= 1.05
    val = prepare_field(val_raw, value_mode="interval")

    result = optimize_parameters(
        weather,
        cal,
        DummyANN(),
        optimized_parameters=["lag_dias", "umbral_termoinhibicion"],
        n_global=20,
        n_local=10,
        seed=7,
    )
    independent = validate_independently(result, weather, val, DummyANN())
    assert not result["results"].empty
    assert "lag_dias" in result["best_params"]
    assert np.isfinite(independent["summary"]["Score_Validacion"])
    assert not independent["sync"].empty
