# -*- coding: utf-8 -*-
"""Búsqueda global/local y validación independiente."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import json
import numpy as np
import pandas as pd
from .data import DEFAULT_OPTIMIZED_PARAMETERS, PARAMETER_SPACE, default_parameters, prepare_weather, prepare_field, ParameterSpec
from .model import simulate_emergence, synchronize_intervals, validation_metrics, objective_score


def _cast_parameter(name: str, value: float) -> float | int:
    spec = PARAMETER_SPACE[name]
    clipped = np.clip(value, spec.low, spec.high)
    return int(round(clipped)) if spec.integer else float(clipped)


def sample_parameter_sets(
    n: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
    fixed_parameters: Mapping[str, float | int] | None = None,
) -> list[dict[str, float | int]]:
    fixed = default_parameters()
    if fixed_parameters:
        fixed.update(fixed_parameters)
    result: list[dict[str, float | int]] = []
    # Muestreo estratificado tipo Latin Hypercube independiente por dimensión.
    draws: dict[str, np.ndarray] = {}
    for name in optimized_parameters:
        spec = PARAMETER_SPACE[name]
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        draws[name] = spec.low + bins * (spec.high - spec.low)
    for i in range(n):
        candidate = dict(fixed)
        for name in optimized_parameters:
            candidate[name] = _cast_parameter(name, float(draws[name][i]))
        result.append(candidate)
    return result


def local_parameter_sets(
    seeds: Sequence[Mapping[str, float | int]],
    n_per_seed: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
    scale: float = 0.10,
) -> list[dict[str, float | int]]:
    result: list[dict[str, float | int]] = []
    for seed in seeds:
        for _ in range(n_per_seed):
            candidate = dict(seed)
            for name in optimized_parameters:
                spec = PARAMETER_SPACE[name]
                sigma = (spec.high - spec.low) * scale
                candidate[name] = _cast_parameter(name, float(seed[name]) + rng.normal(0.0, sigma))
            result.append(candidate)
    return result


def _split_field_groups(field: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(str(group), part.reset_index(drop=True)) for group, part in field.groupby("Grupo", sort=False)]


def evaluate_candidate(
    weather: pd.DataFrame,
    field: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    latitude: float,
    robustness_penalty: float,
    weights: Mapping[str, float] | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    simulation = simulate_emergence(weather, ann_model, params, latitude)
    group_records: list[dict[str, Any]] = []
    sync_frames: list[pd.DataFrame] = []
    for group, group_field in _split_field_groups(field):
        sync = synchronize_intervals(simulation, group_field)
        metrics = validation_metrics(sync)
        score = objective_score(metrics, weights)
        group_records.append({"Grupo": group, "Score": score, **metrics})
        if not sync.empty:
            sync = sync.copy()
            sync["Grupo"] = group
            sync_frames.append(sync)

    group_df = pd.DataFrame(group_records)
    scores = group_df["Score"].to_numpy(float) if not group_df.empty else np.array([0.0])
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    robust_score = mean_score - float(robustness_penalty) * std_score
    summary = {
        "Score_Calibracion": robust_score,
        "Score_Medio": mean_score,
        "Score_SD": std_score,
        "Score_Peor_Grupo": float(np.min(scores)),
        "N_Grupos": int(len(group_df)),
    }
    for metric in ["KGE_Flujos", "NSE_Flujos", "CCC_Acumulado", "RMSE_Acumulado", "F1_Score", "Desfase_T50"]:
        summary[f"{metric}_Media"] = float(group_df[metric].mean()) if not group_df.empty else np.nan
    sync_all = pd.concat(sync_frames, ignore_index=True) if sync_frames else pd.DataFrame()
    return summary, group_df, sync_all


def optimize_parameters(
    weather_calibration: pd.DataFrame,
    field_calibration: pd.DataFrame,
    ann_model: Any,
    *,
    optimized_parameters: Sequence[str] = DEFAULT_OPTIMIZED_PARAMETERS,
    fixed_parameters: Mapping[str, float | int] | None = None,
    n_global: int = 400,
    n_local: int = 200,
    seed: int = 42,
    latitude: float = -33.9443,
    robustness_penalty: float = 0.15,
    weights: Mapping[str, float] | None = None,
    top_seeds: int = 5,
) -> dict[str, Any]:
    unknown = [name for name in optimized_parameters if name not in PARAMETER_SPACE]
    if unknown:
        raise ValueError(f"Parámetros desconocidos: {', '.join(unknown)}")
    weather = prepare_weather(weather_calibration)
    field = prepare_field(field_calibration) if "Observado" not in field_calibration.columns else field_calibration.copy()
    rng = np.random.default_rng(seed)

    candidates = sample_parameter_sets(max(1, int(n_global)), optimized_parameters, rng, fixed_parameters)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        summary, _, _ = evaluate_candidate(weather, field, ann_model, candidate, latitude, robustness_penalty, weights)
        rows.append({**candidate, **summary, "Etapa": "global"})

    global_df = pd.DataFrame(rows).sort_values("Score_Calibracion", ascending=False)
    seeds = global_df.head(max(1, int(top_seeds)))[list(PARAMETER_SPACE)].to_dict("records")
    if n_local > 0:
        local_candidates = local_parameter_sets(
            seeds,
            max(1, int(np.ceil(n_local / len(seeds)))),
            optimized_parameters,
            rng,
        )[: int(n_local)]
        for candidate in local_candidates:
            summary, _, _ = evaluate_candidate(weather, field, ann_model, candidate, latitude, robustness_penalty, weights)
            rows.append({**candidate, **summary, "Etapa": "local"})

    results = pd.DataFrame(rows)
    results = results.drop_duplicates(subset=list(PARAMETER_SPACE), keep="first")
    results = results.sort_values(["Score_Calibracion", "Score_Peor_Grupo"], ascending=False).reset_index(drop=True)
    best_params = {name: results.iloc[0][name] for name in PARAMETER_SPACE}
    for name, spec in PARAMETER_SPACE.items():
        best_params[name] = int(best_params[name]) if spec.integer else float(best_params[name])
    best_summary, calibration_by_group, calibration_sync = evaluate_candidate(
        weather, field, ann_model, best_params, latitude, robustness_penalty, weights
    )
    return {
        "best_params": best_params,
        "best_summary": best_summary,
        "results": results,
        "calibration_by_group": calibration_by_group,
        "calibration_sync": calibration_sync,
        "weather_calibration": weather,
        "field_calibration": field,
    }


def validate_independently(
    optimization_result: Mapping[str, Any],
    weather_validation: pd.DataFrame,
    field_validation: pd.DataFrame,
    ann_model: Any,
    *,
    latitude: float = -33.9443,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    weather = prepare_weather(weather_validation)
    field = prepare_field(field_validation) if "Observado" not in field_validation.columns else field_validation.copy()
    params = optimization_result["best_params"]
    simulation = simulate_emergence(weather, ann_model, params, latitude)
    group_rows: list[dict[str, Any]] = []
    sync_frames: list[pd.DataFrame] = []
    for group, group_field in _split_field_groups(field):
        sync = synchronize_intervals(simulation, group_field)
        metrics = validation_metrics(sync)
        group_rows.append({"Grupo": group, "Score_Validacion": objective_score(metrics, weights), **metrics})
        if not sync.empty:
            sync = sync.copy()
            sync["Grupo"] = group
            sync_frames.append(sync)
    by_group = pd.DataFrame(group_rows)
    sync_all = pd.concat(sync_frames, ignore_index=True) if sync_frames else pd.DataFrame()
    summary = {
        "Score_Validacion": float(by_group["Score_Validacion"].mean()) if not by_group.empty else 0.0,
        "Score_Validacion_SD": float(by_group["Score_Validacion"].std(ddof=0)) if not by_group.empty else 0.0,
        "N_Grupos_Validacion": int(len(by_group)),
    }
    for metric in ["KGE_Flujos", "NSE_Flujos", "CCC_Acumulado", "RMSE_Acumulado", "F1_Score", "Desfase_T50"]:
        summary[metric] = float(by_group[metric].mean()) if not by_group.empty else np.nan
    return {
        "summary": summary,
        "by_group": by_group,
        "sync": sync_all,
        "simulation": simulation,
        "weather_validation": weather,
        "field_validation": field,
    }


def parameter_importance(results: pd.DataFrame, optimized_parameters: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name in optimized_parameters:
        if name in results and results[name].nunique() > 1:
            rho = results[[name, "Score_Calibracion"]].corr(method="spearman").iloc[0, 1]
            rows.append({"Parametro": name, "Rho_Spearman": float(rho), "Importancia_Abs": abs(float(rho))})
    return pd.DataFrame(rows).sort_values("Importancia_Abs", ascending=False).reset_index(drop=True)


def params_to_json(params: Mapping[str, Any]) -> str:
    serializable = {k: (int(v) if PARAMETER_SPACE.get(k, ParameterSpec(0, 0, 0)).integer else float(v)) for k, v in params.items()}
    return json.dumps(serializable, ensure_ascii=False, indent=2)
