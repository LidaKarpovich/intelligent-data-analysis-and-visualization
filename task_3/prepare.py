"""
prepare.py — офлайн-расчёт артефактов XAI / SHAP для дашборда задания 3 (команда team = 5).

Запуск из папки task_3:
    python prepare.py
    python prepare.py --team 5 --path ../data_1/iavd_ml_dataset.parquet --course-dir ../task_2/course_model/IAVD_3.7_ML_team
    python prepare.py --skip-tsne        # если нужно ускорить отладочный прогон (~на 15 с)

Результаты сохраняются в task_3/data/agg/:
    feature_registry.parquet   — реестр 73 полей (card / engineered, тип, допустимые XAI-методы)
    team5_cards.parquet        — 8 карточек команды 5: контекст, исходы, метаданные и сырые поля
    team5_field_values.parquet — длинная таблица (8 × 73): значения полей + SHAP + доли + ранги
    team5_shap_1099.parquet    — детализация SHAP на уровне 1099 модельных фич для 8 карточек
    practice_320_shap.parquet  — 320 карточек практики (40 команд × 8) с SHAP-73 и исходами
    global_importance.parquet  — mean|SHAP|, mean(SHAP), доля, ранг (Приёмка / Эксплуатация / Практика) + Ablation
    beeswarm_sample.parquet    — стратифицированная выборка (до 3000 строк) для SHAP Summary Plot
    curves_numeric.parquet     — ALE, PDP и предупреждение о корреляции Спирмена для 43 числовых признаков
    curves_ice_team5.parquet   — ICE-кривые для 8 карточек team=5 + 40 фоновых карточек
    curves_cat_gpdp.parquet    — group-PDP (what-if) для 12 категориальных полей (OneHotEncoder)
    drift_summary.parquet      — PSI, пропуски и сдвиг SHAP (Приёмка -> Эксплуатация) по 73 полям
    drift_bins.parquet         — готовые бины распределений (Приёмка vs Эксплуатация) для графика дрейфа
    map_hex_shap.parquet       — 2D-гексагоны (SHAP 73D -> PCA 2D) по всей области модели
    map_hex_pca.parquet        — 2D-гексагоны (матрица признаков -> SVD 20D -> 2D)
    map_hex_tsne.parquet       — 2D-гексагоны (SHAP 73D -> t-SNE 2D на стратифицированной выборке)
    map_clusters.parquet       — профили кластеров (MiniBatchKMeans) в обоих пространствах
    map_focus_team5.parquet    — 2D-координаты 8 карточек team=5 во всех проекциях + кластер
    neighbors_team5.parquet    — 20 ближайших соседей в 73D SHAP и 20D SVD для каждой карточки team=5
    verification_team5.parquet — проверка топ-12 признаков каждой карточки team=5 через 95 % ДИ Уилсона
    meta.json                  — контрольные числа § 4, периоды, порог, сверка аддитивности, нагрузки PCA
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MaxAbsScaler

from course_model import CourseAdvisorXAI, find_course_dir, sigmoid
from feature_registry import (
    CAT_FEATURES_12,
    FEATURE_ORDER_73,
    NUMERIC_FEATURES_43,
    REGISTRY,
    registry_frame,
)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
AGG_DIR = BASE / "data" / "agg"
RAW_NAME = "iavd_ml_dataset.parquet"
XLSX_NAME = "3.7_Пр_cards_practice.xlsx"

SEARCH_RAW = [BASE / "data", ROOT / "task_2" / "data", ROOT / "data_1", ROOT / "data"]
SEARCH_XLSX = [ROOT / XLSX_NAME, BASE / XLSX_NAME, ROOT / "task_2" / XLSX_NAME]

WINDOW = (pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC"))
OPER_START = WINDOW[1]
LABELS = {"Решён": 1, "False Positive": 0}
PUBLISHED_TEST = dict(tp=390, fp=107, fn=24, tn=2652)

EXTRA_COLS = [
    "EntityId", "Создан", "Вердикт", "Наименование", "Тип инцидента", "Пиктограмма IRP",
    "Критичность", "Организация", "Вендор из сием", "Источник инцидента", "Группа устранения",
    "Алерт", "Значение Инцидента (группировка)", "Правило корреляции SIEM",
]


# ================================================================ утилиты поиска и парсинга
def find_raw(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"Нет файла {p}")
        return p
    for d in SEARCH_RAW:
        if (d / RAW_NAME).exists():
            return (d / RAW_NAME).resolve()
    raise SystemExit(f"Не нашёл {RAW_NAME} в: " + ", ".join(str(d) for d in SEARCH_RAW))


def load_practice_table(course_dir: Path, xlsx_path: str | None = None) -> tuple[pd.DataFrame, str]:
    """Читает 3.7_Пр_cards_practice.xlsx (или course_pack/cards_practice.csv как резерв)."""
    candidates = [Path(xlsx_path).expanduser()] if xlsx_path else SEARCH_XLSX
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_excel(p, sheet_name="cards_practice")
                return df, str(p.resolve())
            except ImportError:
                print(f"Предупреждение: openpyxl не установлен для чтения {p.name}, беру course_pack/cards_practice.csv")
                break
    csv_path = course_dir / "course_pack" / "cards_practice.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            head = f.readline()
        sep = ";" if head.count(";") > head.count(",") else ","
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
        return df, str(csv_path.resolve())
    raise SystemExit("Не нашёл ни 3.7_Пр_cards_practice.xlsx, ни course_pack/cards_practice.csv")


def name_of(x) -> str | None:
    if not isinstance(x, str) or not x:
        return None
    if x.startswith("{"):
        try:
            return json.loads(x).get("name")
        except ValueError:
            return None
    return x


def first_item(x) -> str | None:
    if not isinstance(x, str) or not x:
        return None
    if x.startswith("["):
        try:
            items = json.loads(x, strict=False)
            return str(items[0]) if items else None
        except (ValueError, IndexError):
            return x
    return x


def clean_display_value(v, max_len: int = 160) -> str:
    """Форматирует сырое/сконструированное значение поля для таблицы карточки."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "— (пропуск)"
    if isinstance(v, (np.floating, float)):
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e9:
            return f"{int(round(v)):,}".replace(",", " ")
        return f"{v:.4g}"
    if isinstance(v, (np.integer, int)):
        return f"{int(v):,}".replace(",", " ")
    s = str(v).strip()
    if not s or s.lower() == "none":
        return "— (пропуск)"
    if s.startswith("{"):
        parsed = name_of(s)
        if parsed:
            s = parsed
    elif s.startswith("["):
        try:
            items = json.loads(s, strict=False)
            if isinstance(items, list):
                s = "; ".join(str(x) for x in items[:5]) + (" …" if len(items) > 5 else "")
        except ValueError:
            pass
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def compute_psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """PSI и готовые бины гистограммы для числового признака."""
    r = ref[np.isfinite(ref)]
    c = cur[np.isfinite(cur)]
    if len(r) < 10 or len(c) < 10:
        return 0.0, []
    uniq = np.unique(r)
    if len(uniq) <= 8:
        cats = sorted(set(uniq.tolist()) | set(np.unique(c).tolist()))
        r_cnt = np.array([(r == v).sum() for v in cats], dtype=float)
        c_cnt = np.array([(c == v).sum() for v in cats], dtype=float)
        labels = [clean_display_value(v, 24) for v in cats]
        edges = cats
    else:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(r, qs))
        if len(edges) < 2:
            return 0.0, []
        edges[0] = min(edges[0], float(c.min())) - 1e-9
        edges[-1] = max(edges[-1], float(c.max())) + 1e-9
        r_cnt, _ = np.histogram(r, bins=edges)
        c_cnt, _ = np.histogram(c, bins=edges)
        labels = [f"{edges[i]:.3g} … {edges[i + 1]:.3g}" for i in range(len(edges) - 1)]

    eps = 1e-6
    r_p = r_cnt / max(r_cnt.sum(), 1.0)
    c_p = c_cnt / max(c_cnt.sum(), 1.0)
    psi = float(np.sum((c_p - r_p) * np.log((c_p + eps) / (r_p + eps))))
    bins = [
        {"bin_idx": i, "bin_label": labels[i], "val_share": float(r_p[i]), "oper_share": float(c_p[i]),
         "val_n": int(r_cnt[i]), "oper_n": int(c_cnt[i])}
        for i in range(len(labels))
    ]
    return psi, bins


def compute_psi_categorical(ref: pd.Series, cur: pd.Series, top_k: int = 10) -> tuple[float, list[dict]]:
    """PSI и доли топ-уровней + «Пропуск» для категориального признака."""
    r = ref.map(lambda v: "— (пропуск)" if pd.isna(v) or str(v).strip() in ("", "None") else clean_display_value(v, 50))
    c = cur.map(lambda v: "— (пропуск)" if pd.isna(v) or str(v).strip() in ("", "None") else clean_display_value(v, 50))
    top_levels = list(pd.concat([r, c]).value_counts().head(top_k).index)
    if "— (пропуск)" not in top_levels and ((r == "— (пропуск)").any() or (c == "— (пропуск)").any()):
        top_levels.append("— (пропуск)")
    r_g = r.where(r.isin(top_levels), "Прочие уровни")
    c_g = c.where(c.isin(top_levels), "Прочие уровни")
    levels = list(dict.fromkeys(top_levels + (["Прочие уровни"] if (r_g.eq("Прочие уровни").any() or c_g.eq("Прочие уровни").any()) else [])))
    r_vc = r_g.value_counts()
    c_vc = c_g.value_counts()
    r_cnt = np.array([r_vc.get(lv, 0) for lv in levels], dtype=float)
    c_cnt = np.array([c_vc.get(lv, 0) for lv in levels], dtype=float)
    eps = 1e-6
    r_p = r_cnt / max(len(r), 1)
    c_p = c_cnt / max(len(c), 1)
    psi = float(np.sum((c_p - r_p) * np.log((c_p + eps) / (r_p + eps))))
    bins = [
        {"bin_idx": i, "bin_label": lv, "val_share": float(r_p[i]), "oper_share": float(c_p[i]),
         "val_n": int(r_cnt[i]), "oper_n": int(c_cnt[i])}
        for i, lv in enumerate(levels)
    ]
    return psi, bins


# ================================================================ загрузка и сплит (как в task_2)
def needed_columns(advisor: CourseAdvisorXAI, available: list[str]) -> list[str]:
    cfg = advisor.cfg
    cols = set(EXTRA_COLS) | set(advisor.data_mod.LIST_LEN_SOURCE)
    for group in cfg["features"]["fields"].values():
        cols |= set(group)
    for key in ("column", "first_event", "last_event"):
        cols.add(cfg["dataset"]["time"][key])
    cols |= {"suser", "duser", "ProcessPath", "Список вредоносных команд", cfg["dataset"]["target"]["column"]}
    return [c for c in available if c in cols]


def load_raw(path: Path, advisor: CourseAdvisorXAI) -> tuple[pd.DataFrame, int]:
    pf = pq.ParquetFile(path)
    total_rows = int(pf.metadata.num_rows)
    cols = needed_columns(advisor, pf.schema_arrow.names if pf.schema_arrow is not None else pf.column_names)
    print(f"Читаю {path.name}: всего {total_rows:,} строк, беру {len(cols)} колонок с 01.06.2025")
    parts = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        created = pd.to_datetime(df["Создан"], format="ISO8601", utc=True)
        parts.append(df[created >= WINDOW[0]])
    return pd.concat(parts, ignore_index=True), total_rows


def in_scope(df: pd.DataFrame, advisor: CourseAdvisorXAI) -> pd.Series:
    flt = advisor.cfg["dataset"]["filters"]
    mask = pd.Series(True, index=df.index)
    if flt.get("rule_not_null"):
        mask &= df["Правило корреляции SIEM"].notna()
    for vendor in flt.get("vendor_not_in", []) or []:
        mask &= ~df["Вендор из сием"].eq(vendor)
    return mask


def window_split(raw: pd.DataFrame, advisor: CourseAdvisorXAI) -> dict[str, set[int]]:
    cfg = advisor.cfg
    created = pd.to_datetime(raw["Создан"], format="ISO8601", utc=True)
    mask = (
        in_scope(raw, advisor)
        & raw[cfg["dataset"]["target"]["column"]].isin(cfg["dataset"]["filters"]["verdict_in"])
        & (created >= WINDOW[0])
        & (created < WINDOW[1])
    )
    df = raw[mask].reset_index(drop=True)
    prepared, y, created_w = advisor.data_mod.prepare(df, cfg)
    parts = advisor.train_mod.make_split(prepared, y, created_w, cfg)
    ids = pd.to_numeric(df["EntityId"]).astype("int64")
    return {k: set(ids.loc[v]) for k, v in parts.items()}


# ================================================================ ALE / PDP / ICE / group-PDP / Ablation
def compute_ale_pdp_ice(
    advisor: CourseAdvisorXAI,
    prepared_sub: pd.DataFrame,
    X_sub: np.ndarray,
    prepared_t5: pd.DataFrame,
    X_t5: np.ndarray,
    team5_ids: np.ndarray,
    period_label: str,
    n_grid: int = 20,
    n_bg_ice: int = 40,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Считает 1D-ALE, 1D-PDP и ICE для всех 43 числовых признаков + макс. корреляцию Спирмена."""
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_sub), size=min(n_bg_ice, len(X_sub)), replace=False)

    # Матрица корреляций Спирмена для предупреждения об искажении PDP при мультиколлинеарности
    num_df = prepared_sub[NUMERIC_FEATURES_43].apply(pd.to_numeric, errors="coerce")
    corr_mat = np.zeros((len(NUMERIC_FEATURES_43), len(NUMERIC_FEATURES_43)), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_corr = spearmanr(num_df.to_numpy(dtype=float), nan_policy="omit")
        if np.ndim(res_corr.statistic) == 2:
            corr_mat = np.nan_to_num(res_corr.statistic, nan=0.0)
    np.fill_diagonal(corr_mat, 0.0)

    curve_rows: list[dict] = []
    ice_rows: list[dict] = []

    for pos, col in enumerate(NUMERIC_FEATURES_43):
        feat_idx = int(advisor.col_indices[col][0])
        raw_v = pd.to_numeric(prepared_sub[col], errors="coerce").to_numpy(dtype=float)
        q_v = X_sub[:, feat_idx].astype(np.float64)
        valid = np.isfinite(raw_v) & np.isfinite(q_v)

        abs_c = np.abs(corr_mat[pos])
        max_j = int(np.argmax(abs_c))
        max_corr = float(abs_c[max_j])
        corr_partner = NUMERIC_FEATURES_43[max_j] if max_corr > 0 else ""

        if valid.sum() < 20:
            continue

        r_ok, q_ok = raw_v[valid], q_v[valid]
        uniq_r = np.unique(r_ok)
        is_discrete = len(uniq_r) <= n_grid

        if is_discrete:
            raw_grid = uniq_r
            q_grid = np.array([float(np.median(q_ok[r_ok == u])) for u in raw_grid])
            quantiles = np.array([float((r_ok <= u).mean()) for u in raw_grid])
        else:
            qs = np.linspace(0.0, 1.0, n_grid)
            order = np.argsort(r_ok)
            r_sorted, q_sorted = r_ok[order], q_ok[order]
            idxs = np.clip((qs * (len(r_sorted) - 1)).astype(int), 0, len(r_sorted) - 1)
            raw_grid_all = r_sorted[idxs]
            q_grid_all = q_sorted[idxs]
            # Убираем дубликаты по raw_grid, сохраняя последние квантили
            _, uniq_idx = np.unique(raw_grid_all, return_index=True)
            uniq_idx = np.sort(uniq_idx)
            raw_grid = raw_grid_all[uniq_idx]
            q_grid = q_grid_all[uniq_idx]
            quantiles = qs[uniq_idx]

        k_pts = len(raw_grid)
        if k_pts < 2:
            continue

        # --- PDP + ICE (одним батчем) ---
        n_s = len(X_sub)
        pdp_mean = np.empty(k_pts, dtype=float)
        pdp_std = np.empty(k_pts, dtype=float)
        ice_bg = np.empty((len(bg_idx), k_pts), dtype=float)
        ice_t5 = np.empty((len(X_t5), k_pts), dtype=float)

        X_tmp = X_sub.copy()
        X_t5_tmp = X_t5.copy()
        for gi, qv in enumerate(q_grid):
            X_tmp[:, feat_idx] = np.float32(qv)
            p_all = advisor.predict_proba_matrix(X_tmp)
            pdp_mean[gi] = float(p_all.mean())
            pdp_std[gi] = float(p_all.std())
            ice_bg[:, gi] = p_all[bg_idx]

            X_t5_tmp[:, feat_idx] = np.float32(qv)
            ice_t5[:, gi] = advisor.predict_proba_matrix(X_t5_tmp)

        # --- ALE (локальные приращения без экстраполяции в нереалистичные зоны) ---
        deltas = np.zeros(k_pts - 1, dtype=float)
        counts = np.zeros(k_pts - 1, dtype=int)
        for bi in range(k_pts - 1):
            lo, hi = raw_grid[bi], raw_grid[bi + 1]
            if is_discrete:
                mask = valid & ((raw_v == lo) | (raw_v == hi))
            else:
                mask = valid & (raw_v >= lo if bi == 0 else raw_v > lo) & (raw_v <= hi)
            cnt = int(mask.sum())
            counts[bi] = cnt
            if cnt == 0:
                continue
            X_lo = X_sub[mask].copy()
            X_hi = X_lo.copy()
            X_lo[:, feat_idx] = np.float32(q_grid[bi])
            X_hi[:, feat_idx] = np.float32(q_grid[bi + 1])
            deltas[bi] = float((advisor.predict_proba_matrix(X_hi) - advisor.predict_proba_matrix(X_lo)).mean())

        ale_raw = np.concatenate([[0.0], np.cumsum(deltas)])
        # Центрирование ALE по весам бинов (взвешенное среднее = 0)
        tot = int(counts.sum())
        if tot > 0:
            mids = 0.5 * (ale_raw[:-1] + ale_raw[1:])
            ale_centered = ale_raw - float(np.sum(counts * mids) / tot)
        else:
            ale_centered = ale_raw - float(ale_raw.mean())

        # Оценка числа наблюдений у каждой узловой точки (для гистограммы плотности под графиком)
        node_counts = np.zeros(k_pts, dtype=int)
        if is_discrete:
            for gi, rv in enumerate(raw_grid):
                node_counts[gi] = int((r_ok == rv).sum())
        else:
            node_counts[:-1] = counts
            node_counts[-1] = counts[-1] if len(counts) else 0

        for gi in range(k_pts):
            curve_rows.append({
                "period": period_label,
                "source_column": col,
                "grid_idx": gi,
                "raw_value": float(raw_grid[gi]),
                "quantile": float(quantiles[gi]),
                "ale": float(ale_centered[gi]),
                "pdp": float(pdp_mean[gi]),
                "pdp_std": float(pdp_std[gi]),
                "bin_n": int(node_counts[gi]),
                "is_discrete": bool(is_discrete),
                "max_spearman": max_corr,
                "corr_partner": corr_partner,
                "collinear_warn": bool(max_corr >= 0.70),
            })

        # Сохраняем ICE только для эксплуатации (чтобы не раздувать файл вдвое)
        if period_label == "Эксплуатация":
            t5_raw = pd.to_numeric(prepared_t5[col], errors="coerce").to_numpy(dtype=float)
            for ti, eid in enumerate(team5_ids):
                for gi in range(k_pts):
                    ice_rows.append({
                        "source_column": col,
                        "curve_kind": "team5",
                        "curve_id": str(int(eid)),
                        "grid_idx": gi,
                        "raw_value": float(raw_grid[gi]),
                        "proba": float(ice_t5[ti, gi]),
                        "card_raw_value": float(t5_raw[ti]) if np.isfinite(t5_raw[ti]) else np.nan,
                    })
            for bi in range(len(bg_idx)):
                for gi in range(k_pts):
                    ice_rows.append({
                        "source_column": col,
                        "curve_kind": "bg",
                        "curve_id": f"bg_{bi}",
                        "grid_idx": gi,
                        "raw_value": float(raw_grid[gi]),
                        "proba": float(ice_bg[bi, gi]),
                        "card_raw_value": np.nan,
                    })

    return pd.DataFrame(curve_rows), pd.DataFrame(ice_rows)


def compute_group_pdp_categorical(
    advisor: CourseAdvisorXAI,
    X_sub: np.ndarray,
    period_label: str,
    top_k: int = 12,
) -> pd.DataFrame:
    """group-PDP (what-if) для 12 категориальных полей (OneHotEncoder): ставим поле целиком в каждый уровень."""
    base_proba = float(advisor.predict_proba_matrix(X_sub).mean())
    fm = advisor.feature_mapping
    rows: list[dict] = []

    for col in CAT_FEATURES_12:
        sub_fm = fm[fm["source_column"] == col]
        all_idxs = sub_fm["feature_index"].to_numpy(dtype=int)
        # Считаем фактическую частоту каждого OHE-уровня в выборке X_sub
        level_counts = X_sub[:, all_idxs].sum(axis=0)
        order = np.argsort(-level_counts)[:top_k]

        X_zero = X_sub.copy()
        X_zero[:, all_idxs] = 0.0
        p_zero = advisor.predict_proba_matrix(X_zero)
        n_zero = int((X_sub[:, all_idxs].sum(axis=1) == 0).sum())
        rows.append({
            "period": period_label,
            "source_column": col,
            "level": "— (пропуск / вне OHE)",
            "n_observed": n_zero,
            "base_proba": base_proba,
            "mean_proba": float(p_zero.mean()),
            "median_proba": float(np.median(p_zero)),
            "delta_proba": float(p_zero.mean() - base_proba),
        })

        for pos in order:
            idx = int(all_idxs[pos])
            fname = str(sub_fm.iloc[pos]["feature"])
            level_raw = fname.split(": ", 1)[1] if ": " in fname else fname
            X_mod = X_zero.copy()
            X_mod[:, idx] = 1.0
            p_mod = advisor.predict_proba_matrix(X_mod)
            rows.append({
                "period": period_label,
                "source_column": col,
                "level": clean_display_value(level_raw, 64),
                "n_observed": int(round(level_counts[pos])),
                "base_proba": base_proba,
                "mean_proba": float(p_mod.mean()),
                "median_proba": float(np.median(p_mod)),
                "delta_proba": float(p_mod.mean() - base_proba),
            })
    return pd.DataFrame(rows)


def compute_ablation(advisor: CourseAdvisorXAI, X_sub: np.ndarray) -> dict[str, float]:
    """Ablation по всем 73 полям: среднее |Δ proba| при обнулении / замене поля на медиану."""
    p_base = advisor.predict_proba_matrix(X_sub)
    med = np.median(X_sub, axis=0).astype(np.float32)
    out: dict[str, float] = {}
    for col in FEATURE_ORDER_73:
        idxs = advisor.col_indices[col]
        X_mod = X_sub.copy()
        if REGISTRY[col].last_step in ("OneHotEncoder", "TfidfVectorizer", "PCA"):
            X_mod[:, idxs] = 0.0
        else:
            X_mod[:, idxs] = med[idxs]
        p_mod = advisor.predict_proba_matrix(X_mod)
        out[col] = float(np.abs(p_base - p_mod).mean())
    return out


# ================================================================ 2D Hexbin + кластеры + 20-NN
def hexbin_aggregate(
    coords: np.ndarray,
    y: np.ndarray,
    proba: np.ndarray,
    tau: float,
    clusters: np.ndarray,
    rules: np.ndarray,
    parts: np.ndarray,
    gridsize: int = 34,
) -> pd.DataFrame:
    """Точное гексагональное бинирование в чистом numpy (масштабируется на 40k–240k+ точек)."""
    x = coords[:, 0].astype(np.float64)
    yv = coords[:, 1].astype(np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(yv.min()), float(yv.max())
    dx = max(xmax - xmin, 1e-9)
    dy = max(ymax - ymin, 1e-9)

    u = (x - xmin) / dx * gridsize
    v = (yv - ymin) / dy * gridsize

    # Плоские сверху гексагоны (axial -> cube round)
    qf = (2.0 / 3.0) * u
    rf = (-1.0 / 3.0) * u + (np.sqrt(3.0) / 3.0) * v
    sf = -qf - rf

    rq, rr, rs = np.round(qf), np.round(rf), np.round(sf)
    dq, dr, ds = np.abs(rq - qf), np.abs(rr - rf), np.abs(rs - sf)
    m_q = (dq > dr) & (dq > ds)
    m_r = (~m_q) & (dr > ds)
    rq = np.where(m_q, -rr - rs, rq).astype(int)
    rr = np.where(m_r, -rq - rs, rr).astype(int)

    df = pd.DataFrame({
        "hq": rq,
        "hr": rr,
        "y": y,
        "proba": proba,
        "pred": (proba >= tau).astype(np.int8),
        "cluster": clusters,
        "rule": rules,
        "is_oper": (parts == "Эксплуатация").astype(np.int8),
    })

    rows = []
    for (hq, hr), g in df.groupby(["hq", "hr"], sort=False):
        uc = 1.5 * hq
        vc = (np.sqrt(3.0) / 2.0) * hq + np.sqrt(3.0) * hr
        xc = xmin + (uc / gridsize) * dx
        yc = ymin + (vc / gridsize) * dy
        lab = g[g["y"] >= 0]
        n_lab = len(lab)
        inc_rate = float(lab["y"].mean()) if n_lab else np.nan
        err_rate = float((lab["pred"] != lab["y"]).mean()) if n_lab else np.nan
        fn_rate = float(((lab["y"] == 1) & (lab["pred"] == 0)).mean()) if n_lab else np.nan
        dom_cl = int(g["cluster"].value_counts().index[0])
        top_r = str(g["rule"].dropna().value_counts().index[0]) if g["rule"].notna().any() else "—"
        rows.append({
            "x": float(xc),
            "y": float(yc),
            "n": int(len(g)),
            "n_labeled": int(n_lab),
            "n_oper": int(g["is_oper"].sum()),
            "inc_rate": inc_rate,
            "pred_rate": float(g["pred"].mean()),
            "err_rate": err_rate,
            "fn_rate": fn_rate,
            "mean_proba": float(g["proba"].mean()),
            "dominant_cluster": dom_cl,
            "top_rule": clean_display_value(top_r, 48),
        })
    return pd.DataFrame(rows)


def _hex_assign(coords: np.ndarray, gridsize: int) -> tuple[np.ndarray, np.ndarray]:
    x, yv = coords[:, 0].astype(np.float64), coords[:, 1].astype(np.float64)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(yv.min()), float(yv.max())
    u = (x - xmin) / max(xmax - xmin, 1e-9) * gridsize
    v = (yv - ymin) / max(ymax - ymin, 1e-9) * gridsize
    qf = (2.0 / 3.0) * u
    rf = (-1.0 / 3.0) * u + (np.sqrt(3.0) / 3.0) * v
    sf = -qf - rf
    rq, rr, rs = np.round(qf), np.round(rf), np.round(sf)
    dq, dr, ds = np.abs(rq - qf), np.abs(rr - rf), np.abs(rs - sf)
    m_q = (dq > dr) & (dq > ds)
    m_r = (~m_q) & (dr > ds)
    rq = np.where(m_q, -rr - rs, rq).astype(np.int32)
    rr = np.where(m_r, -rq - rs, rr).astype(np.int32)
    return rq, rr


def build_similarity_maps(
    shap_73_all: np.ndarray,
    X_all: np.ndarray,
    meta_all: pd.DataFrame,
    team5_ids: np.ndarray,
    tau: float,
    run_tsne: bool = True,
    n_clusters: int = 12,
    n_neighbors: int = 20,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Строит Hexbin-карты (SHAP-PCA, SVD-PCA, SHAP-tSNE), кластеры и многомерный 20-NN."""
    y_all = meta_all["y"].to_numpy(dtype=np.int8)
    p_all = meta_all["proba"].to_numpy(dtype=np.float64)
    r_all = meta_all["rule"].to_numpy()
    part_all = meta_all["part"].to_numpy()
    eids_all = meta_all["EntityId"].to_numpy(dtype=np.int64)
    eid_to_pos = {int(e): i for i, e in enumerate(eids_all)}
    t5_pos = np.array([eid_to_pos[int(e)] for e in team5_ids], dtype=int)

    # 1. Пространство SHAP 73D (без StandardScaler, чтобы не раздувать нулевые шумовые фичи)
    pca_shap = PCA(n_components=2, random_state=42)
    xy_shap = pca_shap.fit_transform(shap_73_all)
    km_shap = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=5)
    cl_shap = km_shap.fit_predict(shap_73_all)

    # Топ-3 нагрузки PC1/PC2 для читаемых осей графика
    shap_axis_info = {}
    for ci, name in enumerate(["PC1", "PC2"]):
        w = pca_shap.components_[ci]
        top3 = np.argsort(-np.abs(w))[:3]
        shap_axis_info[name] = {
            "var_ratio": float(pca_shap.explained_variance_ratio_[ci]),
            "top_features": [f"{REGISTRY[FEATURE_ORDER_73[j]].label_ru} ({w[j]:+.2f})" for j in top3],
        }

    hex_shap = hexbin_aggregate(xy_shap, y_all, p_all, tau, cl_shap, r_all, part_all, gridsize=34)

    # 2. Пространство признаков (MaxAbsScaler + TruncatedSVD 20D)
    X_scaled = MaxAbsScaler().fit_transform(X_all)
    svd = TruncatedSVD(n_components=20, random_state=42)
    Z_pca20 = svd.fit_transform(X_scaled).astype(np.float32)
    xy_pca = Z_pca20[:, :2]
    km_pca = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=5)
    cl_pca = km_pca.fit_predict(Z_pca20)
    hex_pca = hexbin_aggregate(xy_pca, y_all, p_all, tau, cl_pca, r_all, part_all, gridsize=34)
    pca_axis_info = {
        "PC1": {"var_ratio": float(svd.explained_variance_ratio_[0])},
        "PC2": {"var_ratio": float(svd.explained_variance_ratio_[1])},
    }

    # 3. Опционально t-SNE по SHAP (стратифицированная выборка 6 000 + все 8 карточек team=5)
    xy_tsne_t5 = np.full((len(team5_ids), 2), np.nan)
    if run_tsne:
        rng = np.random.default_rng(42)
        t5_set = set(t5_pos.tolist())
        pool = np.array([i for i in range(len(shap_73_all)) if i not in t5_set], dtype=int)
        # Стратифицируем по исходу, чтобы редкие инциденты и FN были хорошо представлены
        take = min(6000 - len(t5_pos), len(pool))
        sub_idx = np.concatenate([t5_pos, rng.choice(pool, size=take, replace=False)])
        z_in = PCA(n_components=20, random_state=42).fit_transform(shap_73_all[sub_idx])
        print(f"Считаю t-SNE по SHAP на {len(sub_idx):,} карточках (20D PCA -> 2D)...")
        xy_tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42).fit_transform(z_in)
        xy_tsne_t5 = xy_tsne[:len(t5_pos)]
        hex_tsne = hexbin_aggregate(
            xy_tsne, y_all[sub_idx], p_all[sub_idx], tau, cl_shap[sub_idx], r_all[sub_idx], part_all[sub_idx], gridsize=30
        )
    else:
        hex_tsne = pd.DataFrame(columns=hex_shap.columns)
    
    hq_sh, hr_sh = _hex_assign(xy_shap, 34)
    hq_pc, hr_pc = _hex_assign(xy_pca, 34)
    hq_ts = np.full(len(y_all), -10**9, dtype=np.int32)
    hr_ts = np.full(len(y_all), -10**9, dtype=np.int32)
    if run_tsne:
        hq_ts[sub_idx], hr_ts[sub_idx] = _hex_assign(xy_tsne, 30)

    points_tau_df = pd.DataFrame({
        "hq_shap": hq_sh, "hr_shap": hr_sh,
        "hq_pca": hq_pc, "hr_pca": hr_pc,
        "hq_tsne": hq_ts, "hr_tsne": hr_ts,
        "y": y_all.astype(np.int8),
        "proba": p_all.astype(np.float32),
    })

    # Профили кластеров (для подписей центроидов на карте)
    cl_rows = []
    for space, xy, cl in [("shap", xy_shap, cl_shap), ("pca", xy_pca, cl_pca)]:
        for cid in range(n_clusters):
            m = cl == cid
            if not m.any():
                continue
            lab_m = m & (y_all >= 0)
            vc = pd.Series(r_all[m]).dropna().value_counts()
            top_rule = clean_display_value(vc.index[0], 44) if len(vc) else "—"
            cl_rows.append({
                "space": space,
                "cluster": int(cid),
                "n": int(m.sum()),
                "cx": float(np.median(xy[m, 0])),
                "cy": float(np.median(xy[m, 1])),
                "inc_rate": float(y_all[lab_m].mean()) if lab_m.any() else np.nan,
                "pred_rate": float((p_all[m] >= tau).mean()),
                "mean_proba": float(p_all[m].mean()),
                "top_rule": top_rule,
            })
    clusters_df = pd.DataFrame(cl_rows)

    # Координаты 8 фокусных карточек team=5
    focus_df = pd.DataFrame({
        "EntityId": team5_ids.astype(np.int64),
        "x_shap": xy_shap[t5_pos, 0],
        "y_shap": xy_shap[t5_pos, 1],
        "cluster_shap": cl_shap[t5_pos].astype(int),
        "x_pca": xy_pca[t5_pos, 0],
        "y_pca": xy_pca[t5_pos, 1],
        "cluster_pca": cl_pca[t5_pos].astype(int),
        "x_tsne": xy_tsne_t5[:, 0],
        "y_tsne": xy_tsne_t5[:, 1],
    })

    # Точный поиск 20 ближайших соседей строго в многомерном пространстве (среди размеченных)
    labeled_idx = np.flatnonzero(y_all >= 0)
    nbr_rows = []
    for space, mat, xy in [("shap", shap_73_all, xy_shap), ("pca", Z_pca20, xy_pca)]:
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine", algorithm="brute")
        nn.fit(mat[labeled_idx])
        dists, inds = nn.kneighbors(mat[t5_pos])
        for ti, eid in enumerate(team5_ids):
            rank = 0
            for dist, li in zip(dists[ti], inds[ti]):
                orig_i = int(labeled_idx[li])
                nbr_eid = int(eids_all[orig_i])
                if nbr_eid == int(eid):
                    continue
                rank += 1
                if rank > n_neighbors:
                    break
                ny = int(y_all[orig_i])
                np_ = float(p_all[orig_i])
                pred = int(np_ >= tau)
                out_lbl = {
                    (1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"
                }[(ny, pred)]
                nbr_rows.append({
                    "space": space,
                    "target_id": int(eid),
                    "neighbor_rank": rank,
                    "neighbor_id": nbr_eid,
                    "cosine_sim": float(1.0 - dist),
                    "part": str(part_all[orig_i]),
                    "y_true": ny,
                    "proba": np_,
                    "outcome_at_tau": out_lbl,
                    "rule": clean_display_value(r_all[orig_i], 56),
                    "x": float(xy[orig_i, 0]),
                    "y": float(xy[orig_i, 1]),
                })
    neighbors_df = pd.DataFrame(nbr_rows)

    tables = {
        "map_hex_shap": hex_shap,
        "map_hex_pca": hex_pca,
        "map_hex_tsne": hex_tsne,
        "map_points_tau": points_tau_df,
        "map_clusters": clusters_df,
        "map_focus_team5": focus_df,
        "neighbors_team5": neighbors_df,
    }
    axis_meta = {"shap": shap_axis_info, "pca": pca_axis_info, "tsne_enabled": bool(run_tsne)}
    return tables, axis_meta


# ================================================================ верификация объяснений (Страница 5)
def build_verification_stats(
    team5_fields: pd.DataFrame,
    prepared_all: pd.DataFrame,
    y_all: np.ndarray,
    parts_all: np.ndarray,
    prepared_t5: pd.DataFrame,
    team5_ids: np.ndarray,
    top_k: int = 12,
) -> pd.DataFrame:
    """Для топ-12 полей каждой карточки team=5 считает фактическую долю инцидентов и 95 % ДИ Уилсона."""
    lab_mask = y_all >= 0
    rows = []
    t5_pos_map = {int(eid): i for i, eid in enumerate(team5_ids)}

    for eid, grp in team5_fields.groupby("EntityId", sort=False):
        t_pos = t5_pos_map[int(eid)]
        top_g = grp.sort_values("abs_shap", ascending=False).head(top_k)

        for r in top_g.itertuples(index=False):
            col = r.source_column
            meta = REGISTRY[col]
            card_v = prepared_t5.iloc[t_pos][col] if col in prepared_t5.columns else None

            for period_name, p_mask in [
                ("Окно обучения", lab_mask & (parts_all != "Эксплуатация")),
                ("Эксплуатация", lab_mask & (parts_all == "Эксплуатация")),
            ]:
                p0 = float(y_all[p_mask].mean()) if p_mask.any() else np.nan
                if col not in prepared_all.columns:
                    continue
                ser = prepared_all.loc[p_mask, col]
                yv = y_all[p_mask]

                if meta.last_step == "QuantileTransformer":
                    s_num = pd.to_numeric(ser, errors="coerce").to_numpy(dtype=float)
                    cv = float(pd.to_numeric(card_v, errors="coerce")) if card_v is not None else np.nan
                    if not np.isfinite(cv):
                        m = ~np.isfinite(s_num)
                        slice_desc = "пропуск"
                    else:
                        uniq = np.unique(s_num[np.isfinite(s_num)])
                        if len(uniq) <= 6:
                            m = s_num == cv
                            slice_desc = f"= {clean_display_value(cv, 20)}"
                        elif cv == 0.0 and (s_num == 0.0).mean() >= 0.05:
                            m = s_num == 0.0
                            slice_desc = "= 0"
                        else:
                            qs = np.unique(np.quantile(s_num[np.isfinite(s_num)], np.linspace(0, 1, 6)))
                            bi = int(np.clip(np.searchsorted(qs, cv, side="right") - 1, 0, max(len(qs) - 2, 0)))
                            lo_e, hi_e = qs[bi], qs[min(bi + 1, len(qs) - 1)]
                            m = (s_num >= lo_e) & (s_num <= hi_e)
                            slice_desc = f"[{lo_e:.3g}; {hi_e:.3g}]"
                elif meta.last_step == "OneHotEncoder":
                    cv_norm = "— (пропуск)" if pd.isna(card_v) or str(card_v).strip() in ("", "None") else str(card_v)
                    s_norm = ser.map(lambda v: "— (пропуск)" if pd.isna(v) or str(v).strip() in ("", "None") else str(v))
                    m = (s_norm == cv_norm).to_numpy()
                    slice_desc = f"= {clean_display_value(card_v, 36)}"
                else:
                    # Для текста / PCA / даты: проверяем заполненность (а для точного совпадения — если n >= 10)
                    cv_null = pd.isna(card_v) or str(card_v).strip() in ("", "None")
                    s_null = ser.map(lambda v: pd.isna(v) or str(v).strip() in ("", "None")).to_numpy()
                    if cv_null:
                        m = s_null
                        slice_desc = "пропуск (пусто)"
                    else:
                        exact = (ser.astype(str) == str(card_v)).to_numpy()
                        if exact.sum() >= 10:
                            m = exact
                            slice_desc = f"= {clean_display_value(card_v, 36)}"
                        else:
                            m = ~s_null
                            slice_desc = "поле заполнено"

                n = int(m.sum())
                k = int(yv[m].sum()) if n > 0 else 0
                rate = k / n if n > 0 else np.nan
                lo, hi = wilson(k, n)

                if n < 10:
                    status = "Мало данных"
                elif r.shap_val > 0 and lo > p0:
                    status = "Подтверждено"
                elif r.shap_val < 0 and hi < p0:
                    status = "Подтверждено"
                elif r.shap_val > 0 and hi < p0:
                    status = "Противоречие"
                elif r.shap_val < 0 and lo > p0:
                    status = "Противоречие"
                else:
                    status = "Нейтрально"

                rows.append({
                    "EntityId": int(eid),
                    "period": period_name,
                    "rank_abs": int(r.rank_abs),
                    "source_column": col,
                    "label_ru": meta.label_ru,
                    "kind": meta.kind,
                    "shap_val": float(r.shap_val),
                    "slice_desc": slice_desc,
                    "n": n,
                    "k": k,
                    "inc_rate": float(rate) if np.isfinite(rate) else np.nan,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "base_inc_rate": p0,
                    "status": status,
                })
    return pd.DataFrame(rows)


# ================================================================ основная сборка
def main(raw_path: Path, course_dir: Path, team: int, xlsx_path: str | None, skip_tsne: bool) -> None:
    t0 = time.time()
    advisor = CourseAdvisorXAI(course_dir)
    tau = advisor.threshold
    logit_tau = float(np.log(tau / (1.0 - tau)))
    print(f"Модель: {advisor.model_dir} · движок: {advisor.engine} · τ = {tau:.4f} (logit = {logit_tau:.4f})")

    # 1. Загрузка сырого датасета и самопроверки
    raw, total_dataset_rows = load_raw(raw_path, advisor)
    delta_ctrl = advisor.self_check(raw)
    print(f"[OK] Самопроверка control_cards (5 карточек): Δ ≤ {delta_ctrl:.2e}")

    pack_check = advisor.verify_shap_pack()
    print(f"[OK] Сверка course_pack/shap_cards.csv ({pack_check['checked_cards']} карточек): "
          f"max|ΔSHAP| = {pack_check['max_shap_diff']:.2e}, аддитивность Δ = {pack_check['max_additivity_diff']:.2e}")

    # 2. Сплит окна обучения и эксплуатации (идентично task_2)
    split = window_split(raw, advisor)
    ids_raw = pd.to_numeric(raw["EntityId"]).astype("int64")
    created_raw = pd.to_datetime(raw["Создан"], format="ISO8601", utc=True)
    scope = in_scope(raw, advisor)

    train_df = raw[ids_raw.isin(split["train"])].copy()
    val_df = raw[ids_raw.isin(split["val"])].copy()
    test_df = raw[ids_raw.isin(split["test"])].copy()
    oper_df = raw[(created_raw >= OPER_START) & scope].copy()

    train_df["_part"] = "Обучение"
    val_df["_part"] = "Валидация"
    test_df["_part"] = "Приёмка"
    oper_df["_part"] = "Эксплуатация"

    all_scope = pd.concat([train_df, val_df, test_df, oper_df], ignore_index=True)
    print(f"Область модели: {len(all_scope):,} карточек "
          f"(train {len(train_df):,} · val {len(val_df):,} · test {len(test_df):,} · oper {len(oper_df):,})")

    # 3. Препроцессинг + TreeSHAP на всей области модели (~41.7 тыс. карточек)
    print("Считаю признаки (preproc) и TreeSHAP на всей области модели...")
    prepared_all, X_all = advisor.prepare_and_transform(all_scope)
    expl_all = advisor.explain_shap_matrix(X_all, return_full=False)
    shap_73_all = expl_all["shap_73"].astype(np.float32)
    proba_all = expl_all["proba"]
    max_add_all = float(np.abs(proba_all - advisor.predict_proba_matrix(X_all)).max())
    print(f"[OK] TreeSHAP готов за {time.time() - t0:.1f} с · аддитивность на всех {len(all_scope):,} карточках: Δ ≤ {max_add_all:.2e}")

    y_all = all_scope["Вердикт"].map(LABELS).fillna(-1).astype(np.int8).to_numpy()
    parts_all = all_scope["_part"].to_numpy()
    eids_all = pd.to_numeric(all_scope["EntityId"]).astype("int64").to_numpy()
    rules_all = all_scope["Правило корреляции SIEM"].map(first_item).to_numpy()

    # Проверка эталонных метрик приёмки (TP 390 / FP 107 / FN 24 / TN 2652)
    m_test = parts_all == "Приёмка"
    pred_t = proba_all[m_test] >= tau
    yt = y_all[m_test]
    got_pub = dict(
        tp=int((pred_t & (yt == 1)).sum()), fp=int((pred_t & (yt == 0)).sum()),
        fn=int((~pred_t & (yt == 1)).sum()), tn=int((~pred_t & (yt == 0)).sum()),
    )
    pub_match = got_pub == PUBLISHED_TEST
    print(f"[{'OK' if pub_match else 'WARN'}] Приёмка: {got_pub} (эталон: {PUBLISHED_TEST})")

    # 4. Карточки практики (320) и команды team=5 (8 карточек)
    practice_df, practice_src = load_practice_table(course_dir, xlsx_path)
    practice_df["EntityId"] = pd.to_numeric(practice_df["EntityId"]).astype("int64")
    team5_nav = practice_df[practice_df["team"] == team].sort_values("rank_in_team").reset_index(drop=True)
    if len(team5_nav) != 8:
        raise SystemExit(f"Для team={team} найдено {len(team5_nav)} карточек вместо 8")

    # Загружаем полные 94 поля из course_pack/cards_raw.parquet для 8 карточек команды 5
    cards_raw_pack = pd.read_parquet(advisor.pack_dir / "cards_raw.parquet")
    cards_raw_pack["_eid"] = pd.to_numeric(cards_raw_pack["EntityId"]).astype("int64")
    pack_idx = {eid: i for i, eid in enumerate(cards_raw_pack["_eid"].to_numpy())}
    t5_raw = cards_raw_pack.iloc[[pack_idx[eid] for eid in team5_nav["EntityId"].to_numpy()]].drop(columns=["_eid"]).reset_index(drop=True)

    prepared_t5, X_t5 = advisor.prepare_and_transform(t5_raw)
    expl_t5 = advisor.explain_shap_matrix(X_t5, return_full=True)

    # Сверка outcome и proba с файлом практики
    pred_t5 = expl_t5["proba"] >= tau
    y_t5 = team5_nav["y_true"].to_numpy(dtype=int)
    calc_out_t5 = np.array([{(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[(y, int(p))] for y, p in zip(y_t5, pred_t5)])
    outcome_match = bool((calc_out_t5 == team5_nav["outcome"].to_numpy()).all())
    max_t5_proba_diff = float(np.abs(expl_t5["proba"] - team5_nav["proba"].to_numpy(dtype=float)).max())
    print(f"[{'OK' if outcome_match else 'WARN'}] Команда team={team}: 8 карточек, совпадение outcome={outcome_match}, "
          f"max|Δproba|={max_t5_proba_diff:.2e}")

    # Сохраняем team5_cards.parquet (навигация + скор + ключевые поля карточки)
    team5_cards = team5_nav.copy()
    team5_cards["proba_calc"] = expl_t5["proba"]
    team5_cards["base"] = expl_t5["base"]
    team5_cards["raw_score"] = expl_t5["raw_score"]
    team5_cards["sum_shap"] = expl_t5["shap_73"].sum(axis=1)
    team5_cards["sum_pos"] = np.clip(expl_t5["shap_73"], 0, None).sum(axis=1)
    team5_cards["sum_neg"] = np.clip(expl_t5["shap_73"], None, 0).sum(axis=1)
    team5_cards["margin_to_tau"] = expl_t5["raw_score"] - logit_tau
    for raw_c, alias in [
        ("Правило корреляции SIEM", "rule"), ("Тип инцидента", "inc_type"),
        ("Пиктограмма IRP", "irp"), ("Критичность", "crit"),
        ("Организация", "org"), ("Вендор из сием", "vendor"), ("Алерт", "alert_text"),
    ]:
        if raw_c in t5_raw.columns:
            team5_cards[alias] = [clean_display_value(v, 140) for v in t5_raw[raw_c]]

    # Длинная таблица (8 × 73) со значениями полей и SHAP
    t5_field_rows = []
    for ti, eid in enumerate(team5_nav["EntityId"].to_numpy()):
        s_vec = expl_t5["shap_73"][ti]
        abs_v = np.abs(s_vec)
        tot_abs = float(abs_v.sum()) or 1.0
        ranks = pd.Series(abs_v).rank(ascending=False, method="first").astype(int).to_numpy()
        for fi, col in enumerate(FEATURE_ORDER_73):
            meta = REGISTRY[col]
            raw_v = prepared_t5.iloc[ti][col] if col in prepared_t5.columns else (t5_raw.iloc[ti][col] if col in t5_raw.columns else None)
            t5_field_rows.append({
                "EntityId": int(eid),
                "source_column": col,
                "label_ru": meta.label_ru,
                "kind": meta.kind,
                "kind_label": meta.kind_label,
                "last_step": meta.last_step,
                "raw_hint": meta.raw_hint,
                "value_str": clean_display_value(raw_v, 160),
                "is_missing": bool(raw_v is None or (isinstance(raw_v, float) and np.isnan(raw_v)) or str(raw_v).strip() in ("", "None")),
                "shap_val": float(s_vec[fi]),
                "abs_shap": float(abs_v[fi]),
                "share_abs_shap": float(abs_v[fi] / tot_abs),
                "rank_abs": int(ranks[fi]),
            })
    team5_fields_df = pd.DataFrame(t5_field_rows)

    # Детализация на уровне 1099 фич для 8 карточек (для переключателя «73 поля / 1099 фич»)
    fm = advisor.feature_mapping
    t5_1099_rows = []
    for ti, eid in enumerate(team5_nav["EntityId"].to_numpy()):
        sv = expl_t5["shap_1099"][ti]
        # Сохраняем топ-60 ненулевых модельных фич на карточку (остальные суммируются в «Прочие»)
        top_idx = np.argsort(-np.abs(sv))[:60]
        for rank_i, fi in enumerate(top_idx, start=1):
            row_fm = fm.iloc[fi]
            t5_1099_rows.append({
                "EntityId": int(eid),
                "rank_abs": rank_i,
                "feature_index": int(fi),
                "feature": str(row_fm["feature"]),
                "source_column": str(row_fm["source_column"]),
                "last_step": str(row_fm["last_step"]),
                "shap_val": float(sv[fi]),
                "abs_shap": float(abs(sv[fi])),
            })
    team5_1099_df = pd.DataFrame(t5_1099_rows)

    # 320 карточек практики (из официального shap_cards.csv + навигация)
    shap_csv_df = pd.read_csv(advisor.pack_dir / "shap_cards.csv", sep=";", encoding="utf-8-sig")
    shap_csv_df["EntityId"] = pd.to_numeric(shap_csv_df["EntityId"]).astype("int64")
    practice_320_df = practice_df[["team", "rank_in_team", "EntityId", "created", "flag"]].merge(
        shap_csv_df, on="EntityId", how="inner"
    )

    # 5. Глобальная важность (Приёмка / Эксплуатация / Практика 320) + Ablation + Beeswarm
    print("Считаю глобальную важность, Ablation и выборку Beeswarm...")
    m_oper = parts_all == "Эксплуатация"
    rng = np.random.default_rng(42)
    idx_oper = np.flatnonzero(m_oper)
    idx_test = np.flatnonzero(m_test)
    ab_idx = rng.choice(idx_oper, size=min(1500, len(idx_oper)), replace=False)
    ablation_map = compute_ablation(advisor, X_all[ab_idx])

    imp_rows = []
    for sample_name, s_mat in [
        ("Приёмка", shap_73_all[m_test]),
        ("Эксплуатация", shap_73_all[m_oper]),
        ("Практика 320", practice_320_df[FEATURE_ORDER_73].to_numpy(dtype=np.float64)),
    ]:
        m_abs = np.abs(s_mat).mean(axis=0)
        m_signed = s_mat.mean(axis=0)
        tot = float(m_abs.sum()) or 1.0
        ranks = pd.Series(m_abs).rank(ascending=False, method="first").astype(int).to_numpy()
        for fi, col in enumerate(FEATURE_ORDER_73):
            imp_rows.append({
                "sample": sample_name,
                "source_column": col,
                "label_ru": REGISTRY[col].label_ru,
                "kind": REGISTRY[col].kind,
                "last_step": REGISTRY[col].last_step,
                "mean_abs_shap": float(m_abs[fi]),
                "mean_shap": float(m_signed[fi]),
                "share_abs_shap": float(m_abs[fi] / tot),
                "rank": int(ranks[fi]),
                "ablation_delta_proba": float(ablation_map[col]),
            })
    global_imp_df = pd.DataFrame(imp_rows)

    # Стратифицированная выборка для Beeswarm (1500 Приёмка + 1500 Эксплуатация)
    bw_parts = []
    for pname, pool_idx in [("Приёмка", idx_test), ("Эксплуатация", idx_oper)]:
        take = rng.choice(pool_idx, size=min(1500, len(pool_idx)), replace=False)
        sub = pd.DataFrame(shap_73_all[take], columns=FEATURE_ORDER_73)
        sub.insert(0, "EntityId", eids_all[take])
        sub.insert(1, "part", pname)
        sub.insert(2, "y", y_all[take])
        sub.insert(3, "proba", proba_all[take])
        # Нормированные значения числовых признаков [0, 1] для цветовой шкалы Beeswarm
        for col in NUMERIC_FEATURES_43:
            v = pd.to_numeric(prepared_all.iloc[take][col], errors="coerce").to_numpy(dtype=float)
            sub[f"val__{col}"] = pd.Series(v).rank(pct=True).fillna(0.5).to_numpy(dtype=np.float32)
        for col in FEATURE_ORDER_73:
            if col not in NUMERIC_FEATURES_43:
                s = prepared_all.iloc[take][col] if col in prepared_all.columns else pd.Series([None] * len(take))
                sub[f"val__{col}"] = s.notna().astype(np.float32).to_numpy()
        bw_parts.append(sub)
    beeswarm_df = pd.concat(bw_parts, ignore_index=True)

    # 6. ALE / PDP / ICE (43 числовых) и group-PDP (12 категориальных)
    print("Считаю ALE, PDP, ICE (43 числовых) и group-PDP (12 категориальных)...")
    t_sub_idx = rng.choice(idx_test, size=min(1200, len(idx_test)), replace=False)
    o_sub_idx = rng.choice(idx_oper, size=min(1500, len(idx_oper)), replace=False)

    curves_t, _ = compute_ale_pdp_ice(
        advisor, prepared_all.iloc[t_sub_idx].reset_index(drop=True), X_all[t_sub_idx],
        prepared_t5, X_t5, team5_nav["EntityId"].to_numpy(), period_label="Приёмка",
    )
    curves_o, ice_t5_df = compute_ale_pdp_ice(
        advisor, prepared_all.iloc[o_sub_idx].reset_index(drop=True), X_all[o_sub_idx],
        prepared_t5, X_t5, team5_nav["EntityId"].to_numpy(), period_label="Эксплуатация",
    )
    curves_num_df = pd.concat([curves_t, curves_o], ignore_index=True)

    gpdp_t = compute_group_pdp_categorical(advisor, X_all[t_sub_idx[:800]], "Приёмка")
    gpdp_o = compute_group_pdp_categorical(advisor, X_all[o_sub_idx[:1000]], "Эксплуатация")
    curves_cat_df = pd.concat([gpdp_t, gpdp_o], ignore_index=True)

    # 7. Дрейф «Приёмка -> Эксплуатация» (PSI, пропуски, ΔSHAP, бины распределений)
    print("Считаю дрейф признаков Приёмка -> Эксплуатация...")
    drift_rows, drift_bin_rows = [], []
    s_val_all = shap_73_all[m_test]
    s_oper_all = shap_73_all[m_oper]
    for fi, col in enumerate(FEATURE_ORDER_73):
        meta = REGISTRY[col]
        if col in prepared_all.columns:
            r_ser = prepared_all.loc[m_test, col]
            c_ser = prepared_all.loc[m_oper, col]
            null_val = float(r_ser.map(lambda v: pd.isna(v) or str(v).strip() in ("", "None")).mean())
            null_oper = float(c_ser.map(lambda v: pd.isna(v) or str(v).strip() in ("", "None")).mean())
            if meta.last_step == "QuantileTransformer":
                psi, bins = compute_psi_numeric(
                    pd.to_numeric(r_ser, errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(c_ser, errors="coerce").to_numpy(dtype=float),
                )
            else:
                psi, bins = compute_psi_categorical(r_ser, c_ser, top_k=10)
        else:
            null_val, null_oper, psi, bins = 0.0, 0.0, 0.0, []

        m_abs_v = float(np.abs(s_val_all[:, fi]).mean())
        m_abs_o = float(np.abs(s_oper_all[:, fi]).mean())
        m_sgn_v = float(s_val_all[:, fi].mean())
        m_sgn_o = float(s_oper_all[:, fi].mean())
        drift_rows.append({
            "source_column": col,
            "label_ru": meta.label_ru,
            "kind": meta.kind,
            "last_step": meta.last_step,
            "psi": psi,
            "null_val": null_val,
            "null_oper": null_oper,
            "delta_null": null_oper - null_val,
            "mean_abs_shap_val": m_abs_v,
            "mean_abs_shap_oper": m_abs_o,
            "delta_abs_shap": m_abs_o - m_abs_v,
            "mean_shap_val": m_sgn_v,
            "mean_shap_oper": m_sgn_o,
            "delta_shap": m_sgn_o - m_sgn_v,
        })
        for b in bins:
            drift_bin_rows.append({"source_column": col, "last_step": meta.last_step, **b})
    drift_summary_df = pd.DataFrame(drift_rows)
    drift_bins_df = pd.DataFrame(drift_bin_rows)

    # 8. Масштабируемая 2D-карта похожести (Hexbin + кластеры + 20-NN)
    print("Строю 2D-карты похожести (SHAP-PCA, SVD-PCA, t-SNE) и 20-NN...")
    meta_all = pd.DataFrame({
        "EntityId": eids_all, "part": parts_all, "y": y_all, "proba": proba_all, "rule": rules_all
    })
    map_tables, axis_meta = build_similarity_maps(
        shap_73_all=shap_73_all, X_all=X_all, meta_all=meta_all,
        team5_ids=team5_nav["EntityId"].to_numpy(), tau=tau, run_tsne=not skip_tsne,
    )

    # 9. Таблица верификации объяснений (95 % ДИ Уилсона для топ-12 полей карточек team=5)
    print("Считаю статистику верификации (95 % ДИ Уилсона) для карточек команды 5...")
    verif_df = build_verification_stats(
        team5_fields=team5_fields_df, prepared_all=prepared_all, y_all=y_all,
        parts_all=parts_all, prepared_t5=prepared_t5, team5_ids=team5_nav["EntityId"].to_numpy(),
    )

    # 10. Сохранение всех артефактов
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    registry_frame().to_parquet(AGG_DIR / "feature_registry.parquet", index=False)
    team5_cards.to_parquet(AGG_DIR / "team5_cards.parquet", index=False)
    team5_fields_df.to_parquet(AGG_DIR / "team5_field_values.parquet", index=False)
    team5_1099_df.to_parquet(AGG_DIR / "team5_shap_1099.parquet", index=False)
    practice_320_df.to_parquet(AGG_DIR / "practice_320_shap.parquet", index=False)
    global_imp_df.to_parquet(AGG_DIR / "global_importance.parquet", index=False)
    beeswarm_df.to_parquet(AGG_DIR / "beeswarm_sample.parquet", index=False)
    curves_num_df.to_parquet(AGG_DIR / "curves_numeric.parquet", index=False)
    ice_t5_df.to_parquet(AGG_DIR / "curves_ice_team5.parquet", index=False)
    curves_cat_df.to_parquet(AGG_DIR / "curves_cat_gpdp.parquet", index=False)
    drift_summary_df.to_parquet(AGG_DIR / "drift_summary.parquet", index=False)
    drift_bins_df.to_parquet(AGG_DIR / "drift_bins.parquet", index=False)
    verif_df.to_parquet(AGG_DIR / "verification_team5.parquet", index=False)
    for name, df in map_tables.items():
        df.to_parquet(AGG_DIR / f"{name}.parquet", index=False)

    # Считаем статистику обогащения и скора для шапки/плиток
    enr_val = float((all_scope.loc[m_test, "Вендор из сием"].notna() & all_scope.loc[m_test, "Алерт"].notna()).mean())
    enr_oper = float((all_scope.loc[m_oper, "Вендор из сием"].notna() & all_scope.loc[m_oper, "Алерт"].notna()).mean())

    meta = {
        "team": int(team),
        "practice_source": practice_src,
        "dataset": raw_path.name,
        "total_dataset_rows": total_dataset_rows,
        "scope_rows": int(len(all_scope)),
        "counts": {
            "train": int(len(train_df)), "val": int(len(val_df)),
            "test": int(len(test_df)), "oper": int(len(oper_df)),
            "practice_total": int(len(practice_df)), "team_cards": int(len(team5_nav)),
            "features_model": int(len(advisor.feature_mapping)), "features_fields": len(FEATURE_ORDER_73),
        },
        "model": "lgbm_gate1__v001",
        "run_id": advisor.run["run_id"],
        "engine": advisor.engine,
        "threshold": tau,
        "logit_tau": logit_tau,
        "base_value": float(expl_t5["base"][0]),
        "checks": {
            "control_cards_delta": delta_ctrl,
            "published_test_match": pub_match,
            "team_outcome_match": outcome_match,
            "team_max_proba_diff": max_t5_proba_diff,
            "all_scope_additivity_diff": max_add_all,
            **pack_check,
        },
        "periods": {
            "train": [str(WINDOW[0].date()), str((WINDOW[1] - pd.Timedelta(days=1)).date())],
            "oper": [str(OPER_START.date()), str(created_raw.max().date())],
        },
        "enrichment": {"val": enr_val, "oper": enr_oper},
        "mean_raw_score": {
            "val": float(expl_all["raw_score"][m_test].mean()),
            "oper": float(expl_all["raw_score"][m_oper].mean()),
        },
        "map_axes": axis_meta,
    }
    (AGG_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 11. Печать диагностики по 8 карточкам команды 5 (для выбора кейсов этапа 2)
    print("\n" + "=" * 88)
    print(f"ДИАГНОСТИКА 8 КАРТОЧЕК КОМАНДЫ team = {team} (для выбора кейсов «Подтверждено» и «Сомнительно»)")
    print("=" * 88)
    nbr_shap = map_tables["neighbors_team5"].query("space == 'shap'")
    v_oper = verif_df.query("period == 'Эксплуатация'")

    for r in team5_cards.itertuples(index=False):
        eid = int(r.EntityId)
        f_top = team5_fields_df.query("EntityId == @eid").sort_values("abs_shap", ascending=False).head(5)
        n_sub = nbr_shap.query("target_id == @eid")
        v_sub = v_oper.query("EntityId == @eid")
        n_ok = int((v_sub["status"] == "Подтверждено").sum())
        n_bad = int((v_sub["status"] == "Противоречие").sum())
        inc_nbr = float(n_sub["y_true"].mean()) if len(n_sub) else np.nan
        print(
            f"\n#{r.rank_in_team} EntityId={eid} | {r.outcome} (y={r.y_true}, p={r.proba:.6f}, "
            f"запас={r.margin_to_tau:+.2f}) | правило: {r.rule}"
        )
        print(f"   Вендор: {r.vendor} | 20-NN(SHAP): доля инцидентов={inc_nbr:.0%}, исходы={n_sub['outcome_at_tau'].value_counts().to_dict()}")
        print(f"   Верификация (топ-12 в проде): Подтверждено={n_ok}, Противоречие={n_bad}")
        for fr in f_top.itertuples(index=False):
            tag = "[ENG]" if fr.kind == "engineered" else "[CARD]"
            print(f"     {fr.rank_abs:2d}. {tag:6s} {fr.label_ru} = {fr.value_str[:55]}  ->  SHAP {fr.shap_val:+.3f} ({fr.share_abs_shap:.0%})")

    print(f"\nГотово за {time.time() - t0:.1f} с. Артефакты сохранены в {AGG_DIR.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=5, help="номер команды в 3.7_Пр_cards_practice.xlsx")
    ap.add_argument("--path", help="путь к iavd_ml_dataset.parquet")
    ap.add_argument("--course-dir", help="путь к папке IAVD_3.7_ML_team")
    ap.add_argument("--xlsx", help="путь к 3.7_Пр_cards_practice.xlsx")
    ap.add_argument("--skip-tsne", action="store_true", help="пропустить расчёт t-SNE")
    args = ap.parse_args()
    main(find_raw(args.path), find_course_dir(args.course_dir), args.team, args.xlsx, args.skip_tsne)