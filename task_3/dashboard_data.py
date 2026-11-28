"""
dashboard_data.py — загрузка предрассчитанных артефактов task_3/data/agg/,
динамический пересчёт исходов при сдвиге порога τ, срез Top-N и поиск похожих карточек.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import FEATURE_ORDER_73, REGISTRY

AGG_DIR = Path(__file__).resolve().parent / "data" / "agg"
DEFAULT_TAU = 0.0193

OUTCOME_RU = {
    "TP": "TP · инцидент найден",
    "FN": "FN · пропуск инцидента",
    "FP": "FP · лишняя проверка",
    "TN": "TN · шум отсеян",
}

PRESETS: dict[str, float | None] = {
    "Рабочий порог курса τ = 0,0193": 0.0193,
    "Мягкий порог τ = 0,0050": 0.0050,
    "Порог 0,0350 (отсекает оба FP команды 5)": 0.0350,
    "Порог 0,5000": 0.5000,
    "Свой порог (слайдер)": None,
}

# Кураторский список кейсов для Страницы 5 (подтверждено / сомнительно / пограничный)
CURATED_CASES: dict[str, dict] = {
    "Подтверждённый кейс · #7 · 521996 (TP)": {
        "EntityId": 521996,
        "verdict_tag": "Подтверждается данными",
        "action": "Эталон: сохранять признаки внешнего IP и алерта сканера",
        "why_case": "9 из 12 топ-признаков подтверждены 95 % ДИ Уилсона; 85 % из 20-NN — реальные инциденты.",
    },
    "Сомнительный кейс (пропуски) · #4 · 526930 (FP)": {
        "EntityId": 526930,
        "verdict_tag": "Выглядит сомнительно",
        "action": "Данные + порог: вернуть обогащение («Алерт», «Вендор», «Команда») или поднять τ ≥ 0,035",
        "why_case": "Все 4 главных вклада в пользу инцидента — пропуски/нули; среди 20-NN — 0 % инцидентов (14 FP, 6 TN).",
    },
    "Сомнительный кейс (слепое пятно) · #1 · 523480 (FN)": {
        "EntityId": 523480,
        "verdict_tag": "Выглядит сомнительно (пропуск)",
        "action": "Данные / переобучение: сдвиг порога не поможет (запас −7,96 log-odds)",
        "why_case": "Алерт «Anti-virus было отключено» тянет модель к FP (−1,193); все 20-NN — TN.",
    },
    "Сомнительный кейс (противоречия) · #2 · 523796 (FN)": {
        "EntityId": 523796,
        "verdict_tag": "Выглядит сомнительно (пропуск)",
        "action": "Данные + признак: заполнить «Вендор», снизить перевес локального IP над ProcessPath",
        "why_case": "3 статистических противоречия в топ-12 (ДИ Уилсона), пропущен вендор, запас −7,67 log-odds.",
    },
    "Пограничный случай · #3 · 522090 (FP)": {
        "EntityId": 522090,
        "verdict_tag": "Пограничный FP у порога",
        "action": "Порог: запас +0,43 log-odds, уходит в TN при τ ≥ 0,035 (сигнал Trojan правдоподобен)",
        "why_case": "Карточка с реальным алертом Kaspersky (DllHijack), 30 % из 20-NN — реальные инциденты.",
    },
}


# ---------------------------------------------------------------- форматирование и математика
def fmt_int(v: float | int) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{int(round(v)):,}".replace(",", " ")


def fmt_num(v: float, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}".replace(".", ",")


def fmt_signed(v: float, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    sign = "+" if v > 0 else ("−" if v < 0 else "")
    return f"{sign}{abs(v):.{digits}f}".replace(".", ",")


def fmt_pct(v: float, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v * 100:.{digits}f} %".replace(".", ",")


def fmt_proba(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.0001:
        return f"{p * 100:.4f} %".replace(".", ",")
    if p < 0.01:
        return f"{p * 100:.3f} %".replace(".", ",")
    return f"{p * 100:.2f} %".replace(".", ",")


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p / (1.0 - p)))


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    res = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))
    return float(res) if np.ndim(z) == 0 else res


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def outcome_of(y_true: int, proba: float, tau: float) -> str:
    pred = proba >= tau
    if y_true == 1:
        return "TP" if pred else "FN"
    return "FP" if pred else "TN"


def confusion_from_proba(y: np.ndarray, proba: np.ndarray, tau: float) -> dict[str, int]:
    pred = proba >= tau
    y = np.asarray(y, dtype=int)
    return dict(
        tp=int((pred & (y == 1)).sum()),
        fn=int((~pred & (y == 1)).sum()),
        fp=int((pred & (y == 0)).sum()),
        tn=int((~pred & (y == 0)).sum()),
        n=int(len(y)),
    )


# ---------------------------------------------------------------- хранилище данных
@dataclass
class Store:
    meta: dict
    registry: pd.DataFrame
    team5_cards: pd.DataFrame
    team5_fields: pd.DataFrame
    team5_1099: pd.DataFrame
    practice_320: pd.DataFrame
    global_imp: pd.DataFrame
    beeswarm: pd.DataFrame
    curves_num: pd.DataFrame
    curves_ice: pd.DataFrame
    curves_cat: pd.DataFrame
    drift_summary: pd.DataFrame
    drift_bins: pd.DataFrame
    hex_shap: pd.DataFrame
    hex_pca: pd.DataFrame
    hex_tsne: pd.DataFrame
    clusters: pd.DataFrame
    focus_team5: pd.DataFrame
    neighbors_team5: pd.DataFrame
    verification: pd.DataFrame
    # Предсобранный пул карточек с 73 SHAP для быстрого Violin по группе похожих
    shap_pool_mat: np.ndarray
    shap_pool_meta: pd.DataFrame
    points_tau: pd.DataFrame | None = None
    corr_numeric: pd.DataFrame | None = None
    numeric_dist: pd.DataFrame | None = None


def load(agg_dir: Path = AGG_DIR) -> Store:
    if not (agg_dir / "meta.json").exists():
        raise SystemExit(f"Нет данных в {agg_dir}. Сначала запустите: python prepare.py")

    meta = json.loads((agg_dir / "meta.json").read_text(encoding="utf-8"))
    read = lambda name: pd.read_parquet(agg_dir / f"{name}.parquet")

    practice_320 = read("practice_320_shap")
    beeswarm = read("beeswarm_sample")

    # Объединяем practice_320 и beeswarm (без дублей по EntityId) для быстрого 40-NN Violin
    p_sub = practice_320[["EntityId", "y_true", "proba", "outcome"] + FEATURE_ORDER_73].copy()
    p_sub["source"] = "practice_320"
    b_sub = beeswarm[["EntityId", "y", "proba"] + FEATURE_ORDER_73].rename(columns={"y": "y_true"}).copy()
    b_sub["outcome"] = [
        outcome_of(int(y), float(p), meta["threshold"]) if y >= 0 else "—"
        for y, p in zip(b_sub["y_true"], b_sub["proba"])
    ]
    b_sub["source"] = beeswarm["part"].to_numpy()
    pool = pd.concat([p_sub, b_sub], ignore_index=True).drop_duplicates(subset=["EntityId"], keep="first").reset_index(drop=True)
    pool_mat = pool[FEATURE_ORDER_73].to_numpy(dtype=np.float64)
    pool_meta = pool[["EntityId", "y_true", "proba", "outcome", "source"]].copy()
    
    read_opt = lambda name: pd.read_parquet(agg_dir / f"{name}.parquet") if (agg_dir / f"{name}.parquet").exists() else None
    points_tau = read_opt("map_points_tau")
    corr_numeric = read_opt("corr_numeric")
    numeric_dist = read_opt("numeric_dist_sample")
    hex_shap = read("map_hex_shap")
    if points_tau is None or "hq" not in hex_shap.columns:
        print("[WARN] Живой пересчёт Hexbin при смене τ неактивен: перезапустите prepare.py "
              f"(points_tau={'ok' if points_tau is not None else 'missing'}, hq_in_hex={'hq' in hex_shap.columns})")

    return Store(
        meta=meta,
        registry=read("feature_registry"),
        team5_cards=read("team5_cards"),
        team5_fields=read("team5_field_values"),
        team5_1099=read("team5_shap_1099"),
        practice_320=practice_320,
        global_imp=read("global_importance"),
        beeswarm=beeswarm,
        curves_num=read("curves_numeric"),
        curves_ice=read("curves_ice_team5"),
        curves_cat=read("curves_cat_gpdp"),
        drift_summary=read("drift_summary"),
        drift_bins=read("drift_bins"),
        hex_shap=read("map_hex_shap"),
        hex_pca=read("map_hex_pca"),
        hex_tsne=read("map_hex_tsne"),
        clusters=read("map_clusters"),
        focus_team5=read("map_focus_team5"),
        neighbors_team5=read("neighbors_team5"),
        verification=read("verification_team5"),
        shap_pool_mat=pool_mat,
        shap_pool_meta=pool_meta,
        points_tau=points_tau,
        corr_numeric=corr_numeric,
        numeric_dist=numeric_dist,
    )


def hex_at_threshold(store: Store, space_key: str, tau: float) -> tuple[pd.DataFrame, bool]:
    """Пересчитывает pred_rate и err_rate для гекс-ячеек при текущем τ (если есть map_points_tau.parquet)."""
    base_hx = {"shap": store.hex_shap, "pca": store.hex_pca, "tsne": store.hex_tsne}[space_key]
    if base_hx.empty or store.points_tau is None or "hq" not in base_hx.columns:
        return base_hx, False

    hq_col, hr_col = f"hq_{space_key}", f"hr_{space_key}"
    pts = store.points_tau
    if hq_col not in pts.columns:
        return base_hx, False

    valid = pts[hq_col].to_numpy() > -10**8
    sub = pts.loc[valid, [hq_col, hr_col, "y", "proba"]].copy()
    sub["pred"] = (sub["proba"].to_numpy() >= tau).astype(np.int8)
    y = sub["y"].to_numpy()
    sub["is_lab"] = (y >= 0).astype(np.int8)
    sub["is_err"] = ((y >= 0) & (sub["pred"].to_numpy() != y)).astype(np.int8)

    agg = sub.groupby([hq_col, hr_col], sort=False).agg(
        n_tot=("pred", "size"),
        n_pred=("pred", "sum"),
        n_lab=("is_lab", "sum"),
        n_err=("is_err", "sum"),
    ).reset_index().rename(columns={hq_col: "hq", hr_col: "hr"})

    agg["pred_rate_dyn"] = agg["n_pred"] / np.maximum(agg["n_tot"], 1)
    agg["err_rate_dyn"] = np.where(agg["n_lab"] > 0, agg["n_err"] / np.maximum(agg["n_lab"], 1), np.nan)

    merged = base_hx.merge(agg[["hq", "hr", "pred_rate_dyn", "err_rate_dyn"]], on=["hq", "hr"], how="left")
    merged["pred_rate"] = merged["pred_rate_dyn"].fillna(merged["pred_rate"])
    merged["err_rate"] = merged["err_rate_dyn"].fillna(merged["err_rate"])
    return merged.drop(columns=["pred_rate_dyn", "err_rate_dyn"]), True


# ---------------------------------------------------------------- хелперы для пагинации дрейфа
DRIFT_PAGE_SIZE = 10


def drift_filtered(store: Store, sort_by: str, only_high_psi: bool) -> pd.DataFrame:
    ds = store.drift_summary.copy()
    if only_high_psi:
        ds = ds[ds["psi"] >= 0.25].copy()
    if sort_by == "По PSI":
        ds = ds.sort_values("psi", ascending=False)
    elif sort_by == "По росту пропусков":
        ds = ds.sort_values("delta_null", ascending=False)
    else:
        ds["abs_d"] = ds["delta_abs_shap"].abs()
        ds = ds.sort_values("abs_d", ascending=False)
    return ds.reset_index(drop=True)


def drift_page_options(total_rows: int, page_size: int = DRIFT_PAGE_SIZE) -> list[str]:
    if total_rows <= 0:
        return ["Стр. 1 из 1 (0 полей)"]
    n_pages = int(np.ceil(total_rows / page_size))
    opts = []
    for p in range(n_pages):
        lo = p * page_size + 1
        hi = min((p + 1) * page_size, total_rows)
        opts.append(f"Стр. {p + 1} из {n_pages} ({lo}–{hi} из {total_rows})")
    return opts


def parse_page_idx(label: str) -> int:
    try:
        return max(0, int(str(label).split()[1]) - 1)
    except (IndexError, ValueError):
        return 0


# ---------------------------------------------------------------- селекторы и срезы
def card_label(row: pd.Series, tau: float = DEFAULT_TAU) -> str:
    out = outcome_of(int(row["y_true"]), float(row["proba_calc"]), tau)
    return f"#{int(row['rank_in_team'])} · {int(row['EntityId'])} · {out} · p={fmt_proba(float(row['proba_calc']))}"


def card_options(store: Store, outcome_filter: str = "Все (8)", tau: float = DEFAULT_TAU) -> list[str]:
    df = store.team5_cards
    opts = []
    for _, r in df.iterrows():
        out = outcome_of(int(r["y_true"]), float(r["proba_calc"]), tau)
        if outcome_filter.startswith("Все") or out == outcome_filter:
            opts.append(card_label(r, tau))
    return opts or [card_label(r, tau) for _, r in df.iterrows()]


def parse_entity_id(label: str, fallback: int) -> int:
    if not label:
        return fallback
    for tok in str(label).split("·"):
        tok = tok.strip()
        if tok.isdigit() and len(tok) >= 5:
            return int(tok)
    return fallback


def card_row(store: Store, eid: int) -> pd.Series:
    sub = store.team5_cards[store.team5_cards["EntityId"] == eid]
    return sub.iloc[0] if len(sub) else store.team5_cards.iloc[0]


def slice_top_n(
    store: Store,
    eid: int,
    top_n: int = 10,
    detail: str = "73 поля SOC",
) -> dict:
    """Формирует срез Top-N для Waterfall и полосы баланса с точным сохранением аддитивности."""
    row = card_row(store, eid)
    base_val = float(row["base"])
    raw_score = float(row["raw_score"])
    sum_total = float(row["sum_shap"])
    top_n = int(np.clip(top_n, 5, 20))

    if detail.startswith("73"):
        df = store.team5_fields[store.team5_fields["EntityId"] == eid].sort_values("abs_shap", ascending=False).reset_index(drop=True)
        top = df.head(top_n).copy()
        rest = df.iloc[top_n:]
        other_val = float(rest["shap_val"].sum())
        other_label = f"Прочие ({len(rest)} полей)"
        total_abs = float(df["abs_shap"].sum()) or 1.0
        top["display_name"] = [
            f"{r.label_ru} {'[eng]' if r.kind == 'engineered' else ''}".strip()
            for r in top.itertuples(index=False)
        ]
        top["hover_val"] = top["value_str"]
        sum_pos_all = float(np.clip(df["shap_val"], 0, None).sum())
        sum_neg_all = float(np.clip(df["shap_val"], None, 0).sum())
    else:
        df = store.team5_1099[store.team5_1099["EntityId"] == eid].sort_values("abs_shap", ascending=False).reset_index(drop=True)
        top = df.head(top_n).copy()
        other_val = float(sum_total - top["shap_val"].sum())
        other_label = f"Прочие ({1099 - len(top)} фич модели)"
        total_abs = float(df["abs_shap"].sum()) or 1.0
        top["display_name"] = top["feature"].astype(str)
        top["hover_val"] = top["source_column"].map(lambda c: f"поле: {REGISTRY[c].label_ru}" if c in REGISTRY else str(c))
        sum_pos_all = float(row["sum_pos"])
        sum_neg_all = float(row["sum_neg"])

    sum_top_net = float(top["shap_val"].sum())
    sum_top_pos = float(np.clip(top["shap_val"], 0, None).sum())
    sum_top_neg = float(np.clip(top["shap_val"], None, 0).sum())
    coverage = float(top["abs_shap"].sum() / total_abs)

    return {
        "eid": int(eid),
        "base": base_val,
        "raw_score": raw_score,
        "proba": float(row["proba_calc"]),
        "top_df": top,
        "top_n": top_n,
        "other_val": other_val,
        "other_label": other_label,
        "sum_top_net": sum_top_net,
        "sum_top_pos": sum_top_pos,
        "sum_top_neg": sum_top_neg,
        "sum_all_net": sum_total,
        "sum_all_pos": sum_pos_all,
        "sum_all_neg": sum_neg_all,
        "coverage": min(1.0, coverage),
        "additivity_diff": abs(float(sigmoid(base_val + sum_top_net + other_val)) - float(row["proba_calc"])),
    }


def build_card_table(store: Store, eid: int, top_n: int = 10, only_top: bool = False) -> pd.DataFrame:
    df = store.team5_fields[store.team5_fields["EntityId"] == eid].sort_values("rank_abs")
    if only_top:
        df = df.head(top_n)
    return pd.DataFrame({
        "№": df["rank_abs"].astype(int),
        "Поле": df["label_ru"],
        "Тип": df["kind_label"],
        "Значение в карточке": df["value_str"],
        "SHAP (log-odds)": df["shap_val"].map(lambda v: fmt_signed(v, 3)),
        "Доля |SHAP|": df["share_abs_shap"].map(lambda v: fmt_pct(v, 1)),
        "Направление": np.where(df["shap_val"] > 0, "→ инцидент", np.where(df["shap_val"] < 0, "→ FP", "нейтрально")),
    }).reset_index(drop=True)


def similar_group_shap(store: Store, eid: int, mode: str = "40 ближайших по SHAP", top_k_feats: int = 8) -> dict:
    """Возвращает матрицу SHAP группы похожих карточек для Violin/Raincloud по топ-8 полям текущей карточки."""
    card_f = store.team5_fields[store.team5_fields["EntityId"] == eid].sort_values("rank_abs")
    top_cols = card_f.head(top_k_feats)["source_column"].tolist()
    top_labels = [REGISTRY[c].label_ru for c in top_cols]
    card_vals = card_f.set_index("source_column").loc[top_cols, "shap_val"].to_numpy(dtype=float)

    # Вектор 73 SHAP текущей карточки
    full_vec = card_f.set_index("source_column").loc[FEATURE_ORDER_73, "shap_val"].to_numpy(dtype=float)
    meta = store.shap_pool_meta
    mat = store.shap_pool_mat

    if mode.startswith("40"):
        norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(full_vec) or 1.0)
        sim = (mat @ full_vec) / np.where(norms > 1e-12, norms, 1.0)
        sim[meta["EntityId"].to_numpy() == eid] = -2.0
        idx = np.argsort(-sim)[:40]
        group_desc = "40 ближайших карточек по косинусному сходству SHAP"
    elif mode.startswith("Тот же исход"):
        out = str(card_row(store, eid)["outcome"])
        idx = np.flatnonzero((meta["source"] == "practice_320") & (meta["outcome"] == out))
        group_desc = f"Все карточки практики с исходом {out} ({len(idx)} шт.)"
    else:
        idx = np.flatnonzero(meta["source"] == "practice_320")
        group_desc = f"Весь срез практики ({len(idx)} карточек)"

    col_pos = [FEATURE_ORDER_73.index(c) for c in top_cols]
    return {
        "cols": top_cols,
        "labels": top_labels,
        "card_vals": card_vals,
        "group_mat": mat[np.ix_(idx, col_pos)],
        "group_y": meta.iloc[idx]["y_true"].to_numpy(dtype=int),
        "group_desc": group_desc,
    }


def neighbors_table(store: Store, eid: int, space: str = "shap", tau: float = DEFAULT_TAU) -> pd.DataFrame:
    sp = "pca" if ("SVD" in space or "Признаки" in space or space == "pca") else "shap"
    df = store.neighbors_team5[(store.neighbors_team5["target_id"] == eid) & (store.neighbors_team5["space"] == sp)].copy()
    if df.empty:
        return pd.DataFrame(columns=["№", "EntityId", "Сходство", "Выборка", "Факт", "Оценка", "Исход при τ", "Правило SIEM"])
    out_dyn = [outcome_of(int(y), float(p), tau) for y, p in zip(df["y_true"], df["proba"])]
    return pd.DataFrame({
        "№": df["neighbor_rank"].astype(int),
        "EntityId": df["neighbor_id"].astype(int),
        "Сходство": df["cosine_sim"].map(lambda v: fmt_num(v, 3)),
        "Выборка": df["part"],
        "Факт": np.where(df["y_true"] == 1, "Инцидент", "FP"),
        "Оценка": df["proba"].map(fmt_proba),
        "Исход при τ": out_dyn,
        "Правило SIEM": df["rule"],
    }).reset_index(drop=True)


def verification_table(store: Store, eid: int, period: str = "Эксплуатация") -> pd.DataFrame:
    df = store.verification[(store.verification["EntityId"] == eid) & (store.verification["period"] == period)].sort_values("rank_abs")
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "№": df["rank_abs"].astype(int),
        "Поле": df["label_ru"],
        "Значение / срез": df["slice_desc"],
        "SHAP": df["shap_val"].map(lambda v: fmt_signed(v, 3)),
        "Факт (k из n)": [f"{int(r.k)} из {fmt_int(r.n)}" for r in df.itertuples(index=False)],
        "Доля инцидентов": df["inc_rate"].map(lambda v: fmt_pct(v, 1)),
        "95 % ДИ Уилсона": [f"{fmt_pct(r.ci_lo, 1)} — {fmt_pct(r.ci_hi, 1)}" if np.isfinite(r.ci_lo) else "—" for r in df.itertuples(index=False)],
        "Статус": df["status"],
    }).reset_index(drop=True)


def case_recommendation(store: Store, eid: int, tau: float) -> tuple[str, str]:
    """Короткая рекомендация по § 3 методички (данные / порог / признак) для плитки на Странице 5."""
    for info in CURATED_CASES.values():
        if info["EntityId"] == eid:
            return info["action"].split(":", 1)[0], info["action"]
    row = card_row(store, eid)
    out = outcome_of(int(row["y_true"]), float(row["proba_calc"]), tau)
    margin = float(row["raw_score"]) - logit(tau)
    if out == "FN":
        return ("Порог" if abs(margin) < 1.0 else "Данные / обучение"), f"запас {fmt_signed(margin, 2)} log-odds"
    if out == "FP":
        return ("Порог" if abs(margin) < 1.0 else "Признак / данные"), f"запас {fmt_signed(margin, 2)} log-odds"
    return "Без изменений", "решение модели совпадает с фактом SOC"