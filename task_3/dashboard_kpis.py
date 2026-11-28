"""
dashboard_kpis.py — KPI-плитки всех 5 страниц дашборда задания 3 (в точном формате task_2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dashboard_data as dd
from feature_registry import REGISTRY

DASH = "—"
PSI_HIGH = 0.25
CORR_WARN = 0.70

TILES = {
    "card": [
        ("c_true", "Факт SOC (y_true)"),
        ("c_score", "Оценка и решение модели"),
        ("c_out", "Исход при пороге τ"),
        ("c_topn", "Вклад Top-N признаков"),
        ("c_add", "Аддитивность и запас до τ"),
    ],
    "similar": [
        ("s_inc", "Инцидентов среди 20-NN"),
        ("s_agree", "Совпадение вердикта 20-NN"),
        ("s_med", "Медианный скор 20-NN"),
        ("s_clst", "Кластер карточки"),
    ],
    "global": [
        ("g_top1", "Главный сигнал потока"),
        ("g_eng", "Доля сконструированных полей"),
        ("g_abl", "Согласие SHAP и Ablation"),
        ("g_corr", "Корреляция признака (PDP)"),
    ],
    "drift": [
        ("d_psi", "Полей с сильным дрейфом"),
        ("d_shap", "Макс. сдвиг важности SHAP"),
        ("d_enr", "Обогащение: заполнены «Вендор» и «Алерт»"),
        ("d_shift", "Сдвиг среднего скора"),
    ],
    "cases": [
        ("v_ok", "Подтверждено статистикой"),
        ("v_bad", "Противоречий (ДИ Уилсона)"),
        ("v_gap", "Запас до порога τ"),
        ("v_act", "Что делать (§ 3)"),
    ],
}
TEXT_TILES = {"c_true", "c_out", "s_clst", "g_top1", "d_shap", "v_act"}


def markup(page: str) -> str:
    cells = []
    for key, label in TILES[page]:
        value_cls = "kpi-value kpi-value-text" if key in TEXT_TILES else "kpi-value"
        cells.append(
            f"<|part|class_name={{{key}_c}}|\n"
            f"<|{label}|text|class_name=kpi-label|>\n"
            f"<|{{{key}_v}}|text|class_name={value_cls}|>\n"
            f"<|{{{key}_n}}|text|class_name=kpi-note|>\n|>"
        )
    return (
        f"<|layout|columns={' '.join(['1'] * len(cells))}|gap=12px|class_name=kpis|\n"
        + "\n".join(cells)
        + "\n|>"
    )


def _tile(key: str, value: str, note: str = "", alert: bool = False, kind: str = "") -> dict:
    cls = "kpi" + (f" {kind}" if kind else "") + (" kpi-alert" if alert else "")
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": cls}


def _short(s, n: int = 28) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- 1. Почему так? (Карточка)
def page_card(store: dd.Store, eid: int, tau: float, slc: dict) -> dict:
    r = dd.card_row(store, eid)
    y_true = int(r["y_true"])
    proba = float(r["proba_calc"])
    pred_inc = proba >= tau
    out = dd.outcome_of(y_true, proba, tau)
    margin = float(r["raw_score"]) - dd.logit(tau)

    fact_txt = "Инцидент (1)" if y_true == 1 else "Ложная тревога · FP (0)"
    verdict_txt = "в приоритет (инцидент)" if pred_inc else "отложить (FP)"
    tau_note = f"{verdict_txt} · оценка (калибровка не проверялась)"

    out_tiles = {}
    out_tiles |= _tile(
        "c_true", fact_txt,
        f"EntityId {eid} · #{int(r['rank_in_team'])} · флаг таблицы: {r['flag']}",
        kind="kpi-real" if y_true == 1 else "kpi-fp",
    )
    out_tiles |= _tile(
        "c_score", dd.fmt_proba(proba), tau_note,
        kind="kpi-real" if pred_inc else "kpi-fp",
    )
    out_tiles |= _tile(
        "c_out", dd.OUTCOME_RU[out],
        f"при τ = {dd.fmt_num(tau, 4)}" + (" (порог изменён)" if abs(tau - dd.DEFAULT_TAU) > 1e-6 else " (эталон курса)"),
        alert=(out in ("FN", "FP")),
        kind="kpi-real" if out == "TP" else ("kpi-fp" if out == "TN" else ""),
    )
    out_tiles |= _tile(
        "c_topn", dd.fmt_signed(slc["sum_top_net"], 2),
        f"Top-{slc['top_n']}: {dd.fmt_signed(slc['sum_top_pos'], 2)} / {dd.fmt_signed(slc['sum_top_neg'], 2)} · "
        f"покрытие {dd.fmt_pct(slc['coverage'], 0)} |SHAP|",
    )
    ok_add = slc["additivity_diff"] <= 1e-6
    out_tiles |= _tile(
        "c_add", dd.fmt_signed(margin, 2),
        f"запас до τ (log-odds) · Δ = {slc['additivity_diff']:.1e} {'✓' if ok_add else '✗'}",
        alert=not ok_add,
    )
    return out_tiles


# ---------------------------------------------------------------- 2. Похожие карточки
def page_similar(store: dd.Store, eid: int, space: str, tau: float) -> dict:
    sp = "pca" if ("SVD" in space or "Признаки" in space or space == "pca") else "shap"
    nbr = store.neighbors_team5[(store.neighbors_team5["target_id"] == eid) & (store.neighbors_team5["space"] == sp)]
    r = dd.card_row(store, eid)
    out_card = dd.outcome_of(int(r["y_true"]), float(r["proba_calc"]), tau)

    if nbr.empty:
        return {k: DASH for key, _ in TILES["similar"] for k in (f"{key}_v", f"{key}_n")}

    inc_rate = float(nbr["y_true"].mean())
    nbr_pred = (nbr["proba"].to_numpy() >= tau).astype(int)
    card_pred = int(float(r["proba_calc"]) >= tau)
    same_pred = float((nbr_pred == card_pred).mean())
    med_p = float(nbr["proba"].median())

    foc = store.focus_team5[store.focus_team5["EntityId"] == eid].iloc[0]
    cid = int(foc["cluster_pca"] if sp == "pca" else foc["cluster_shap"])
    cl_row = store.clusters[(store.clusters["space"] == sp) & (store.clusters["cluster"] == cid)]
    cl_note = f"инцидентов {dd.fmt_pct(float(cl_row['inc_rate'].iloc[0]), 0)} · {dd.fmt_int(int(cl_row['n'].iloc[0]))} карт." if len(cl_row) else ""
    cl_rule = _short(cl_row["top_rule"].iloc[0], 24) if len(cl_row) else f"Кластер #{cid}"

    # Тревожный сигнал: факт карточки противоречит окружению 20-NN
    mismatch = (int(r["y_true"]) == 1 and inc_rate < 0.25) or (int(r["y_true"]) == 0 and inc_rate > 0.50)

    out = _tile("s_inc", dd.fmt_pct(inc_rate, 0), f"{int(nbr['y_true'].sum())} из {len(nbr)} · у карточки y={int(r['y_true'])}", alert=mismatch, kind="kpi-real")
    out |= _tile("s_agree", dd.fmt_pct(same_pred, 0), f"вердикт модели как у карточки ({out_card})", kind="kpi-fp")
    out |= _tile("s_med", dd.fmt_proba(med_p), f"у выбранной карточки {dd.fmt_proba(float(r['proba_calc']))}")
    out |= _tile("s_clst", f"#{cid} · {cl_rule}", cl_note)
    return out


# ---------------------------------------------------------------- 3. Весь поток
def page_global(store: dd.Store, sample: str, num_feat: str, curve_period: str) -> dict:
    sub = store.global_imp[store.global_imp["sample"] == sample].sort_values("rank")
    top1 = sub.iloc[0]
    eng_share = float(sub.loc[sub["kind"] == "engineered", "share_abs_shap"].sum())
    top5_shap = set(sub.head(5)["source_column"])
    top5_abl = set(sub.sort_values("ablation_delta_proba", ascending=False).head(5)["source_column"])
    overlap = len(top5_shap & top5_abl)

    cn = store.curves_num[(store.curves_num["source_column"] == num_feat) & (store.curves_num["period"] == curve_period)]
    n_pairs = 0
    if store.corr_numeric is not None:
        cm = store.corr_numeric
        n_pairs = int(((cm["feat_a"] < cm["feat_b"]) & (cm["spearman"].abs() >= CORR_WARN)).sum())
    if len(cn):
        max_c = float(cn["max_spearman"].iloc[0])
        partner = str(cn["corr_partner"].iloc[0])
        warn = bool(cn["collinear_warn"].iloc[0])
        f_ru = _short(REGISTRY[num_feat].label_ru if num_feat in REGISTRY else num_feat, 20)
        p_ru = _short(REGISTRY[partner].label_ru if partner in REGISTRY else partner, 20)
        corr_v = f"|ρ| = {dd.fmt_num(max_c, 2)}"
        pairs_txt = f" · пар ≥0,7: {n_pairs}" if n_pairs else ""
        corr_n = f"«{f_ru}» ↔ «{p_ru}»{pairs_txt}"
    else:
        corr_v, corr_n, warn = DASH, "нет данных", False

    out = _tile("g_top1", _short(top1["label_ru"], 26), f"mean|SHAP| {dd.fmt_num(top1['mean_abs_shap'], 3)} ({dd.fmt_pct(top1['share_abs_shap'], 0)})", kind="kpi-real")
    out |= _tile("g_eng", dd.fmt_pct(eng_share, 0), "доля 43 engineered-признаков в сумме |SHAP|")
    out |= _tile("g_abl", f"{overlap} из 5", "совпадение топ-5 полей SHAP и Ablation", alert=(overlap < 2))
    out |= _tile("g_corr", corr_v, corr_n, alert=warn)
    return out


# ---------------------------------------------------------------- 4. Дрейф
def page_drift(store: dd.Store) -> dict:
    ds = store.drift_summary
    n_high = int((ds["psi"] >= PSI_HIGH).sum())
    worst_psi = ds.sort_values("psi", ascending=False).iloc[0]
    worst_sh = ds.reindex(ds["delta_abs_shap"].abs().sort_values(ascending=False).index).iloc[0]

    enr_v = store.meta["enrichment"]["val"]
    enr_o = store.meta["enrichment"]["oper"]
    ms_v = store.meta["mean_raw_score"]["val"]
    ms_o = store.meta["mean_raw_score"]["oper"]

    out = _tile("d_psi", dd.fmt_int(n_high), f"PSI ≥ 0,25 · лидер: {_short(worst_psi['label_ru'], 22)} ({dd.fmt_num(worst_psi['psi'], 2)})", alert=n_high > 0)
    out |= _tile("d_shap", _short(worst_sh["label_ru"], 26), f"Δmean|SHAP| {dd.fmt_signed(worst_sh['delta_abs_shap'], 3)}")
    out |= _tile("d_enr", dd.fmt_pct(enr_o, 0),
                 f"оба поля не пусты · на приёмке {dd.fmt_pct(enr_v, 0)}",
                 alert=(enr_o < enr_v * 0.75))
    out |= _tile("d_shift", dd.fmt_signed(ms_o - ms_v, 2), f"приёмка {dd.fmt_num(ms_v, 2)} → прод {dd.fmt_num(ms_o, 2)} (log-odds)")
    return out


# ---------------------------------------------------------------- 5. Проверка кейсов
def page_cases(store: dd.Store, eid: int, period: str, tau: float) -> dict:
    v = store.verification[(store.verification["EntityId"] == eid) & (store.verification["period"] == period)]
    r = dd.card_row(store, eid)
    margin = float(r["raw_score"]) - dd.logit(tau)

    n_ok = int((v["status"] == "Подтверждено").sum())
    n_bad = int((v["status"] == "Противоречие").sum())
    n_low = int((v["status"] == "Мало данных").sum())
    act_short, act_full = dd.case_recommendation(store, eid, tau)

    out = _tile("v_ok", f"{n_ok} из {len(v)}", f"знак SHAP совпадает с 95 % ДИ ({period.lower()})", kind="kpi-real")
    out |= _tile("v_bad", dd.fmt_int(n_bad), f"ещё {n_low} с малым n (< 10)", alert=(n_bad > 0))
    out |= _tile("v_gap", dd.fmt_signed(margin, 2), "порог поможет" if abs(margin) < 1.0 else "далеко от τ — порог не спасёт", alert=(margin < -2.0 and int(r["y_true"]) == 1))
    out |= _tile("v_act", _short(act_short, 26), _short(act_full, 58), alert=(n_bad >= 2))
    return out