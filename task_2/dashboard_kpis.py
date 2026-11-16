"""
dashboard_kpis.py — KPI-плитки страниц: подпись, крупное число, строка контекста.

Плитка, на которую стоит обратить внимание, получает класс `kpi-alert` (красная полоса и ▲).
Пороги сигналов собраны в начале файла. Строка под числом — только короткий числовой контекст
(«на приёмке …», «N из M»); развёрнутые выводы — в REPORT.md (report.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dashboard_data as dd

DASH = "—"
LOW_SHARE_DROP = 0.5    # разгрузка очереди упала более чем вдвое против приёмки — сигнал
AGREE_DROP = 0.05       # согласие с аналитиком (на инцидентах или на FP) ниже приёмки больше чем на 5 п. п.
                        # даже с учётом 95 %-ного интервала — сигнал
HIGH_CRIT = ["Высокая", "Критическая"]
AUC_DROP = 0.05         # AUC ниже приёмки больше чем на 0,05 — сигнал
OVERCONF = 1.5          # модель завышает долю инцидентов в 1,5 раза и больше — сигнал
CONFIDENT_MISS = 0.5    # больше половины пропусков далеко от порога — сигнал (сдвиг порога не поможет)
TOP_SHARE = 0.2         # «верх очереди»: доля карточек, которую успевают разобрать первой

TILES = {
    "results": [("r_top", "Инцидентов в верхних 20 % очереди"), ("r_found", "Найдено реальных инцидентов"),
                ("r_miss", "Пропущено реальных инцидентов"), ("r_low", "Очередь разгружена")],
    "agreement": [("a_all", "Общее согласие с аналитиками"), ("a_inc", "Согласие на реальных инцидентах"),
                  ("a_fp", "Согласие на False Positive"), ("a_none", "Без вердикта SOC")],
    "misses": [("m_rate", "Доля пропусков"), ("m_months", "Месяцев хуже приёмки"),
               ("m_high", "Пропущено высокой критичности"), ("m_near", "Уверенные пропуски")],
    "errors": [("e_worst", "Где пропусков больше всего"), ("e_rare", "Пропуски на редких правилах"),
               ("e_new", "Пропуски на правилах вне обучения"), ("e_fp", "FP в приоритете")],
    "trust": [("t_auc", "Качество ранжирования (AUC)"), ("t_cal", "Модель завышает вероятность"),
              ("t_flow", "Обогащение заполнено"), ("t_worst", "Худший месяц")],
}
TEXT_TILES = {"e_worst", "t_worst"}


def markup(page: str) -> str:
    """Разметка Taipy для ряда плиток страницы."""
    cells = []
    for key, label in TILES[page]:
        value_cls = "kpi-value kpi-value-text" if key in TEXT_TILES else "kpi-value"
        cells.append(f"<|part|class_name={{{key}_c}}|\n<|{label}|text|class_name=kpi-label|>\n"
                     f"<|{{{key}_v}}|text|class_name={value_cls}|>\n<|{{{key}_n}}|text|class_name=kpi-note|>\n|>")
    return f"<|layout|columns={' '.join(['1'] * len(cells))}|gap=12px|class_name=kpis|\n" + "\n".join(cells) + "\n|>"


def _tile(key: str, value: str, note: str = "", alert: bool = False, kind: str = "") -> dict:
    cls = "kpi" + (f" {kind}" if kind else "") + (" kpi-alert" if alert else "")
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": cls}


def empty(page: str) -> dict:
    out = {}
    for key, _ in TILES[page]:
        out.update(_tile(key, DASH, "нет карточек с вердиктом"))
    return out


def _short(s, n: int = 30) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- 1. Итоги
def _top_share(d: pd.DataFrame, share: float = TOP_SHARE) -> float:
    g = dd.gains(d, points=1001)
    return float(np.interp(share, g["x"], g["y"])) if len(g) else np.nan


def below_acceptance(k: int, n: int, ref: float, drop: float = AGREE_DROP) -> bool:
    """Доля k/n ниже эталона приёмки ref больше чем на drop даже по верхней границе интервала Уилсона."""
    return bool(n and np.isfinite(ref) and dd.wilson(k, n)[1] < ref - drop)


def results(oper: pd.DataFrame, c: dd.Confusion, cv: dd.Confusion, target_miss: float,
            val: pd.DataFrame | None = None) -> dict:
    if c.n == 0:
        return empty("results")
    top, top_v = _top_share(oper), (_top_share(val) if val is not None else np.nan)
    out = _tile("r_top", dd.fmt_pct(top, 0), f"на приёмке {dd.fmt_pct(top_v, 0)} · без модели 20 %", kind="kpi-real")
    out |= _tile("r_found", dd.fmt_pct(c.recall), f"{dd.fmt_int(c.tp)} из {dd.fmt_int(c.pos)} · на приёмке {dd.fmt_pct(cv.recall)}",
                 kind="kpi-real")
    out |= _tile("r_miss", dd.fmt_int(c.fn), f"{dd.fmt_pct(c.miss)} · на приёмке {dd.fmt_pct(target_miss)}",
                 alert=dd.wilson(c.fn, c.pos)[0] > target_miss)       # хуже приёмки с учётом случайности
    dropped = cv.n and c.low_share < cv.low_share * LOW_SHARE_DROP
    out |= _tile("r_low", dd.fmt_pct(c.low_share, 0),
                 f"{dd.fmt_int(c.fn + c.tn)} из {dd.fmt_int(c.n)} с вердиктом · на приёмке {dd.fmt_pct(cv.low_share, 0)}",
                 alert=bool(dropped))
    return out


# ---------------------------------------------------------------- 2. Согласие
def agreement(oper: pd.DataFrame, c: dd.Confusion, cv: dd.Confusion) -> dict:
    n_none = int((oper["y"] < 0).sum())
    out = _tile("a_none", dd.fmt_int(n_none), "дубликаты, спам, открытые")
    if c.n == 0:
        return empty("agreement") | out
    ratio = c.neg / c.pos if c.pos else np.nan
    out |= _tile("a_all", dd.fmt_pct(c.agreement),
                 f"FP в {dd.fmt_num(ratio, 1)} раза больше инцидентов" if c.pos else "")
    out |= _tile("a_inc", dd.fmt_pct(c.recall), f"на приёмке {dd.fmt_pct(cv.recall)}",
                 alert=below_acceptance(c.tp, c.pos, cv.recall), kind="kpi-real")
    out |= _tile("a_fp", dd.fmt_pct(c.specificity), f"на приёмке {dd.fmt_pct(cv.specificity)}",
                 alert=below_acceptance(c.tn, c.neg, cv.specificity), kind="kpi-fp")
    return out


# ---------------------------------------------------------------- 3. Пропуски
def misses(oper: pd.DataFrame, c: dd.Confusion, mon_oper: pd.DataFrame, thr: float, target_miss: float,
           val: pd.DataFrame | None = None) -> dict:
    if c.pos == 0:
        return empty("misses")
    lo, hi = dd.wilson(c.fn, c.pos)
    out = _tile("m_rate", dd.fmt_pct(c.miss), f"интервал {dd.fmt_pct(lo)} — {dd.fmt_pct(hi)} · на приёмке {dd.fmt_pct(target_miss)}",
                alert=lo > target_miss)
    if len(mon_oper):
        bad = mon_oper[mon_oper["miss_lo"] > target_miss]
        note = (", ".join(dd.month_label(m, short=True) for m in bad["month"])) if len(bad) else "на уровне приёмки"
        out |= _tile("m_months", f"{len(bad)} из {len(mon_oper)}", note, alert=len(bad) > 0)
    else:
        out |= _tile("m_months", DASH)
    # пропуски высокой и критической критичности: доля среди таких инцидентов против приёмки
    inc_h = oper[(oper["y"] == 1) & oper["crit"].isin(HIGH_CRIT)]
    fn_h = int((~inc_h["pred"]).sum())
    ref_h = target_miss
    if val is not None:
        val_h = val[(val["y"] == 1) & val["crit"].isin(HIGH_CRIT)]
        if len(val_h) >= dd.MIN_POS:                    # на малой приёмке берём общий уровень пропусков
            ref_h = float((~val_h["pred"]).mean())
    out |= _tile("m_high", dd.fmt_int(fn_h),
                 f"{fn_h} из {dd.fmt_int(len(inc_h))} ({dd.fmt_pct(fn_h / len(inc_h), 0) if len(inc_h) else DASH}) · "
                 f"на приёмке {dd.fmt_pct(ref_h)}",
                 alert=bool(len(inc_h) and dd.wilson(fn_h, len(inc_h))[0] > ref_h))
    near, n = dd.near_threshold_share(oper, thr)
    far = n - near                                       # оценка ниже половины порога: сдвиг порога не вернёт
    out |= _tile("m_near", dd.fmt_pct(far / n, 0) if n else DASH,
                 f"{far} из {n} · оценка < ½ порога" if n else "пропусков нет",
                 alert=bool(n and far / n > CONFIDENT_MISS))
    return out


# ---------------------------------------------------------------- 4. Зоны ошибок
def errors(oper: pd.DataFrame, c: dd.Confusion, cv: dd.Confusion | None = None) -> dict:
    if c.n == 0:
        return empty("errors")
    worst = None
    for by in dd.SEGMENTS:
        if by == "Правило SIEM":
            continue
        t = dd.segment_table(oper, by, "miss")
        if t.empty:
            continue
        t = t[t["systematic"] & (t["seg"] != dd.OTHER)]
        if len(t):
            r = t.sort_values("lo", ascending=False).iloc[0]
            if worst is None or r["lo"] > worst[1]["lo"]:
                worst = (by, r)
    if worst:
        by, r = worst
        out = _tile("e_worst", _short(r["seg"], 28),
                    f"{by[0].lower() + by[1:]} · {int(r['fn'])} из {int(r['n'])}", alert=True)
    else:
        out = _tile("e_worst", "Нет явных зон", "в пределах случайности")
    inc = oper[oper["y"] == 1]
    rare = inc[(inc["rule_train_n"] > 0) & (inc["rule_train_n"] < 100)]
    if len(rare):
        rr = float((~rare["pred"]).mean())
        out |= _tile("e_rare", dd.fmt_pct(rr, 0),
                     f"{int((~rare['pred']).sum())} из {len(rare)} · в среднем {dd.fmt_pct(c.miss)}",
                     alert=rr > 2 * c.miss)
    else:
        out |= _tile("e_rare", DASH, "нет инцидентов редких правил")
    unseen = inc[inc["rule_train_n"] == 0]
    if len(unseen):
        ru = float((~unseen["pred"]).mean())
        out |= _tile("e_new", dd.fmt_pct(ru, 0), f"{int((~unseen['pred']).sum())} из {len(unseen)} · в среднем {dd.fmt_pct(c.miss)}",
                     alert=ru > 2 * c.miss)
    else:
        out |= _tile("e_new", DASH, "нет инцидентов новых правил")
    out |= _tile("e_fp", dd.fmt_pct(1 - c.specificity, 0), f"{dd.fmt_int(c.fp)} карточек · на приёмке {dd.fmt_pct(1 - cv.specificity, 0)}"
                 if cv is not None and cv.neg else f"{dd.fmt_int(c.fp)} карточек",
                 alert=bool(cv is not None and below_acceptance(c.tn, c.neg, cv.specificity)),
                 kind="kpi-fp")
    return out


# ---------------------------------------------------------------- 5. Доверие к оценке
def trust(oper: pd.DataFrame, val: pd.DataFrame, mon_oper: pd.DataFrame, meta: dict) -> dict:
    lab, lab_v = oper[oper["y"] >= 0], val[val["y"] >= 0]
    if lab.empty:
        return empty("trust")
    a, av = dd.auc(lab["y"], lab["proba"]), dd.auc(lab_v["y"], lab_v["proba"])
    out = _tile("t_auc", dd.fmt_num(a, 2) if np.isfinite(a) else DASH,
                f"на приёмке {dd.fmt_num(av, 2)}" if np.isfinite(av) else "",
                alert=bool(np.isfinite(a) and np.isfinite(av) and av - a > AUC_DROP))
    pred, act = float(lab["proba"].mean()), float(lab["y"].mean())
    ratio = pred / act if act else np.nan
    out |= _tile("t_cal", f"×{dd.fmt_num(ratio, 1)}" if np.isfinite(ratio) else DASH,
                 f"оценка {dd.fmt_pct(pred)} · факт {dd.fmt_pct(act)}",
                 alert=bool(np.isfinite(ratio) and (ratio >= OVERCONF or ratio <= 1 / OVERCONF)))
    enr = float((lab["has_vendor"] & lab["has_alert"]).mean())
    enr_v = float((lab_v["has_vendor"] & lab_v["has_alert"]).mean()) if len(lab_v) else np.nan
    out |= _tile("t_flow", dd.fmt_pct(enr, 0),
                 f"на приёмке {dd.fmt_pct(enr_v, 0)}",
                 alert=bool(np.isfinite(enr_v) and enr < enr_v * 0.75))
    m = mon_oper.dropna(subset=["auc"])
    if len(m):
        w = m.sort_values("auc").iloc[0]
        out |= _tile("t_worst", dd.month_label(w["month"]).capitalize(),
                     f"AUC {dd.fmt_num(w['auc'], 2)}",
                     alert=w["auc"] < 0.6)
    else:
        out |= _tile("t_worst", DASH)
    return out
