"""
dashboard_kpis.py — KPI-плитки страниц: подпись, крупное число, строка контекста.

Плитка, на которую стоит обратить внимание, получает класс `kpi-alert` (красная полоса и ▲).
Пороги сигналов собраны в начале файла. Развёрнутые выводы — в REPORT.md (report.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dashboard_data as dd

DASH = "—"
LOW_SHARE_DROP = 0.5    # разгрузка очереди упала более чем вдвое против приёмки — сигнал
AUC_DROP = 0.05         # AUC ниже приёмки больше чем на 0,05 — сигнал
OVERCONF = 1.5          # модель завышает долю инцидентов в 1,5 раза и больше — сигнал

TILES = {
    "results": [("r_n", "Карточек оценено"), ("r_found", "Найдено реальных инцидентов"),
                ("r_miss", "Пропущено реальных инцидентов"), ("r_low", "Очередь разгружена")],
    "agreement": [("a_all", "Общее согласие с аналитиками"), ("a_inc", "Согласие на реальных инцидентах"),
                  ("a_fp", "Согласие на False Positive"), ("a_none", "Без вердикта SOC")],
    "misses": [("m_rate", "Доля пропусков"), ("m_months", "Месяцев хуже цели"),
               ("m_high", "Пропущено высокой критичности"), ("m_near", "Пропуски у самого порога")],
    "errors": [("e_worst", "Где пропусков больше всего"), ("e_rare", "Пропуски на редких правилах"),
               ("e_new", "Карточки с новыми правилами"), ("e_fp", "FP в приоритете")],
    "trust": [("t_auc", "Качество ранжирования (AUC)"), ("t_cal", "Модель завышает вероятность"),
              ("t_flow", "Доля инцидентов в потоке"), ("t_worst", "Худший месяц")],
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
def results(oper: pd.DataFrame, c: dd.Confusion, cv: dd.Confusion, target_miss: float) -> dict:
    if c.n == 0:
        return empty("results")
    out = _tile("r_n", dd.fmt_int(len(oper)), f"с вердиктом SOC: {dd.fmt_int(c.n)} — по ним считаем качество")
    out |= _tile("r_found", dd.fmt_pct(c.recall), f"{dd.fmt_int(c.tp)} из {dd.fmt_int(c.pos)} попали в приоритет",
                 kind="kpi-real")
    out |= _tile("r_miss", dd.fmt_int(c.fn), f"{dd.fmt_pct(c.miss)} реальных инцидентов · цель приёмки ≤ {dd.fmt_pct(target_miss, 0)}",
                 alert=c.miss > target_miss)
    dropped = cv.n and c.low_share < cv.low_share * LOW_SHARE_DROP
    out |= _tile("r_low", dd.fmt_pct(c.low_share, 0),
                 f"{dd.fmt_int(c.fn + c.tn)} карточек отложено · на приёмке {dd.fmt_pct(cv.low_share, 0)}",
                 alert=bool(dropped))
    return out


# ---------------------------------------------------------------- 2. Согласие
def agreement(oper: pd.DataFrame, c: dd.Confusion, cv: dd.Confusion) -> dict:
    n_none = int((oper["y"] < 0).sum())
    out = _tile("a_none", dd.fmt_int(n_none), "дубликаты, спам и карточки без вердикта: модель их оценила, сверить не с чем")
    if c.n == 0:
        return empty("agreement") | out
    ratio = c.neg / c.pos if c.pos else np.nan
    out |= _tile("a_all", dd.fmt_pct(c.agreement),
                 f"FP в {dd.fmt_num(ratio, 1)} раза больше инцидентов: общая доля скрывает разницу классов" if c.pos else "")
    out |= _tile("a_inc", dd.fmt_pct(c.recall), f"на приёмке {dd.fmt_pct(cv.recall)}", kind="kpi-real")
    out |= _tile("a_fp", dd.fmt_pct(c.specificity), f"на приёмке {dd.fmt_pct(cv.specificity)}",
                 alert=bool(cv.n and c.specificity < cv.specificity * LOW_SHARE_DROP), kind="kpi-fp")
    return out


# ---------------------------------------------------------------- 3. Пропуски
def misses(oper: pd.DataFrame, c: dd.Confusion, mon_oper: pd.DataFrame, thr: float, target_miss: float) -> dict:
    if c.pos == 0:
        return empty("misses")
    lo, hi = dd.wilson(c.fn, c.pos)
    out = _tile("m_rate", dd.fmt_pct(c.miss), f"95 %-ный интервал {dd.fmt_pct(lo)} — {dd.fmt_pct(hi)}",
                alert=lo > target_miss)
    if len(mon_oper):
        bad = mon_oper[mon_oper["miss_lo"] > target_miss]
        note = ("хуже: " + ", ".join(dd.month_label(m) for m in bad["month"])) if len(bad) else "все месяцы в пределах цели"
        out |= _tile("m_months", f"{len(bad)} из {len(mon_oper)}", note, alert=len(bad) > 0)
    else:
        out |= _tile("m_months", DASH)
    miss = oper[(oper["y"] == 1) & ~oper["pred"]]
    high = int(miss["crit"].isin(["Высокая", "Критическая"]).sum())
    out |= _tile("m_high", dd.fmt_int(high),
                 f"из {dd.fmt_int(len(miss))} {dd.plural(len(miss), ('пропуска', 'пропусков', 'пропусков'))}; "
                 "остальные — средней и низкой",
                 alert=high > 0)
    near, n = dd.near_threshold_share(oper, thr)
    out |= _tile("m_near", dd.fmt_pct(near / n, 0) if n else DASH,
                 f"{near} из {n}: оценка не ниже половины порога — их ловит более низкий порог" if n else "пропусков нет")
    return out


# ---------------------------------------------------------------- 4. Зоны ошибок
def errors(oper: pd.DataFrame, c: dd.Confusion) -> dict:
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
                    f"{by.lower()}: пропущено {int(r['fn'])} из {int(r['n'])} ({dd.fmt_pct(r['rate'], 0)})", alert=True)
    else:
        out = _tile("e_worst", "Нет явных зон", "ни один срез не выходит за случайный разброс")
    inc = oper[oper["y"] == 1]
    rare = inc[(inc["rule_train_n"] > 0) & (inc["rule_train_n"] < 100)]
    if len(rare):
        rr = float((~rare["pred"]).mean())
        out |= _tile("e_rare", dd.fmt_pct(rr, 0),
                     f"{int((~rare['pred']).sum())} из {len(rare)} инцидентов правил, которых в обучении было < 100 · в среднем {dd.fmt_pct(c.miss)}",
                     alert=rr > 2 * c.miss)
    else:
        out |= _tile("e_rare", DASH, "нет инцидентов редких правил")
    lab = oper[oper["y"] >= 0]
    new = float((lab["rule_train_n"] == 0).mean())
    out |= _tile("e_new", dd.fmt_pct(new, 0), f"{dd.fmt_int((lab['rule_train_n'] == 0).sum())} карточек правил, которых модель не видела")
    out |= _tile("e_fp", dd.fmt_pct(1 - c.specificity, 0), f"{dd.fmt_int(c.fp)} ложных карточек модель не отложила",
                 kind="kpi-fp")
    return out


# ---------------------------------------------------------------- 5. Доверие к оценке
def trust(oper: pd.DataFrame, val: pd.DataFrame, mon_oper: pd.DataFrame, meta: dict) -> dict:
    lab, lab_v = oper[oper["y"] >= 0], val[val["y"] >= 0]
    if lab.empty:
        return empty("trust")
    a, av = dd.auc(lab["y"], lab["proba"]), dd.auc(lab_v["y"], lab_v["proba"])
    out = _tile("t_auc", dd.fmt_num(a, 2) if np.isfinite(a) else DASH,
                f"на приёмке {dd.fmt_num(av, 2)} · 0,5 — как монетка, 1 — идеально" if np.isfinite(av) else "",
                alert=bool(np.isfinite(a) and np.isfinite(av) and av - a > AUC_DROP))
    pred, act = float(lab["proba"].mean()), float(lab["y"].mean())
    ratio = pred / act if act else np.nan
    out |= _tile("t_cal", f"×{dd.fmt_num(ratio, 1)}" if np.isfinite(ratio) else DASH,
                 f"пишет в среднем {dd.fmt_pct(pred)}, реальных инцидентов {dd.fmt_pct(act)}",
                 alert=bool(np.isfinite(ratio) and (ratio >= OVERCONF or ratio <= 1 / OVERCONF)))
    out |= _tile("t_flow", dd.fmt_pct(act),
                 f"на приёмке {dd.fmt_pct(lab_v['y'].mean()) if len(lab_v) else DASH} · "
                 f"в обучении {dd.fmt_pct(meta.get('pos_rate_train', np.nan))}", kind="kpi-real")
    m = mon_oper.dropna(subset=["auc"])
    if len(m):
        w = m.sort_values("auc").iloc[0]
        out |= _tile("t_worst", dd.month_label(w["month"]).capitalize(),
                     f"AUC {dd.fmt_num(w['auc'], 2)} · доля инцидентов {dd.fmt_pct(w['inc_rate'])}",
                     alert=w["auc"] < 0.6)
    else:
        out |= _tile("t_worst", DASH)
    return out
