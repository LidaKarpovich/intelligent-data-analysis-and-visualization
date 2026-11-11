"""
dashboard_kpis.py — KPI-плитки страниц дашборда.

Каждая страница начинается с ряда плиток: подпись, крупное число и короткая строка контекста.
Плитка, на которую стоит обратить внимание, получает класс `kpi-alert` (красная полоса слева).
Развёрнутые выводы — в отчёте REPORT.md (см. report.py), а не на листах дашборда.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import dashboard_data as dd

DASH = "—"

# key -> подпись; порядок = порядок плиток на странице
TILES = {
    "overview": [
        ("ov_n", "Карточек"),
        ("ov_fp", "Доля False Positive"),
        ("ov_real", "Реальных инцидентов"),
        ("ov_ratio", "Ложных на один инцидент"),
        ("ov_none", "Без вердикта"),
        ("ov_react", "Медиана взятия в работу"),
        ("ov_crit", "Взятие критических карточек"),
        ("ov_trend", "Поток к концу периода"),
    ],
    "compare": [
        ("cp_top", "Топ-10 правил SIEM"),
        ("cp_noise", "Правил-«шумов» в топ-10"),
        ("cp_irp", "Самая «шумная» категория IRP"),
        ("cp_peak", "Пик нагрузки"),
    ],
    "eda": [
        ("ed_day", "Ждут взятия дольше суток"),
        ("ed_inst", "Взяты «мгновенно»"),
        ("ed_sigma", "Часов выше «среднее + 3σ»"),
        ("ed_l2", "Решение на L2 против L1"),
    ],
    "flows": [
        ("fl_delta", "Изменение потока"),
        ("fl_main", "Главный вклад в изменение"),
        ("fl_tune", "Правил без инцидентов в топ-20"),
        ("fl_ext", "Внешних адресов в топ-20 src"),
    ],
}
TEXT_TILES = {"cp_irp", "cp_peak", "fl_main"}          # значение — название, а не число: шрифт мельче


def markup(page: str) -> str:
    """Разметка Taipy для ряда плиток страницы (по 4 в строке)."""
    tiles = TILES[page]
    rows = [tiles[i:i + 4] for i in range(0, len(tiles), 4)]
    out = []
    for row in rows:
        cells = []
        for key, label in row:
            value_cls = "kpi-value kpi-value-text" if key in TEXT_TILES else "kpi-value"
            cells.append(f"<|part|class_name={{{key}_c}}|\n<|{label}|text|class_name=kpi-label|>\n"
                         f"<|{{{key}_v}}|text|class_name={value_cls}|>\n<|{{{key}_n}}|text|class_name=kpi-note|>\n|>")
        out.append(f"<|layout|columns={' '.join(['1'] * len(row))}|gap=12px|class_name=kpis|\n"
                   + "\n".join(cells) + "\n|>")
    return "\n\n".join(out)


def _tile(key: str, value: str, note: str = "", alert: bool = False, kind: str = "") -> dict:
    cls = "kpi" + (f" {kind}" if kind else "") + (" kpi-alert" if alert else "")
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": cls}


def empty(page: str) -> dict:
    out = {}
    for key, _ in TILES[page]:
        out.update(_tile(key, DASH))
    return out


def _share(d: pd.DataFrame, cls: str) -> float:
    return float((d["cls"] == cls).mean()) if len(d) else 0.0


def _short(s, n: int = 30) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _times(a: float, b: float) -> str:
    r = a / b
    return f"×{dd.fmt_num(r, 1)}" if r < 10 else f"×{dd.fmt_int(r)}"


# ================================================================ Обзор
def overview(d: pd.DataFrame, dates: pd.DatetimeIndex | None = None) -> dict:
    if d.empty:
        return empty("overview")
    n = len(d)
    days = len(dates) if dates is not None and len(dates) else int(d["date"].nunique())
    fp, real, none = _share(d, dd.CLS_FP), _share(d, dd.CLS_REAL), _share(d, dd.CLS_NONE)
    out = {}
    out.update(_tile("ov_n", dd.fmt_int(n), f"≈ {dd.fmt_int(n / max(days, 1))} в день · {dd.fmt_int(days)} дн."))
    out.update(_tile("ov_fp", dd.fmt_pct(fp), f"{dd.fmt_int(fp * n)} карточек", kind="kpi-fp"))
    out.update(_tile("ov_real", dd.fmt_int(real * n), f"{dd.fmt_pct(real)} потока", kind="kpi-real"))
    if real > 0:
        ratio = fp / real
        out.update(_tile("ov_ratio", dd.fmt_num(ratio, 1), "False Positive на одну карточку «Решён»",
                         alert=ratio >= 3))
    else:
        out.update(_tile("ov_ratio", DASH, "подтверждённых инцидентов нет"))

    if none > 0:
        nv = d.loc[d["cls"] == dd.CLS_NONE, "rule"].value_counts()
        note = f"{dd.fmt_pct(nv.iloc[0] / nv.sum(), 0)} из них — {_short(nv.index[0])}"
        out.update(_tile("ov_none", dd.fmt_pct(none), note, alert=none >= 0.2))
    else:
        out.update(_tile("ov_none", dd.fmt_pct(0), "все карточки размечены"))

    react_all = pd.to_numeric(d["react_min"], errors="coerce").dropna()
    react = dd.plotted_minutes(react_all)
    if len(react):
        out.update(_tile("ov_react", dd.fmt_minutes(react.median()),
                         f"без «мгновенных» ({dd.fmt_pct(dd.instant_share(react_all), 0)})"))
    else:
        out.update(_tile("ov_react", DASH))

    cr = dd.crit_vs_rest(d)
    if cr.n_crit < dd.MIN_GROUP:
        out.update(_tile("ov_crit", DASH, f"меньше {dd.MIN_GROUP} критических карточек"))
    elif cr.n_rest < dd.MIN_GROUP:
        why = ("других карточек в фильтре нет" if cr.n_rest == 0
               else f"других карточек меньше {dd.MIN_GROUP}")
        out.update(_tile("ov_crit", dd.fmt_minutes(cr.crit), f"медиана; не с чем сравнить: {why}"))
    else:
        rest = dd.fmt_minutes(cr.rest)
        if cr.slower:
            note = f"медиана; {_times(cr.crit, cr.rest)} к остальным ({rest})"
        elif round(cr.ratio, 1) < 1.0:
            note = f"медиана; быстрее остальных ({rest})"
        else:
            note = f"медиана; как у остальных ({rest})"
        # красная плитка — только когда множитель на плитке действительно больше ×1,0
        out.update(_tile("ov_crit", dd.fmt_minutes(cr.crit), note, alert=cr.slower))

    trend = _trend(d, dates)
    if trend and trend[0] > 0:
        a, b = trend
        chg = (b - a) / a
        sign = "+" if chg > 0 else "−"
        # малые значения не округляем до нуля: «0,04 в день» ≠ «0», а −99,6 % ≠ −100 %
        digits = 0 if 0.1 <= abs(chg) < 0.995 else 1
        out.update(_tile("ov_trend", f"{sign}{dd.fmt_pct(abs(chg), digits)}",
                         f"{_per_day(a)} → {_per_day(b)} карточек в день"))
    else:
        out.update(_tile("ov_trend", DASH, "период короче 8 дней"))
    return out


def _per_day(x: float) -> str:
    return dd.fmt_int(x) if x >= 10 else dd.fmt_num(x, 1 if x >= 0.1 else 2)


def _trend(d: pd.DataFrame, dates: pd.DatetimeIndex | None = None) -> tuple[float, float] | None:
    """Средний поток в день: первая и последняя четверть периода (нужно ≥ 8 дней)."""
    s = d.groupby("date").size()
    idx = dates if dates is not None and len(dates) else (
        pd.date_range(s.index.min(), s.index.max(), freq="D") if len(s) else s.index)
    s = s.reindex(idx, fill_value=0)
    if len(s) < 8:
        return None
    q = max(len(s) // 4, 2)
    return float(s.iloc[:q].mean()), float(s.iloc[-q:].mean())


# ================================================================ Сравнение
def compare(d: pd.DataFrame) -> dict:
    if d.empty:
        return empty("compare")
    out = {}
    vc = d["rule"].value_counts()
    k = min(10, len(vc))
    out.update(_tile("cp_top", dd.fmt_pct(vc.head(k).sum() / len(d)),
                     f"потока из {dd.fmt_int(len(vc))} правил" if k == 10 else f"потока, правил всего {k}"))

    top = vc.head(k).index
    real = d[d["rule"].isin(top)].groupby("rule")["cls"].apply(lambda s: (s == dd.CLS_REAL).mean())
    noisy = real[real < 0.01].index
    out.update(_tile("cp_noise", dd.fmt_int(len(noisy)),
                     f"{dd.fmt_pct(vc[noisy].sum() / len(d))} потока, инцидентов < 1 %" if len(noisy)
                     else "у всех правил топа есть инциденты", alert=len(noisy) > 0))

    g = d.groupby("irp")["cls"].agg(n="size", fp=lambda s: (s == dd.CLS_FP).mean())
    g = g[g["n"] >= 100]
    if len(g) >= 2:
        avg = _share(d, dd.CLS_FP)
        hi = g["fp"].idxmax()
        dev = (g.loc[hi, "fp"] - avg) * 100
        out.update(_tile("cp_irp", _short(hi, 34),
                         f"False Positive {dd.fmt_pct(g.loc[hi, 'fp'], 0)}, на {dd.fmt_num(dev, 0)} п.п. выше среднего"))
    else:
        out.update(_tile("cp_irp", DASH, "мало категорий с ≥ 100 карточками"))

    t = pd.crosstab(d["dow"], d["hour"])
    i, j = np.unravel_index(np.argmax(t.values), t.shape)
    night = d["hour"].between(0, 5).mean()
    out.update(_tile("cp_peak", f"{dd.WEEKDAYS[t.index[i]]}, {t.columns[j]:02d}:00",
                     f"ночью (00–06) — {dd.fmt_pct(night, 0)} карточек"))
    return out


# ================================================================ Распределения
def eda(d: pd.DataFrame, hourly: pd.Series | None = None) -> dict:
    if d.empty:
        return empty("eda")
    out = {}
    react_all = pd.to_numeric(d["react_min"], errors="coerce").dropna()
    if len(react_all):
        day = float((react_all > 1440).mean())
        out.update(_tile("ed_day", dd.fmt_pct(day), f"из {dd.fmt_int(len(react_all))} карточек с временем взятия",
                         alert=day >= 0.05))
        out.update(_tile("ed_inst", dd.fmt_pct(dd.instant_share(react_all)), "быстрее 0,6 с — автоматика"))
    else:
        out.update(_tile("ed_day", DASH))
        out.update(_tile("ed_inst", DASH))

    x = hourly.to_numpy(float) if hourly is not None else np.array([])
    if len(x) >= 20 and x.std() > 0:
        above = float((x > x.mean() + 3 * x.std()).mean())
        out.update(_tile("ed_sigma", dd.fmt_pct(above, 2), "при нормальном законе — 0,13 %", alert=above > 0.005))
    else:
        out.update(_tile("ed_sigma", DASH, "меньше 20 часовых интервалов"))

    med = {}
    for grp in ("L1", "L2"):
        v = dd.plotted_minutes(d.loc[d["group"] == grp, "solve_min"])
        if len(v) >= 5:
            med[grp] = v.median()
    if len(med) == 2 and med["L1"] > 0:
        r = med["L2"] / med["L1"]
        out.update(_tile("ed_l2", _times(med["L2"], med["L1"]),
                         f"медианы: L1 {dd.fmt_minutes(med['L1'])}, L2 {dd.fmt_minutes(med['L2'])}", alert=r >= 10))
    else:
        out.update(_tile("ed_l2", DASH, "мало карточек L1 или L2"))
    return out


# ================================================================ Потоки и плотность
def flows(d: pd.DataFrame, prev: pd.DataFrame, cur: pd.DataFrame, n_days: int, top_src: pd.DataFrame) -> dict:
    out = {}
    if len(prev) or len(cur):
        delta = len(cur) - len(prev)
        sign = "+" if delta >= 0 else "−"
        value = f"{sign}{dd.fmt_pct(abs(delta) / len(prev), 0)}" if len(prev) else f"+{dd.fmt_int(len(cur))}"
        out.update(_tile("fl_delta", value,
                         f"{sign}{dd.fmt_int(abs(delta))} карточек: {n_days} дн. против предыдущих {n_days}"))
        a, b = prev["irp"].value_counts(), cur["irp"].value_counts()
        idx = a.index.union(b.index)
        dlt = b.reindex(idx, fill_value=0) - a.reindex(idx, fill_value=0)
        if len(dlt) and dlt.abs().max() > 0:
            key = dlt.abs().idxmax()
            v = dlt[key]
            out.update(_tile("fl_main", _short(key, 34), f"{'+' if v > 0 else '−'}{dd.fmt_int(abs(v))} карточек"))
        else:
            out.update(_tile("fl_main", DASH, "состав не изменился"))
    else:
        out.update(_tile("fl_delta", DASH, "нет карточек в окне"))
        out.update(_tile("fl_main", DASH))

    if not d.empty:
        g = d.groupby("rule").agg(n=("cls", "size"), real=("cls", lambda s: (s == dd.CLS_REAL).mean()))
        g = g[g["n"] >= 30].sort_values("n", ascending=False).head(20)
        zero = g[g["real"] < 0.01]
        out.update(_tile("fl_tune", dd.fmt_int(len(zero)),
                         f"{dd.fmt_int(zero['n'].sum())} карточек — кандидаты на тюнинг" if len(zero)
                         else "у всех есть подтверждённые инциденты", alert=len(zero) > 0))
    else:
        out.update(_tile("fl_tune", DASH))

    if len(top_src):
        src = top_src["src"].astype(str)
        private = src.str.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|127\.)")
        google = src.str.match(r"^142\.25[01]\.")
        note = f"из них {int(google.sum())} — подсеть Google 142.250.0.0/15" if google.sum() else "все за весь период"
        out.update(_tile("fl_ext", dd.fmt_int(int((~private).sum())), note))
    else:
        out.update(_tile("fl_ext", DASH))
    return out
