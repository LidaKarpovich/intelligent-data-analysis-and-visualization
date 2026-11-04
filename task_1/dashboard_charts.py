"""
dashboard_charts.py — Plotly-фигуры дашборда (данные -> go.Figure).

Общие правила оформления:
  * без 3D, теней и рамок; сетка — редкая, бледная, только по оси значений;
  * нейтральная серая база + один фокусный цвет; цвета исходов одинаковы на всех графиках
    (синий — False Positive, оранжевый — реальный инцидент);
  * подписи осей с единицами измерения; прямые подписи значений там, где серий мало;
  * заголовок и вывод — в карточке страницы над графиком (заголовок Plotly не дублируется);
  * подписи на графике не должны перекрывать друг друга и данные.
Палитры проверены на различимость при дальтонизме.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import dashboard_data as dd

# ---------------------------------------------------------------- палитра
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e8e7e1", "#c3c2b7", "#ffffff"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GRAY, GRAY_LIGHT = "#a8a69e", "#d6d4cc"
NEUTRAL_MID = "#f0efec"

CLASS_COLORS = {dd.CLS_REAL: ORANGE, dd.CLS_FP: BLUE, dd.CLS_NONE: GRAY, dd.CLS_OTHER: GRAY_LIGHT}
SEQ_BLUE = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORDINAL_5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # порядковая шкала (валидирована)
ORDINAL_3 = ["#86b6ef", "#2a78d6", "#104281"]
CRIT_COLORS = dict(zip(dd.CRIT_ORDER, ORDINAL_5))
GROUP_COLORS = dict(zip(dd.GROUP_ORDER, ORDINAL_3))

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
RNG_SEED = 42

# Ось времени в минутах, отложенных в log10 (подписи — человекочитаемые)
_T = [(0.1, "6 с"), (1, "1 мин"), (10, "10 мин"), (60, "1 ч"), (1440, "1 сут"), (10080, "1 нед"),
      (43200, "30 сут"), (525600, "1 год")]
TIME_TICKS = dict(tickvals=[np.log10(v) for v, _ in _T], ticktext=[t for _, t in _T])

PLOT_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": "hover",
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"]}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


_AXIS = dict(zeroline=False, automargin=True, title=dict(font=dict(size=12, color=INK_2)),
             tickfont=dict(color=MUTED, size=11))
# Шаблон задаёт значения по умолчанию; явные настройки конкретного графика их перекрывают
TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=", ",
    font=dict(family=FONT, size=12, color=INK_2),
    xaxis=dict(showgrid=False, showline=True, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4, **_AXIS),
    yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, **_AXIS),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color=INK_2)),
    bargap=0.25, hovermode="closest",
))


def base(fig: go.Figure, height: int = 360, legend: bool = False, **kw) -> go.Figure:
    """Единый «чистый» стиль для всех графиков (шаблон + высота, поля, легенда)."""
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=8, r=16, t=12, b=8), showlegend=legend)
    if kw:
        fig.update_layout(**kw)
    return fig


def empty(msg: str = "Нет данных для выбранных фильтров", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base(fig, height)


def _short(s: str, n: int = 38) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ================================================================ 1. ОБЗОР
def _daily_series(d: pd.DataFrame, dates: pd.DatetimeIndex | None) -> pd.Series:
    """Карточки по дням. dates — календарь фильтра: день без карточек остаётся нулём."""
    s = d.groupby("date").size() if not d.empty else pd.Series(dtype=float)
    if dates is not None and len(dates):
        return s.reindex(pd.DatetimeIndex(dates), fill_value=0)
    if s.empty:
        return s
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0)


def fig_daily(d: pd.DataFrame, dates: pd.DatetimeIndex | None = None) -> go.Figure:
    """Поток карточек по дням и скользящее среднее за 7 дней."""
    s = _daily_series(d, dates)
    if s.empty:
        return empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="за день",
                             line=dict(color=GRAY_LIGHT, width=1.5),
                             hovertemplate="%{x|%d.%m.%Y}: %{y:,} карточек<extra></extra>"))
    if len(s) >= 14:
        r = s.rolling(7, min_periods=4, center=True).mean()
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name="скользящее среднее за 7 дней",
                                 line=dict(color=BLUE, width=2.5),
                                 hovertemplate="%{x|%d.%m.%Y}: в среднем %{y:,.0f}/день<extra></extra>"))
    fig.update_yaxes(title_text="Карточек в день", rangemode="tozero", tickformat=",d")
    fig.update_xaxes(tickformat="%d.%m.%y", hoverformat="%d.%m.%Y", range=[s.index.min(), s.index.max()])
    return base(fig, 340, legend=len(fig.data) > 1, margin=dict(l=8, r=16, t=36, b=8), hovermode="x unified")


def _grain(d: pd.DataFrame) -> tuple[str, str]:
    span = (d["date"].max() - d["date"].min()).days
    if span <= 45:
        return "D", "%d.%m"
    if span <= 200:
        return "W", "%d.%m"
    return "M", "%m.%Y"


def grain_label(d: pd.DataFrame) -> str:
    return {"D": "дням", "W": "неделям", "M": "месяцам"}[_grain(d)[0]]


def fig_class_mix(d: pd.DataFrame) -> go.Figure:
    """Доли исходов карточек по интервалам (100-процентная столбчатая с накоплением)."""
    if d.empty:
        return empty()
    freq, fmt = _grain(d)
    key = d["date"].dt.to_period(freq).dt.start_time
    t = pd.crosstab(key, d["cls"]).reindex(columns=dd.CLASS_ORDER, fill_value=0)
    t = t[t.sum(axis=1) >= 30]                       # почти пустые интервалы дают «100 %» из единиц карточек
    if t.empty:
        return empty("Слишком мало карточек в интервалах")
    sh = t.div(t.sum(axis=1), axis=0)
    labels = [i.strftime(fmt) for i in sh.index]
    fig = go.Figure()
    for c in dd.CLASS_ORDER:
        fig.add_trace(go.Bar(x=labels, y=sh[c], name=c, marker=dict(color=CLASS_COLORS[c], line=dict(width=1, color=SURFACE)),
                             customdata=t[c], hovertemplate=f"{c}<br>%{{x}}: %{{y:.1%}} (%{{customdata:,}})<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.15, legend=dict(traceorder="normal"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title_text="Доля карточек")
    fig.update_xaxes(type="category", tickangle=0, nticks=14)
    return base(fig, 360, legend=True, margin=dict(l=8, r=16, t=56, b=8))


def fig_verdicts(d: pd.DataFrame) -> go.Figure:
    """Доли вердиктов (столбцы долей; круговая не подходит — 7 категорий, есть очень малые)."""
    if d.empty:
        return empty()
    vc = d["verdict_txt"].value_counts()
    sh = vc / vc.sum()
    order = vc.index[::-1]
    colors = []
    for v in order:
        v2 = str(v).replace("ё", "е")
        colors.append(ORANGE if v2 == "Решен" else BLUE if v == "False Positive" else GRAY)
    fig = go.Figure(go.Bar(
        x=sh[order], y=list(order), orientation="h", marker=dict(color=colors, cornerradius=3),
        text=[f"{dd.fmt_pct(sh[v])} · {dd.fmt_int(vc[v])}" for v in order], textposition="outside",
        cliponaxis=False, textfont=dict(color=INK_2),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, sh.max() * 1.45])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2, size=12), ticklabelstandoff=6)
    return base(fig, 360, bargap=0.35)


# ================================================================ 2. СРАВНЕНИЕ
def fig_top_rules(d: pd.DataFrame, k: int = 10) -> go.Figure:
    """Топ правил по числу карточек: один цвет — сравнивается длина."""
    if d.empty:
        return empty()
    vc = d["rule"].value_counts().head(k)[::-1]
    fig = go.Figure(go.Bar(
        x=vc.values, y=[_short(r, 44) for r in vc.index], orientation="h", customdata=list(vc.index),
        marker=dict(color=BLUE, cornerradius=3), text=[dd.fmt_int(v) for v in vc.values],
        textposition="outside", cliponaxis=False, textfont=dict(color=INK_2),
        hovertemplate="%{customdata}<br>%{x:,} карточек<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, vc.max() * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2, size=11), ticklabelstandoff=6)
    return base(fig, 380, bargap=0.3)


def fig_fp_deviation(d: pd.DataFrame, min_n: int = 100) -> go.Figure:
    """Отклонение доли False Positive категории IRP от средней доли FP (в процентных пунктах)."""
    if d.empty:
        return empty()
    g = d.groupby("irp").agg(n=("cls", "size"), fp=("cls", lambda s: (s == dd.CLS_FP).mean()))
    g = g[g["n"] >= min_n]
    if len(g) < 2:
        return empty("Мало категорий с ≥ 100 карточками")
    avg = (d["cls"] == dd.CLS_FP).mean()
    g["dev"] = (g["fp"] - avg) * 100
    g = g.sort_values("dev")
    colors = [RED if v > 0 else BLUE for v in g["dev"]]
    fig = go.Figure(go.Bar(
        x=g["dev"], y=g.index, orientation="h", marker=dict(color=colors, cornerradius=3),
        text=[f"{v:+.0f} п.п.".replace("-", "−") for v in g["dev"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), customdata=np.c_[g["fp"] * 100, g["n"]],
        hovertemplate="%{y}<br>доля FP %{customdata[0]:.1f} % (карточек: %{customdata[1]:,})"
                      "<br>отклонение %{x:+.1f} п.п.<extra></extra>"))
    lim = max(abs(g["dev"]).max() * 1.35, 5)
    fig.add_vline(x=0, line=dict(color=INK_2, width=1))
    fig.update_xaxes(range=[-lim, lim], title_text=f"Отклонение от средней доли FP ({dd.fmt_pct(avg)}), п.п.",
                     zeroline=False, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2, size=12), ticklabelstandoff=6)
    return base(fig, 380, bargap=0.3)


def fig_heatmap_hour(d: pd.DataFrame, dates: pd.DatetimeIndex | None = None) -> go.Figure:
    """Среднее число карточек в час по дням недели.

    Знаменатель — все даты фильтра этого дня недели, включая дни без карточек.
    """
    if d.empty and (dates is None or len(dates) == 0):
        return empty()
    dow = d["dow"] if not d.empty else pd.Series(dtype=int)
    hour = d["hour"] if not d.empty else pd.Series(dtype=int)
    cnt = pd.crosstab(dow, hour).reindex(index=range(7), columns=range(24), fill_value=0)
    if dates is not None and len(dates):
        n_days = pd.Series(pd.DatetimeIndex(dates).dayofweek).value_counts().reindex(range(7), fill_value=0)
        n_days = n_days.replace(0, np.nan)
    else:
        n_days = d.groupby("dow")["date"].nunique().reindex(range(7)).replace(0, np.nan) if not d.empty else pd.Series(np.nan, index=range(7))
    z = cnt.div(n_days, axis=0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f"{h:02d}" for h in range(24)], y=dd.WEEKDAYS, colorscale=[[i / 7, c] for i, c in enumerate(SEQ_BLUE)],
        xgap=2, ygap=2, colorbar=dict(title=dict(text="карточек<br>в час", font=dict(size=11)), thickness=10,
                                      outlinewidth=0, tickfont=dict(color=MUTED)),
        hovertemplate="%{y}, %{x}:00 — в среднем %{z:.1f} карточек<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
    fig.update_xaxes(title_text="Час суток (МСК)", showline=False, ticks="", type="category",
                     tickvals=[f"{h:02d}" for h in range(0, 24, 3)])
    return base(fig, 300)


def rules_table(d: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """Табличное представление: точные цифры, а не только цвет и длина."""
    cols = ["Правило корреляции", "Карточек", "Доля FP, %", "Доля «Решён», %", "Медиана взятия в работу, мин"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    g = d.groupby("rule").agg(n=("cls", "size"), fp=("cls", lambda s: (s == dd.CLS_FP).mean() * 100),
                              real=("cls", lambda s: (s == dd.CLS_REAL).mean() * 100),
                              react=("react_min", "median"))
    g = g.sort_values("n", ascending=False).head(k).reset_index()
    g.columns = cols
    return g.round({"Доля FP, %": 1, "Доля «Решён», %": 1, "Медиана взятия в работу, мин": 1})


# ================================================================ 3. РАСПРЕДЕЛЕНИЯ
def fig_histogram(d: pd.DataFrame, col: str = "react_min") -> go.Figure:
    """Гистограмма времени: форма распределения и хвосты. Бины равной ширины в log10(мин)."""
    lv = dd.log_minutes(d[col]) if not d.empty else np.array([])
    if len(lv) < 10:
        return empty("Меньше 10 значений — гистограмма не информативна")
    h, e = np.histogram(lv, bins=np.arange(-2, 6.01, 0.2))
    mid = (e[:-1] + e[1:]) / 2
    lo, hi = 10 ** e[:-1], 10 ** e[1:]
    txt = [f"{dd.fmt_minutes(a)} – {dd.fmt_minutes(b)}" for a, b in zip(lo, hi)]
    fig = go.Figure(go.Bar(x=mid, y=h, width=0.19, marker=dict(color=BLUE, cornerradius=2), customdata=txt,
                           hovertemplate="%{customdata}: %{y:,} карточек<extra></extra>"))
    # медиана и 90-й процентиль — по ВСЕМ значениям (вместе с «мгновенными»), как в подписи над графиком
    shown = dd.plotted_minutes(d[col])
    med, p90 = shown.median(), shown.quantile(0.9)
    marks = [(m, lab) for m, lab in [(med, "медиана"), (p90, "90 % быстрее")] if m >= dd.INSTANT_MIN]
    close = len(marks) == 2 and np.log10(marks[1][0]) - np.log10(marks[0][0]) < 1.6
    for i, (m, lab) in enumerate(marks):
        x = np.log10(m)
        # медиана подписывается слева от линии, процентиль — справа; у краёв оси — внутрь графика
        side = "right" if i == 0 else "left"
        if side == "right" and x < -0.6:
            side = "left"
        if side == "left" and x > 4.6:
            side = "right"
        fig.add_vline(x=x, line=dict(color=INK_2, width=1.2))
        fig.add_annotation(x=x, y=1, yref="paper", yanchor="bottom", text=f"{lab}: {dd.fmt_minutes(m)}",
                           showarrow=False, xanchor=side, xshift=-4 if side == "right" else 4,
                           yshift=-18 if (close and i == 1 and side == "right") else 0,
                           bgcolor="rgba(255,255,255,0.85)", font=dict(size=11, color=INK))
    fig.update_xaxes(title_text="Время взятия в работу (логарифмическая шкала)", range=[-2.1, 6], **TIME_TICKS)
    fig.update_yaxes(title_text="Карточек", tickformat=",d")
    return base(fig, 360, margin=dict(l=8, r=16, t=28, b=8), bargap=0)


def fig_qq(counts: pd.Series, log: bool = False) -> tuple[go.Figure, float, int]:
    """QQ-график: число карточек в час против нормального закона.

    counts — ряд только по часам, которые разрешает фильтр периода и типа дня.
    """
    if counts is None or len(counts) == 0:
        return empty(), float("nan"), 0
    x = counts.to_numpy(float)
    if len(x) < 20:
        return empty("Меньше 20 часовых интервалов — выберите больший период"), float("nan"), len(x)
    if log:
        x = np.log1p(x)
    (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
    n = len(osm)
    idx = np.unique(np.r_[np.linspace(0, n - 1, min(n, 800)).astype(int), np.arange(max(0, n - 25), n)])
    fig = go.Figure()
    xs = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="эталон: нормальное распределение",
                             line=dict(color=INK_2, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=osm[idx], y=osr[idx], mode="markers", name="наблюдаемые квантили",
                             marker=dict(color=BLUE, size=6, opacity=0.6, line=dict(width=0)),
                             hovertemplate="теор. %{x:.2f} · факт %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="Теоретические квантили N(0, 1)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="ln(1 + карточек в час)" if log else "Карточек в час")
    return base(fig, 340, legend=True, margin=dict(l=8, r=16, t=36, b=8)), r ** 2, n


KDE_FEATURES = {
    "Длина командной строки (CMDLINE), символов": ("cmd_len", True),
    "Энтропия командной строки (CMDLINE), бит/символ": ("cmd_entropy", False),
    "Задержка SIEM → SOAR, мин": ("lag_min", True),
    "Длительность атаки, мин": ("attack_dur_min", True),
}
_LEN_TICKS = dict(tickvals=[0, 1, 2, 3, 4, 5], ticktext=["1", "10", "100", "1 тыс.", "10 тыс.", "100 тыс."])


def fig_kde(d: pd.DataFrame, feature_label: str) -> tuple[go.Figure, dict]:
    """Сглаженная плотность (KDE) признака у реальных инцидентов и у False Positive."""
    col, is_log = KDE_FEATURES[feature_label]
    rng = np.random.default_rng(RNG_SEED)
    fig, ns, curves = go.Figure(), {}, []
    for cls in [dd.CLS_FP, dd.CLS_REAL]:
        v = pd.to_numeric(d.loc[d["cls"] == cls, col], errors="coerce").dropna().to_numpy(float)
        if is_log:
            v = np.log10(v[v > 0])
        ns[cls] = len(v)
        if len(v) < 50 or np.std(v) == 0:
            continue
        if len(v) > 20000:
            v = rng.choice(v, 20000, replace=False)
        curves.append((cls, v))
    if not curves:
        return empty("Меньше 50 значений в каждом классе — сглаживание ненадёжно"), ns
    lo = min(np.quantile(v, 0.002) for _, v in curves)
    hi = max(np.quantile(v, 0.998) for _, v in curves)
    grid = np.linspace(lo, hi, 300)
    for cls, v in curves:
        y = stats.gaussian_kde(v)(grid)
        fig.add_trace(go.Scatter(x=grid, y=y, mode="lines", name=f"{cls} (n = {dd.fmt_int(ns[cls])})",
                                 line=dict(color=CLASS_COLORS[cls], width=2.5), fill="tozeroy",
                                 fillcolor=_rgba(CLASS_COLORS[cls], 0.15), hoverinfo="skip"))
    if is_log:
        if col in ("lag_min", "attack_dur_min"):
            fig.update_xaxes(**TIME_TICKS)
        else:
            fig.update_xaxes(**_LEN_TICKS)
    fig.update_xaxes(title_text=feature_label + (" (логарифмическая шкала)" if is_log else ""), range=[lo, hi])
    fig.update_yaxes(title_text="Плотность вероятности", rangemode="tozero")
    return base(fig, 340, legend=True, margin=dict(l=8, r=16, t=36, b=8)), ns


VIOLIN_GROUPS = ["Критичность", "Источник инцидента", "Пиктограмма IRP (топ-6)"]


def violin_groups(d: pd.DataFrame, by: str) -> tuple[str, list]:
    if by == "Критичность":
        return "crit", list(dd.CRIT_ORDER)
    if by == "Источник инцидента":
        return "source_grp", dd.SOURCES[1:]
    return "irp", list(d["irp"].value_counts().head(6).index)


def fig_violin(d: pd.DataFrame, by: str) -> go.Figure:
    """Violin (плотность + ящик с усами): время взятия в работу по группам."""
    if d.empty:
        return empty()
    col, groups = violin_groups(d, by)
    color = (lambda g: CRIT_COLORS[g]) if col == "crit" else (lambda g: BLUE)
    rng = np.random.default_rng(RNG_SEED)
    fig, shown = go.Figure(), 0
    for g in groups:
        v = dd.log_minutes(d.loc[d[col] == g, "react_min"])
        if len(v) < 30:                              # по 30 точкам контур скрипки — артефакт сглаживания
            continue
        if len(v) > 5000:
            v = rng.choice(v, 5000, replace=False)
        c = color(g)
        fig.add_trace(go.Violin(
            x=v, y=[_short(g, 28)] * len(v), orientation="h", name=g, width=0.85, spanmode="hard",
            line=dict(color=c, width=1.5), fillcolor=_rgba(c, 0.30), points=False,
            box=dict(visible=True, width=0.25, fillcolor=SURFACE, line=dict(color=INK_2, width=1)),
            meanline=dict(visible=False), hoverinfo="skip"))
        shown += 1
    if not shown:
        return empty("В группах меньше 30 значений")
    fig.update_xaxes(title_text="Время взятия в работу (логарифмическая шкала)", range=[-2.1, 6], **TIME_TICKS,
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2), autorange="reversed", ticklabelstandoff=6)
    return base(fig, 380, violinmode="overlay", violingap=0.1)


def fig_raincloud(d: pd.DataFrame, n_points: int = 400) -> go.Figure:
    """Raincloud: облако (плотность) сверху, ящик на оси, сырые точки снизу — время решения по группам."""
    if d.empty:
        return empty()
    rng = np.random.default_rng(RNG_SEED)
    fig, shown = go.Figure(), 0
    for g in dd.GROUP_ORDER[::-1]:                 # категории снизу вверх: L1 окажется сверху
        v = dd.log_minutes(d.loc[d["group"] == g, "solve_min"])
        if len(v) < 5:
            continue
        n_all = len(v)
        if len(v) > n_points:
            v = rng.choice(v, n_points, replace=False)
        c = GROUP_COLORS[g]
        fig.add_trace(go.Violin(
            x=v, y=[f"{g} (n = {dd.fmt_int(n_all)})"] * len(v), orientation="h", side="positive", width=1.25,
            spanmode="hard", line=dict(color=c, width=1.5), fillcolor=_rgba(c, 0.30), name=g,
            points="all", pointpos=-0.32, jitter=0.18,
            marker=dict(size=3.5, color=c, opacity=0.6, line=dict(width=0)),
            box=dict(visible=True, width=0.1, fillcolor=SURFACE, line=dict(color=INK_2, width=1)),
            meanline=dict(visible=False), hoverinfo="skip"))
        shown += 1
    if not shown:
        return empty("Меньше 5 значений в каждой группе")
    fig.update_xaxes(title_text="Время решения (логарифмическая шкала)", range=[-2.1, 6], **TIME_TICKS,
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2), ticklabelstandoff=6)
    return base(fig, 380, violingap=0, violinmode="overlay")


# ================================================================ 4. ПОТОКИ И ПЛОТНОСТЬ
SANKEY_LEFT = {"Пиктограмма IRP": "irp", "Критичность": "crit", "Источник инцидента": "source_grp",
               "Правило корреляции (топ-8)": "rule"}


def fig_sankey(d: pd.DataFrame, left_label: str, height: int = 460) -> go.Figure:
    """Sankey: сколько карточек каждой категории получили каждый вердикт.
    Узлы расставлены вручную, а подписи вынесены за пределы лент, чтобы текст не лежал поверх потоков."""
    if d.empty:
        return empty()
    col = SANKEY_LEFT[left_label]
    left = d[col].astype(str)
    k = 8 if col in ("rule", "irp") else 10
    top = left.value_counts().head(k).index
    left = left.where(left.isin(top), "Прочие")
    t = pd.crosstab(left, d["cls"]).reindex(columns=dd.CLASS_ORDER, fill_value=0)
    if col == "crit":
        t = t.reindex([c for c in dd.CRIT_ORDER if c in t.index])
    else:
        order = t.sum(axis=1).sort_values(ascending=False).index.tolist()
        if "Прочие" in order:
            order.remove("Прочие")
            order.append("Прочие")
        t = t.loc[order]
    t = t.loc[t.sum(axis=1) > 0]
    if t.empty or float(t.values.sum()) == 0:
        return empty()
    rtot = t.sum(axis=0)
    rnames = [c for c in dd.CLASS_ORDER if rtot[c] > 0]
    lnames, ltot, total = list(t.index), t.sum(axis=1), float(t.values.sum())

    # --- геометрия узлов в пикселях (как считает plotly: общий масштаб по самому плотному столбцу)
    H, pad = height - 16, 14
    ky = (H - (max(len(lnames), len(rnames)) - 1) * pad) / total
    def centers(vals, spread):
        h = np.array(vals, float) * ky
        gap = (H - h.sum()) / (len(h) - 1) if (spread and len(h) > 1) else pad
        y, out = 0.0, []
        for hi in h:
            out.append(y + hi / 2)
            y += hi + gap
        return [c / H for c in out]
    ly, ry = centers(ltot[lnames], True), centers(rtot[rnames], True)
    node_x = [0.001] * len(lnames) + [0.999] * len(rnames)
    node_y = [min(max(v, 0.001), 0.999) for v in ly + ry]

    labels = [f"{n}: {dd.fmt_int(ltot[n])}" for n in lnames] + [f"{n}: {dd.fmt_int(rtot[n])}" for n in rnames]
    node_colors = [GRAY] * len(lnames) + [CLASS_COLORS[n] for n in rnames]
    src, tgt, val, lcol = [], [], [], []
    for i, ln in enumerate(lnames):
        for j, rn in enumerate(rnames):
            v = int(t.loc[ln, rn])
            if v:
                src.append(i)
                tgt.append(len(lnames) + j)
                val.append(v)
                lcol.append(_rgba(CLASS_COLORS[rn], 0.35))
    # Ленты в центре, по бокам — место под подписи. Ширина полей считается по самой длинной подписи
    # (≈ 6,4 px на символ при 11 px, график не уже ≈ 560 px), чтобы текст не обрезался и не ложился на ленты.
    lab_l = [f"{_short(n, 26)}  {dd.fmt_int(ltot[n])}" for n in lnames]
    lab_r = [max(n.split(" (")[0], "(" + n.split(" (")[1] if " (" in n else "", key=len) + f"  {dd.fmt_int(rtot[n])}"
             for n in rnames]
    X0 = min(0.42, (max(map(len, lab_l)) * 6.4 + 12) / 560)
    X1 = max(0.62, 1 - (max(map(len, lab_r)) * 6.4 + 12) / 560)
    fig = go.Figure(go.Sankey(
        arrangement="fixed", valueformat=",d", domain=dict(x=[X0, X1], y=[0, 1]),
        # встроенные подписи пустые (они легли бы поверх лент) — ниже свои подписи снаружи
        node=dict(label=[""] * len(labels), customdata=labels, x=node_x, y=node_y, color=node_colors, pad=pad,
                  thickness=12, line=dict(width=0), hovertemplate="%{customdata}<extra></extra>"),
        link=dict(source=src, target=tgt, value=val, color=lcol,
                  customdata=[f"{labels[a].split(':')[0]} → {labels[b].split(':')[0]}" for a, b in zip(src, tgt)],
                  hovertemplate="%{customdata}<br>%{value:,} карточек<extra></extra>")))
    for n, y in zip(lnames, ly):
        fig.add_annotation(x=X0 - 0.006, y=1 - y, xref="paper", yref="paper", xanchor="right", showarrow=False,
                           text=f"{_short(n, 26)}  <b>{dd.fmt_int(ltot[n])}</b>", font=dict(size=11, color=INK_2))
    for n, y in zip(rnames, ry):
        fig.add_annotation(x=X1 + 0.006, y=1 - y, xref="paper", yref="paper", xanchor="left", showarrow=False,
                           align="left", text=f"{n.replace(' (', '<br>(')}  <b>{dd.fmt_int(rtot[n])}</b>",
                           font=dict(size=11, color=INK_2))
    return base(fig, height, margin=dict(l=4, r=4, t=8, b=8))


def fig_waterfall(prev: pd.DataFrame, cur: pd.DataFrame, n_days: int, top: int = 7) -> go.Figure:
    """Waterfall: как изменился поток карточек и за счёт каких категорий IRP."""
    if prev.empty and cur.empty:
        return empty()
    a = prev["irp"].value_counts()
    b = cur["irp"].value_counts()
    delta = b.reindex(a.index.union(b.index), fill_value=0) - a.reindex(a.index.union(b.index), fill_value=0)
    delta = delta.reindex(delta.abs().sort_values(ascending=False).index)
    head, rest = delta.head(top), delta.iloc[top:].sum()
    names = [f"Предыдущие {n_days} дн."] + [_short(n, 22) for n in head.index] + \
            (["Прочие категории"] if len(delta) > top else []) + [f"Текущие {n_days} дн."]
    vals = [len(prev)] + head.tolist() + ([rest] if len(delta) > top else []) + [len(cur)]
    meas = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
    text = [dd.fmt_int(vals[0])] + [(f"+{dd.fmt_int(v)}" if v > 0 else f"−{dd.fmt_int(-v)}") for v in vals[1:-1]] + \
           [dd.fmt_int(vals[-1])]
    fig = go.Figure(go.Waterfall(
        x=names, y=vals, measure=meas, text=text, textposition="outside", cliponaxis=False,
        textfont=dict(color=INK_2), connector=dict(line=dict(color=AXIS, width=1)),
        increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=GRAY)), hovertemplate="%{x}: %{text}<extra></extra>"))
    top_y = max(np.cumsum(vals[:-1]).max(), vals[0], vals[-1])
    fig.update_yaxes(title_text="Карточек за окно", range=[0, top_y * 1.18], tickformat=",d")
    fig.update_xaxes(tickangle=-35, tickfont=dict(color=INK_2, size=11))
    return base(fig, 460)


# ---------- Hexbin: шестиугольные ячейки (настоящие соты, а не прямоугольники)
def _hexbin(x: np.ndarray, y: np.ndarray, xr, yr, nx: int = 40, aspect: float = 2.8):
    """Шестиугольное бинирование (алгоритм как в matplotlib.hexbin: две смещённые решётки).
    aspect — отношение ширины области построения к высоте в пикселях (график во всю ширину страницы ≈ 2,8):
    при нём соты выглядят правильными шестиугольниками."""
    sx = (xr[1] - xr[0]) / nx
    sy = np.sqrt(3) * sx * aspect * (yr[1] - yr[0]) / (xr[1] - xr[0])
    ix, iy = (x - xr[0]) / sx, (y - yr[0]) / sy
    i1, j1 = np.floor(ix + 0.5), np.floor(iy + 0.5)          # решётка 1: целые узлы
    i2, j2 = np.floor(ix), np.floor(iy)                      # решётка 2: узлы, сдвинутые на (0,5; 0,5)
    d1 = (ix - i1) ** 2 + 3 * (iy - j1) ** 2
    d2 = (ix - i2 - 0.5) ** 2 + 3 * (iy - j2 - 0.5) ** 2
    near1 = d1 < d2
    cu = np.where(near1, i1, i2 + 0.5)
    cv = np.where(near1, j1, j2 + 0.5)
    keys, counts = np.unique(np.c_[cu, cv], axis=0, return_counts=True)
    cx, cy = xr[0] + keys[:, 0] * sx, yr[0] + keys[:, 1] * sy
    # вершины «острого сверху» шестиугольника (как в matplotlib)
    hx = sx * np.array([0.5, 0.5, 0, -0.5, -0.5, 0, 0.5])
    hy = sy / 3 * np.array([-0.5, 0.5, 1, 0.5, -0.5, -1, -0.5])
    return cx, cy, counts, hx, hy


HEX_MODES = ["Время взятия в работу × время решения", "Час суток × карточек за этот час"]


def fig_hexbin(d: pd.DataFrame, mode: str, hourly: pd.Series | None = None) -> tuple[go.Figure, int]:
    if d.empty and mode == HEX_MODES[0]:
        return empty(), 0
    if mode == HEX_MODES[0]:
        t = d[["react_min", "solve_min"]].dropna()
        t = t[(t["react_min"] >= dd.INSTANT_MIN) & (t["solve_min"] >= dd.INSTANT_MIN)]
        x, y = np.log10(t["react_min"].to_numpy()), np.log10(t["solve_min"].to_numpy())
        xr, yr = (-2, 5.5), (-2, 5.5)
        xt, yt = "Время взятия в работу (лог. шкала)", "Время решения (лог. шкала)"
    else:
        if hourly is None or len(hourly) == 0:
            return empty(), 0
        s = hourly.sort_index()
        x = s.index.hour.to_numpy(float) + np.random.default_rng(RNG_SEED).uniform(-0.5, 0.5, len(s))
        y = s.to_numpy(float)
        xr, yr = (-0.5, 23.5), (0, max(10.0, float(np.quantile(y, 0.995)) * 1.05))
        keep = y <= yr[1]
        x, y = x[keep], y[keep]
        xt, yt = "Час суток (МСК)", "Карточек за час"
    n = len(x)
    if n < 100:                                    # при < 100 точках информативнее обычный scatter
        return empty("Меньше 100 точек — для такого объёма подходит обычная диаграмма рассеяния"), n
    cx, cy, cnt, hx, hy = _hexbin(x, y, xr, yr)
    # 7 классов плотности, границы — по логарифмической шкале числа точек
    edges = np.unique(np.round(np.geomspace(1, cnt.max() + 1, 8)).astype(int))
    cls = np.clip(np.searchsorted(edges, cnt, side="right") - 1, 0, len(edges) - 2)
    steps = SEQ_BLUE[1:]
    fig = go.Figure()
    for k in range(len(edges) - 1):
        m = cls == k
        if not m.any():
            continue
        px = np.concatenate([np.r_[cxi + hx, np.nan] for cxi in cx[m]])
        py = np.concatenate([np.r_[cyi + hy, np.nan] for cyi in cy[m]])
        lo, hi = edges[k], edges[k + 1] - 1
        nm = f"≥ {dd.fmt_int(lo)}" if k == len(edges) - 2 else (f"{dd.fmt_int(lo)}" if lo >= hi
                                                                  else f"{dd.fmt_int(lo)}–{dd.fmt_int(hi)}")
        col = steps[min(k * len(steps) // max(1, len(edges) - 1), len(steps) - 1)]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", fill="toself", fillcolor=col, name=nm,
                                 line=dict(color=SURFACE, width=0.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=10, opacity=0), showlegend=False,
                             customdata=cnt, hovertemplate="%{customdata:,} точек в ячейке<extra></extra>"))
    if mode == HEX_MODES[0]:
        fig.update_xaxes(**TIME_TICKS)
        fig.update_yaxes(**TIME_TICKS)
    else:
        fig.update_xaxes(tickvals=list(range(0, 24, 3)), ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])
    fig.update_xaxes(range=list(xr), title_text=xt)
    fig.update_yaxes(range=list(yr), title_text=yt, showgrid=False)
    fig.update_layout(legend=dict(title=dict(text="точек в соте:"), traceorder="normal",
                                  itemclick=False, itemdoubleclick=False))
    return base(fig, 460, legend=True, margin=dict(l=8, r=16, t=44, b=8)), n


def fig_dotplot(top_src: pd.DataFrame) -> go.Figure:
    """Dot Plot: топ-20 src по числу карточек — компактнее столбцов."""
    d = top_src.sort_values("n_cards")
    xs, ys = [], []
    for name, v in zip(d["src"], d["n_cards"]):
        xs += [0, v, None]
        ys += [name, name, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["n_cards"], y=d["src"], mode="markers+text", marker=dict(color=BLUE, size=9),
                             text=[dd.fmt_int(v) for v in d["n_cards"]], textposition="middle right",
                             textfont=dict(color=INK_2, size=11),
                             hovertemplate="%{y}: %{x:,} карточек<extra></extra>"))
    fig.update_xaxes(title_text="Карточек, в которых встречается адрес", range=[0, d["n_cards"].max() * 1.15],
                     tickformat=",d",
                     showgrid=True, gridcolor=GRID, showline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_2, size=11, family="ui-monospace, Menlo, Consolas, monospace"))
    return base(fig, 520)


def fig_bubble(d: pd.DataFrame, k: int = 20) -> go.Figure:
    """Bubble: X — медиана времени решения, Y — доля реальных инцидентов, размер — число карточек."""
    if d.empty:
        return empty()
    g = d.groupby("rule").agg(n=("cls", "size"), solve=("solve_min", "median"),
                              real=("cls", lambda s: (s == dd.CLS_REAL).mean() * 100),
                              crit=("crit_ord", "mean"))
    g = g[g["n"] >= 30].sort_values("n", ascending=False).head(k).dropna(subset=["solve"])
    if len(g) < 3:
        return empty("Меньше 3 правил с ≥ 30 карточками")
    max_d = 46                                       # диаметр самого крупного пузырька, px
    sizeref = 2.0 * g["n"].max() / (max_d ** 2)
    fig = go.Figure(go.Scatter(
        x=g["solve"], y=g["real"], mode="markers", customdata=np.c_[g.index, g["n"], g["crit"]],
        marker=dict(size=g["n"], sizemode="area", sizeref=sizeref, sizemin=4, color=_rgba(BLUE, 0.45),
                    line=dict(color=SURFACE, width=1.5)),
        hovertemplate="%{customdata[0]}<br>карточек: %{customdata[1]:,}<br>медиана решения: %{x:.1f} мин"
                      "<br>доля «Решён»: %{y:.1f} %<br>средняя критичность: %{customdata[2]:.1f} из 4<extra></extra>"))
    xr = [-0.06 * g["solve"].max() - 0.2, g["solve"].max() * 1.12]
    yr = [-0.08 * g["real"].max() - 1, g["real"].max() * 1.18]
    # Подписи: только самые важные пузырьки и только если подпись ни с чем не пересекается.
    # Координаты переводятся в пиксели для типичной области построения ≈ 620 × 400 px.
    W, H = 620, 400
    to_px = lambda x, y: ((x - xr[0]) / (xr[1] - xr[0]) * W, (y - yr[0]) / (yr[1] - yr[0]) * H)
    bubbles = [(*to_px(r["solve"], r["real"]), np.sqrt(r["n"] / sizeref) / 2) for _, r in g.iterrows()]
    order = pd.concat([g.nlargest(2, "real"), g.head(3), g.nlargest(1, "solve"), g.nsmallest(1, "real")])
    placed = []
    for name in dict.fromkeys(order.index):
        r = g.loc[name]
        px, py = to_px(r["solve"], r["real"])
        rad = np.sqrt(r["n"] / sizeref) / 2
        text = _short(name, 28)
        w, h = 6.2 * len(text), 14
        x0 = px if px < 0.2 * W else px - w / 2 if px < 0.8 * W else px - w
        x0 = min(max(x0, 0), W - w)
        box = (x0, py + rad + 3, x0 + w, py + rad + 3 + h)
        if box[3] > H or any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
            continue
        # подпись не должна лежать на чужих пузырьках
        if any(bx - br < box[2] and bx + br > box[0] and by - br < box[3] and by + br > box[1]
               for bx, by, br in bubbles if (bx, by) != (px, py)):
            continue
        placed.append(box)
        fig.add_annotation(x=r["solve"], y=r["real"], text=text, showarrow=False, yshift=rad + 3 + h / 2,
                           xanchor="left" if px < 0.2 * W else "center" if px < 0.8 * W else "right",
                           font=dict(size=10, color=INK_2), bgcolor="rgba(255,255,255,0.8)")
    fig.update_xaxes(title_text="Медиана времени решения, мин", showgrid=True, gridcolor=GRID, range=xr)
    fig.update_yaxes(title_text="Доля карточек «Решён», %", range=yr)
    return base(fig, 480)
