"""
dashboard_charts.py — Plotly-фигуры дашборда «Можно ли доверять модели» (данные -> go.Figure).

Оформление то же, что в task_1:
  * без 3D, теней и рамок; сетка редкая и бледная, только по оси значений;
  * нейтральная серая база + смысловые цвета, одинаковые на всех графиках:
      синий       — False Positive (вердикт аналитика),
      оранжевый   — реальный инцидент, который модель поставила в приоритет,
      тёмно-красный — пропущенный реальный инцидент (главный риск для SOC),
      серый       — приёмка (эталон), синий — эксплуатация (то, что проверяем);
    тройка синий / оранжевый / тёмно-красный проверена на различимость при дальтонизме
    (обычный красный рядом с оранжевым проверку не проходит);
  * у каждой серии есть подпись или легенда — цвет не единственный носитель смысла;
  * одна ось значений на графике: разные единицы — отдельные панели.
WHY — короткий ответ «почему именно этот график» для подписи под заголовком карточки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dashboard_data as dd

# ---------------------------------------------------------------- палитра
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e8e7e1", "#c3c2b7", "#ffffff"
BLUE, ORANGE, MISS = "#2a78d6", "#eb6834", "#a3162b"
BLUE_LIGHT = "#9ec5f4"
GRAY, GRAY_LIGHT, BAND = "#a8a69e", "#d6d4cc", "#f3f2ee"
VAL_COLOR, OPER_COLOR = GRAY, BLUE

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
PLOT_CONFIG = {"displaylogo": False, "responsive": True, "displayModeBar": "hover",
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"]}

WHY = {
    "waterfall": "Каскад показывает путь от всей очереди к приоритетной: сколько карточек модель убрала "
                 "и сколько из убранных оказались реальными инцидентами. Шаги складываются, поэтому видно и выигрыш, и цену.",
    "gains": "Кривая поиска отвечает на вопрос руководителя «сколько карточек надо разобрать, чтобы найти "
             "большую часть инцидентов». Диагональ — очередь без модели; чем выше кривая, тем полезнее сортировка.",
    "sankey": "Санки показывает объёмы связей «вердикт аналитика → решение модели». Ширина ленты — число "
              "карточек, так что согласие и расхождения видны сразу, без чтения матрицы ошибок.",
    "agree": "Линии по месяцам показывают, когда модель начала расходиться с аналитиками. Согласие на "
             "инцидентах и на FP разведено: общая доля согласия маскируется численным перевесом FP.",
    "miss": "Доля пропусков по месяцам с 95 %-ным интервалом: в месяце всего несколько сотен инцидентов, "
            "и без интервала случайный всплеск легко принять за поломку. Пунктир — обещание, данное при приёмке.",
    "missed_rules": "Линейчатая диаграмма ранжирует правила по числу пропусков: длинные названия читаются "
                    "по горизонтали, а подпись «N из M» показывает, частая это ошибка или единичная.",
    "segments": "Точечная диаграмма с интервалами сравнивает доли ошибок по срезам компактнее столбцов. "
                "Срез, чей интервал целиком правее среднего, ошибается систематически, а не случайно.",
    "hist": "Гистограмма оценок модели по вердиктам показывает, где проходят ошибки: пропуски рядом с порогом "
            "лечатся его сдвигом, а пропуски с почти нулевой оценкой — только новыми признаками или переобучением.",
    "calibration": "Диаграмма калибровки отвечает на вопрос «если модель пишет 30 %, это правда 30 %?». "
                   "Точки на диагонали — оценке можно верить как вероятности; размер точки — число карточек.",
    "dumbbell": "Гантели сравнивают одни и те же показатели «на приёмке» и «в работе» на общей шкале 0–100 %: "
                "длина перемычки — насколько изменилось поведение модели.",
    "drift": "Три панели с общей осью времени вместо одного графика с двумя шкалами: качество ранжирования, "
             "доля инцидентов в потоке и разгрузка очереди меняются вместе, и это видно по одной вертикали.",
}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


_AXIS = dict(zeroline=False, automargin=True, title=dict(font=dict(size=12, color=INK_2)),
             tickfont=dict(color=MUTED, size=11))
TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=", ",
    font=dict(family=FONT, size=12, color=INK_2),
    xaxis=dict(showgrid=False, showline=True, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4, **_AXIS),
    yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, **_AXIS),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color=INK_2)),
    bargap=0.3, hovermode="closest",
))


def base(fig: go.Figure, height: int = 360, legend: bool = False, **kw) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=8, r=16, t=12, b=8), showlegend=legend)
    if kw:
        fig.update_layout(**kw)
    return fig


def empty(msg: str = "Нет карточек с вердиктом для выбранных фильтров", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base(fig, height)


def _short(s: str, n: int = 46) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _pct_axis(**kw) -> dict:
    return dict(tickformat=".0%", **kw)


def _month_axis(fig: go.Figure, mon: pd.DataFrame, row=None, col=None, show_labels=True) -> None:
    ticks = [str(m) for m in mon["month"]]
    text = [dd.month_label(m, short=True) for m in mon["month"]]
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=ticks, tickvals=ticks,
                     ticktext=text if show_labels else [""] * len(ticks), row=row, col=col)


def _band_span(mon: pd.DataFrame) -> tuple[float, float] | None:
    """Границы месяцев приёмки на категориальной оси: индексы ± половина шага."""
    idx = np.flatnonzero((mon["part"] == dd.VAL).to_numpy())
    return (idx[0] - 0.5, idx[-1] + 0.5) if len(idx) else None


def _sel_index(mon: pd.DataFrame, sel: pd.Period | None) -> int | None:
    if sel is None:
        return None
    idx = np.flatnonzero((mon["month"] == sel).to_numpy())
    return int(idx[0]) if len(idx) else None


def _val_band(fig: go.Figure, mon: pd.DataFrame, sel: pd.Period | None, label: bool = True, **rc) -> None:
    """Серая подложка под месяцами приёмки и рамка выбранного месяца эксплуатации."""
    span = _band_span(mon)
    if span:
        fig.add_vrect(x0=span[0], x1=span[1], fillcolor=BAND, line_width=0, layer="below",
                      **(dict(annotation_text="приёмка", annotation_position="top left",
                              annotation_font=dict(size=11, color=MUTED)) if label else {}), **rc)
    i = _sel_index(mon, sel)
    if i is not None:
        fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor="rgba(0,0,0,0)", line=dict(color=AXIS, width=1, dash="dot"),
                      layer="below", **rc)


# ================================================================ 1. ИТОГИ
def fig_queue_waterfall(c: dd.Confusion) -> go.Figure:
    """Каскад очереди: все карточки с вердиктом -> минус верно отложенные FP -> минус ошибочно отложенные
    реальные инциденты -> приоритетная очередь. Шаг с пропусками окрашен отдельно: это цена разгрузки."""
    if c.n == 0:
        return empty()
    keep = c.tp + c.fp
    x = ["Все карточки<br>с вердиктом", "Отложены верно:<br>False Positive",
         "Отложены ошибочно:<br>реальные инциденты", "Осталось<br>в приоритете"]
    bases = [0, c.n - c.tn, keep, 0]
    heights = [c.n, c.tn, c.fn, keep]
    text = [dd.fmt_int(c.n), "−" + dd.fmt_int(c.tn), "−" + dd.fmt_int(c.fn),
            f"{dd.fmt_int(keep)} ({dd.fmt_pct(keep / c.n, 0)})"]
    fig = go.Figure(go.Bar(x=x, y=heights, base=bases, marker=dict(color=[GRAY, BLUE_LIGHT, MISS, BLUE], line=dict(width=0)),
                           text=text, textposition="outside", cliponaxis=False,
                           textfont=dict(color=[INK, INK, MISS, INK], size=12),
                           hovertemplate="%{x}: %{text} карточек<extra></extra>"))
    for i, lvl in enumerate([c.n, c.n - c.tn, keep]):          # соединители между шагами
        fig.add_shape(type="line", x0=i + 0.35, x1=i + 0.65, y0=lvl, y1=lvl, line=dict(color=AXIS, width=1))
    fig.update_xaxes(tickfont=dict(size=11, color=INK_2))
    fig.update_yaxes(title_text="Карточек", tickformat=",d", range=[0, c.n * 1.15])
    return base(fig, 360, bargap=0.35, margin=dict(l=8, r=16, t=24, b=8))


def fig_gains(val: pd.DataFrame, oper: pd.DataFrame, thr: float) -> go.Figure:
    """Кривая поиска инцидентов: приёмка и эксплуатация + диагональ «без модели» + рабочая точка порога."""
    gv, go_ = dd.gains(val), dd.gains(oper)
    if go_.empty:
        return empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="без модели (случайный порядок)",
                             line=dict(color=GRAY_LIGHT, width=1.5, dash="dash"), hoverinfo="skip"))
    if not gv.empty:
        fig.add_trace(go.Scatter(x=gv["x"], y=gv["y"], mode="lines", name="приёмка",
                                 line=dict(color=VAL_COLOR, width=2),
                                 hovertemplate="приёмка: разобрано %{x:.0%} очереди → найдено %{y:.0%} инцидентов<extra></extra>"))
    fig.add_trace(go.Scatter(x=go_["x"], y=go_["y"], mode="lines", name="эксплуатация",
                             line=dict(color=OPER_COLOR, width=2.5),
                             hovertemplate="эксплуатация: разобрано %{x:.0%} очереди → найдено %{y:.0%} инцидентов<extra></extra>"))
    for d, color, name in [(val, VAL_COLOR, "приёмка"), (oper, OPER_COLOR, "эксплуатация")]:
        c = dd.confusion(d)
        if c.n and c.pos:
            x, y = 1 - c.low_share, c.recall
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers", showlegend=False,
                marker=dict(size=11, color=color, line=dict(color=SURFACE, width=2)),
                hovertemplate=f"порог модели ({name}): в приоритете {x:.0%} очереди, найдено {y:.1%} инцидентов<extra></extra>"))
            fig.add_annotation(x=x, y=y, text=f"порог: {dd.fmt_pct(x, 0)} очереди → {dd.fmt_pct(y, 0)}",
                               showarrow=True, arrowhead=0, arrowcolor=AXIS, ax=-24,
                               ay=36 if name == "приёмка" else 110, font=dict(size=11, color=INK_2),
                               xanchor="right", bgcolor="rgba(255,255,255,0.85)")
    fig.update_xaxes(title_text="Доля очереди, разобранной сверху вниз по оценке модели", range=[0, 1],
                     **_pct_axis(dtick=0.2))
    fig.update_yaxes(title_text="Доля найденных реальных инцидентов", range=[0, 1.02], **_pct_axis(dtick=0.2))
    return base(fig, 360, legend=True, margin=dict(l=8, r=16, t=36, b=8))


# ================================================================ 2. СОГЛАСИЕ
def fig_sankey(c: dd.Confusion) -> go.Figure:
    """Вердикт аналитика (слева) -> решение модели (справа); ширина ленты = число карточек."""
    if c.n == 0:
        return empty()
    labels = [f"Реальный инцидент<br>{dd.fmt_int(c.pos)}", f"False Positive<br>{dd.fmt_int(c.neg)}",
              f"Модель: в приоритет<br>{dd.fmt_int(c.tp + c.fp)}", f"Модель: низкий приоритет<br>{dd.fmt_int(c.fn + c.tn)}"]
    links = [
        (0, 2, c.tp, _rgba(ORANGE, 0.55), "найден"),
        (0, 3, c.fn, _rgba(MISS, 0.75), "ПРОПУЩЕН"),
        (1, 2, c.fp, _rgba(BLUE, 0.22), "лишняя работа"),
        (1, 3, c.tn, _rgba(BLUE, 0.5), "отложен верно"),
    ]
    links = [lk for lk in links if lk[2] > 0]
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=labels, pad=26, thickness=16, color=[ORANGE, BLUE, GRAY, GRAY_LIGHT],
                  line=dict(width=0), x=[0.001, 0.001, 0.999, 0.999], y=[0.15, 0.65, 0.4, 0.92],
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=[lk[0] for lk in links], target=[lk[1] for lk in links], value=[lk[2] for lk in links],
                  color=[lk[3] for lk in links], customdata=[lk[4] for lk in links],
                  hovertemplate="%{source.label} → %{target.label}<br>%{customdata}: %{value:,} карточек<extra></extra>"),
        textfont=dict(family=FONT, size=12, color=INK),
    ))
    fig.add_annotation(x=0.5, y=-0.04, xref="paper", yref="paper", showarrow=False, yanchor="top",
                       text=f"<b style='color:{MISS}'>пропущено {dd.fmt_int(c.fn)} инцидентов</b> · "
                            f"FP в приоритете: {dd.fmt_int(c.fp)} · отложено верно: {dd.fmt_int(c.tn)}",
                       font=dict(size=12, color=INK_2))
    return base(fig, 400, margin=dict(l=8, r=8, t=8, b=44))


def fig_agreement_monthly(mon: pd.DataFrame, sel: pd.Period | None = None) -> go.Figure:
    """Согласие с аналитиками по месяцам: отдельно на реальных инцидентах и на False Positive."""
    if mon.empty:
        return empty()
    x = [str(m) for m in mon["month"]]
    fig = go.Figure()
    for col, name, color, dash in [("recall", "на инцидентах", ORANGE, "solid"),
                                   ("spec", "на FP", BLUE, "solid")]:
        fig.add_trace(go.Scatter(
            x=x, y=mon[col], mode="lines+markers", name=name, line=dict(color=color, width=2, dash=dash),
            marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)),
            customdata=np.c_[[dd.month_label(m) for m in mon["month"]]],
            hovertemplate="%{customdata[0]}: модель согласна в %{y:.1%} случаев " + name + "<extra></extra>"))
        fig.add_annotation(x=x[-1], y=mon[col].iloc[-1], text=name, showarrow=False, xanchor="left", xshift=8,
                           font=dict(size=11, color=INK_2))
    _val_band(fig, mon, sel)
    _month_axis(fig, mon)
    fig.update_yaxes(title_text="Модель согласна с аналитиком", range=[0, 1.05], **_pct_axis(dtick=0.2))
    fig.update_xaxes(tickangle=0)
    return base(fig, 360, margin=dict(l=8, r=96, t=28, b=8), hovermode="x unified")


# ================================================================ 3. ПРОПУСКИ
def fig_miss_monthly(mon: pd.DataFrame, target_miss: float, sel: pd.Period | None = None) -> go.Figure:
    """Доля пропущенных реальных инцидентов по месяцам + 95 %-ный интервал + цель приёмки."""
    if mon.empty:
        return empty()
    x = [str(m) for m in mon["month"]]
    over = mon["miss_lo"] > target_miss
    colors = [MISS if o else (VAL_COLOR if p == dd.VAL else OPER_COLOR) for o, p in zip(over, mon["part"])]
    fig = go.Figure()
    fig.add_hline(y=target_miss, line=dict(color=MUTED, width=1.5, dash="dash"),
                  annotation_text=f"цель приёмки: не больше {dd.fmt_pct(target_miss, 0)}",
                  annotation_position="top right", annotation_font=dict(size=11, color=INK_2))
    fig.add_trace(go.Scatter(
        x=x, y=mon["miss"], mode="markers+text", showlegend=False,
        error_y=dict(type="data", symmetric=False, array=mon["miss_hi"] - mon["miss"],
                     arrayminus=mon["miss"] - mon["miss_lo"], color=AXIS, thickness=1.5, width=6),
        marker=dict(size=11, color=colors, line=dict(color=SURFACE, width=2)),
        text=[f"{dd.fmt_int(fn)} из {dd.fmt_int(p)}" for fn, p in zip(mon["fn"], mon["pos"])],
        textposition="middle right", textfont=dict(size=11, color=INK_2), cliponaxis=False,
        customdata=np.c_[[dd.month_label(m) for m in mon["month"]], mon["fn"], mon["pos"], mon["miss_lo"], mon["miss_hi"]],
        hovertemplate=("%{customdata[0]}: пропущено %{customdata[1]} из %{customdata[2]} реальных инцидентов "
                       "(%{y:.1%}; 95 %-ный интервал %{customdata[3]:.1%} — %{customdata[4]:.1%})<extra></extra>")))
    for name, color in [("приёмка", VAL_COLOR), ("эксплуатация", OPER_COLOR), ("хуже цели даже с учётом интервала", MISS)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, marker=dict(size=10, color=color)))
    _val_band(fig, mon, sel)
    _month_axis(fig, mon)
    top = max(float(mon["miss_hi"].max()), target_miss) * 1.15
    fig.update_yaxes(title_text="Доля пропущенных реальных инцидентов", range=[0, top], **_pct_axis())
    return base(fig, 380, legend=True, margin=dict(l=8, r=64, t=36, b=8))


def fig_missed_rules(d: pd.DataFrame, top: int = 10) -> go.Figure:
    """Правила SIEM с наибольшим числом пропущенных реальных инцидентов."""
    inc = d[d["y"] == 1]
    miss = inc[~inc["pred"]]
    if miss.empty:
        return empty("Пропущенных реальных инцидентов нет")
    g = (inc.groupby("rule").agg(pos=("pred", "size"), fn=("pred", lambda s: int((~s).sum())))
         .query("fn > 0").sort_values(["fn", "pos"], ascending=[False, True]).head(top).iloc[::-1])
    fig = go.Figure(go.Bar(
        y=[_short(r, 52) for r in g.index], x=g["fn"], orientation="h", marker=dict(color=MISS, line=dict(width=0)),
        text=[f"{fn} из {dd.fmt_int(p)}" for fn, p in zip(g["fn"], g["pos"])], textposition="outside",
        cliponaxis=False, textfont=dict(size=11, color=INK_2), customdata=np.c_[g.index, g["pos"]],
        hovertemplate="%{customdata[0]}<br>пропущено %{x} из %{customdata[1]} реальных инцидентов<extra></extra>"))
    fig.update_xaxes(title_text="Пропущено реальных инцидентов", showgrid=True, gridcolor=GRID, showline=False,
                     tickformat=",d", range=[0, g["fn"].max() * 1.25])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 360, bargap=0.35, margin=dict(l=8, r=24, t=8, b=8))


# ================================================================ 4. ЗОНЫ ОШИБОК
def fig_segments(tbl: pd.DataFrame, metric: str) -> go.Figure:
    """Доля ошибок по уровням среза: точка = доля, усы = 95 %-ный интервал, вертикаль = среднее по срезу."""
    if tbl is None or tbl.empty:
        return empty("Недостаточно карточек для сравнения срезов")
    t = tbl.iloc[::-1]
    overall = float(t["overall"].iloc[0])
    is_miss = metric == "miss"
    k_col = "fn" if is_miss else "fp"
    focus = MISS if is_miss else BLUE
    colors = [focus if s else (GRAY_LIGHT if b else GRAY) for s, b in zip(t["systematic"], t["better"])]
    what = "реальных инцидентов пропущено" if is_miss else "FP попали в приоритет"
    fig = go.Figure()
    fig.add_vline(x=overall, line=dict(color=MUTED, width=1.5, dash="dash"),
                  annotation_text=f"в среднем {dd.fmt_pct(overall)}", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=INK_2))
    fig.add_trace(go.Scatter(
        y=[_short(s, 44) for s in t["seg"]], x=t["rate"], mode="markers", showlegend=False,
        error_x=dict(type="data", symmetric=False, array=t["hi"] - t["rate"], arrayminus=t["rate"] - t["lo"],
                     color=AXIS, thickness=1.5, width=0),
        marker=dict(size=11, color=colors, line=dict(color=SURFACE, width=2)),
        customdata=np.c_[t["seg"], t[k_col], t["n"], t["lo"], t["hi"],
                         np.where(t["systematic"], "выше среднего — систематически",
                                  np.where(t["better"], "ниже среднего", "в пределах случайности"))],
        hovertemplate=("%{customdata[0]}<br>%{customdata[1]} из %{customdata[2]} " + what +
                       " (%{x:.1%}; интервал %{customdata[3]:.1%} — %{customdata[4]:.1%})<br>%{customdata[5]}<extra></extra>")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="систематически выше среднего",
                             marker=dict(size=10, color=focus)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="в пределах случайности",
                             marker=dict(size=10, color=GRAY)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="ниже среднего",
                             marker=dict(size=10, color=GRAY_LIGHT)))
    hi = float(t["hi"].max())
    fig.update_xaxes(title_text=("Доля пропущенных среди реальных инцидентов среза" if is_miss
                                 else "Доля FP, которые модель поставила в приоритет"),
                     range=[0, min(1.0, hi * 1.08) if hi > 0 else 0.1], showgrid=True, gridcolor=GRID,
                     showline=False, **_pct_axis())
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, max(300, 70 + 34 * len(t)), legend=True, margin=dict(l=8, r=24, t=36, b=8))


def fig_score_hist(d: pd.DataFrame, thr: float) -> go.Figure:
    """Распределение оценок модели (лог-шкала) по вердиктам аналитика; пропуски — отдельным цветом."""
    lab = d[d["y"] >= 0]
    if lab.empty:
        return empty()
    edges = np.logspace(-3, 0, 43)
    lp = np.log10(np.clip(lab["proba"].to_numpy(), 1e-3, 1))
    centers = (np.log10(edges[:-1]) + np.log10(edges[1:])) / 2
    y = lab["y"].to_numpy()
    groups = [
        ("False Positive", y == 0, BLUE_LIGHT),
        ("реальный инцидент: найден", (y == 1) & (lab["proba"].to_numpy() >= thr), ORANGE),
        ("реальный инцидент: пропущен", (y == 1) & (lab["proba"].to_numpy() < thr), MISS),
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5])
    for i, (name, m, color) in enumerate(groups):
        cnt, _ = np.histogram(lp[m], bins=np.log10(edges))
        row = 1 if i == 0 else 2
        fig.add_trace(go.Bar(x=centers, y=cnt, name=name, marker=dict(color=color, line=dict(width=0)),
                             width=(np.log10(edges[1]) - np.log10(edges[0])) * 0.9,
                             customdata=np.c_[edges[:-1], edges[1:]],
                             hovertemplate=name + ": %{y:,} карточек с оценкой %{customdata[0]:.1%} — %{customdata[1]:.1%}<extra></extra>"),
                      row=row, col=1)
    for row in (1, 2):
        fig.add_vline(x=np.log10(thr), line=dict(color=INK_2, width=1.5, dash="dash"), row=row, col=1)
    fig.add_annotation(x=np.log10(thr), y=1, yref="paper", text=f"порог {dd.fmt_pct(thr)}", showarrow=False,
                       xanchor="left", xshift=6, yanchor="bottom", font=dict(size=11, color=INK_2))
    fig.add_annotation(x=np.log10(thr) - 0.05, y=0.97, yref="paper", text="← низкий приоритет", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(size=11, color=MUTED))
    fig.add_annotation(x=np.log10(thr) + 0.05, y=0.97, yref="paper", text="в приоритет →", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=MUTED))
    near, n_miss = dd.near_threshold_share(lab.assign(pred=lab["proba"] >= thr), thr)
    if n_miss:
        fig.add_annotation(x=np.log10(thr) - 0.05, y=0.3, yref="paper", xanchor="right", showarrow=False, align="right",
                           text=f"<b style='color:{MISS}'>пропущено {n_miss}</b>, из них {near}<br>с оценкой не ниже "
                                f"половины порога", font=dict(size=11, color=INK_2))
    lo = float(np.floor(np.log10(max(1e-3, np.quantile(lab["proba"], 0.001))) * 2) / 2)
    tv = [v for v in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1] if np.log10(v) >= lo - 1e-9]
    fig.update_xaxes(tickvals=np.log10(tv), ticktext=[dd.fmt_pct(v, 1 if v < 0.01 else 0) for v in tv], row=2, col=1,
                     title_text="Оценка модели: вероятность реального инцидента (логарифмическая шкала)")
    fig.update_xaxes(range=[lo, 0], showline=True, linecolor=AXIS)
    fig.update_yaxes(title_text="FP", tickformat=",d", row=1, col=1)
    fig.update_yaxes(title_text="Инциденты", tickformat=",d", row=2, col=1)
    return base(fig, 400, legend=True, barmode="overlay", margin=dict(l=8, r=16, t=36, b=8))


# ================================================================ 5. ДОВЕРИЕ К ОЦЕНКЕ
def fig_calibration(val: pd.DataFrame, oper: pd.DataFrame) -> go.Figure:
    """Средняя оценка модели в корзине против фактической доли реальных инцидентов."""
    cv, co = dd.calibration(val), dd.calibration(oper)
    if co.empty and cv.empty:
        return empty()
    top = float(max(cv["pred"].max() if len(cv) else 0, co["pred"].max() if len(co) else 0,
                    cv["hi"].max() if len(cv) else 0, co["hi"].max() if len(co) else 0))
    top = min(1.0, top * 1.08 + 0.02)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="оценке можно верить как вероятности",
                             line=dict(color=GRAY_LIGHT, width=1.5, dash="dash"), hoverinfo="skip"))
    nmax = max(cv["n"].max() if len(cv) else 1, co["n"].max() if len(co) else 1)
    for t, name, color in [(cv, "приёмка", VAL_COLOR), (co, "эксплуатация", OPER_COLOR)]:
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["pred"], y=t["actual"], mode="lines+markers", name=name, line=dict(color=color, width=2),
            marker=dict(size=8 + 22 * np.sqrt(t["n"] / nmax), color=color, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=t["hi"] - t["actual"], arrayminus=t["actual"] - t["lo"],
                         color=_rgba(color, 0.6), thickness=1.2, width=0),
            customdata=np.c_[t["n"], t["k"]],
            hovertemplate=(name + ": модель в среднем пишет %{x:.1%}, на деле инцидентов %{y:.1%} "
                           "(%{customdata[1]:,} из %{customdata[0]:,} карточек)<extra></extra>")))
    fig.update_xaxes(title_text="Что пишет модель: средняя оценка в группе карточек", range=[0, top], **_pct_axis())
    fig.update_yaxes(title_text="Что оказалось: доля реальных инцидентов", range=[0, top], **_pct_axis())
    return base(fig, 380, legend=True, margin=dict(l=8, r=16, t=36, b=8))


def fig_dumbbell(rows: list[tuple[str, float, float]]) -> go.Figure:
    """rows: (показатель, приёмка, эксплуатация) — доли 0..1."""
    rows = [r for r in rows if np.isfinite(r[1]) and np.isfinite(r[2])]
    if not rows:
        return empty()
    names = [r[0] for r in rows][::-1]
    v = [r[1] for r in rows][::-1]
    o = [r[2] for r in rows][::-1]
    fig = go.Figure()
    for n, a, b in zip(names, v, o):
        fig.add_shape(type="line", x0=a, x1=b, y0=n, y1=n, line=dict(color=GRAY_LIGHT, width=3), layer="below")
    fig.add_trace(go.Scatter(x=v, y=names, mode="markers", name="приёмка",
                             marker=dict(size=13, color=VAL_COLOR, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y}<br>приёмка: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=o, y=names, mode="markers+text", name="эксплуатация",
                             marker=dict(size=13, color=OPER_COLOR, line=dict(color=SURFACE, width=2)),
                             text=[dd.fmt_pct(x, 0) for x in o],
                             textposition=["middle right" if b >= a else "middle left" for a, b in zip(v, o)],
                             textfont=dict(size=11, color=INK), cliponaxis=False,
                             hovertemplate="%{y}<br>эксплуатация: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.08], showgrid=True, gridcolor=GRID, showline=False, **_pct_axis(dtick=0.2))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=INK_2))
    return base(fig, 300, legend=True, margin=dict(l=8, r=24, t=36, b=8))


def fig_drift(mon: pd.DataFrame, sel: pd.Period | None = None) -> go.Figure:
    """Три панели с общей осью месяцев: AUC, доля реальных инцидентов в потоке, доля очереди в низком приоритете."""
    if mon.empty:
        return empty()
    panels = [("auc", "Качество ранжирования (AUC): насколько инциденты оценены выше FP", OPER_COLOR, [0.3, 1.0]),
              ("inc_rate", "Доля реальных инцидентов в потоке", ORANGE, None),
              ("low_share", "Очередь, которую модель отложила (низкий приоритет)", INK_2, None)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=[p[1] for p in panels])
    x = [str(m) for m in mon["month"]]
    labels = [dd.month_label(m) for m in mon["month"]]
    for i, (col, title, color, rng) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=x, y=mon[col], mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=7, color=color, line=dict(color=SURFACE, width=1.5)),
                                 customdata=labels, showlegend=False,
                                 hovertemplate="%{customdata}: " + title.lower() + " %{y:.1%}<extra></extra>"
                                 if col != "auc" else "%{customdata}: AUC %{y:.2f}<extra></extra>"),
                      row=i, col=1)
        if col == "auc":
            fig.add_hline(y=0.5, line=dict(color=GRAY_LIGHT, width=1, dash="dot"), row=i, col=1,
                          annotation_text="0,5 — как монетка", annotation_position="bottom right",
                          annotation_font=dict(size=10, color=MUTED))
            fig.update_yaxes(range=rng, tickformat=".1f", row=i, col=1)
        else:
            top = float(mon[col].max()) * 1.2 if mon[col].max() > 0 else 0.1
            fig.update_yaxes(range=[0, top], tickformat=".0%", nticks=4, row=i, col=1)
        _val_band(fig, mon, sel, label=(i == 1), row=i, col=1)
        _month_axis(fig, mon, row=i, col=1, show_labels=(i == 3))
    for a in fig.layout.annotations[:3]:   # заголовки панелей — первые три аннотации make_subplots
        a.update(font=dict(size=12, color=INK_2), x=0, xanchor="left")
    fig.update_xaxes(showline=True, linecolor=AXIS)
    return base(fig, 460, margin=dict(l=8, r=16, t=28, b=8))
