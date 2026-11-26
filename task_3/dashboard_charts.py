"""
dashboard_charts.py — Plotly-фигуры дашборда задания 3 (SHAP / XAI) и словарь обоснований WHY.

Наследует стиль, палитру (синий FP, оранжевый инцидент, тёмно-красный пропуск, серый эталон)
и шаблон TEMPLATE из task_2/dashboard_charts.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dashboard_data as dd
from feature_registry import FEATURE_ORDER_73, REGISTRY

# ---------------------------------------------------------------- палитра (идентична task_2)
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e8e7e1", "#c3c2b7", "#ffffff"
BLUE, ORANGE, MISS = "#2a78d6", "#eb6834", "#a3162b"
BLUE_LIGHT, ORANGE_LIGHT = "#9ec5f4", "#f7b599"
GRAY, GRAY_LIGHT, BAND = "#a8a69e", "#d6d4cc", "#f3f2ee"
VAL_COLOR, OPER_COLOR = GRAY, BLUE

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
PLOT_CONFIG = {
    "displaylogo": False, "responsive": True, "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
}

# Ответы «Почему выбран именно этот график» и «Почему не альтернатива» (для REPORT.md)
WHY = {
    "card_waterfall": "Каскадная диаграмма (Waterfall) показывает аддитивный путь решения модели от базового уровня "
                      "к итоговому скору через Top-N полей и группу «Прочие»; вертикаль порога τ сразу показывает, "
                      "какой именно признак перевёл карточку в приоритет или оставил в шуме.",
    "card_balance": "Дивергентная линейчатая диаграмма сводит суммарные положительные («к инциденту») и "
                    "отрицательные («к FP») вклады для всех 73 полей и для выбранного Top-N, сразу показывая "
                    "баланс аргументов модели и полноту покрытия Top-N.",
    "sim_hexmap": "Двухслойная карта с гексагональным бинированием (Hexbin) агрегирует ~41,7 тыс. карточек "
                  "в ~1 000 ячеек без перегрузки Plotly и наслоения точек, а поверх показывает 8 карточек "
                  "команды 5 и истинных 20-NN из многомерного пространства.",
    "group_violin": "Скрипичная диаграмма с сырыми точками (Violin + strip) показывает форму распределения, "
                    "медиану и выбросы вкладов SHAP в группе похожих карточек, а звезда текущей карточки "
                    "выявляет, типично ли её объяснение для окружения.",
    "beeswarm": "SHAP Summary Plot (Beeswarm) одновременно кодирует глобальную важность поля, направление "
                "вклада и зависимость знака вклада от значения признака (высокое/низкое или заполнено/пропуск).",
    "global_dot": "Точечная диаграмма категорий (Dot Plot) сравнивает важность полей в трёх выборках "
                  "(Приёмка, Эксплуатация, Практика 320) компактнее сгруппированных столбцов, а вторая панель "
                  "проверяет устойчивость рейтинга независимым методом Ablation (Δ вероятности).",
    "ale_pdp_ice": "ALE (Accumulated Local Effects) оценивает локальное влияние числового признака без "
                   "экстраполяции в нереалистичные комбинации при мультиколлинеарности; ICE-кривые "
                   "показывают индивидуальное поведение карточек команды 5 поверх среднего эффекта.",
    "cat_gpdp": "Групповой PDP (what-if) подставляет категориальное поле целиком, исключая невозможные "
                "комбинации one-hot столбцов; точечная диаграмма сравнивает сдвиг вероятности на приёмке и в проде.",
    "drift_dumbbell": "Диаграмма гантелей (Dumbbell) показывает направление и величину изменения важности "
                      "признаков между приёмкой и эксплуатацией с минимальным визуальным шумом и меткой PSI.",
    "drift_dist": "Сопоставление распределений «Приёмка vs Эксплуатация» (гистограмма для числовых бинов и "
                  "гантели долей для категорий с выделенной строкой пропусков) объясняет причину дрейфа вкладов.",
    "verify_wilson": "Двухпанельное сопоставление SHAP-вкладов кейса с фактической долей реальных инцидентов "
                     "и 95 %-ным интервалом Уилсона переводит XAI из гипотезы в статистически проверяемый факт.",
}

WHY_ALT = {
    "card_waterfall": "обычная столбчатая диаграмма: не показывает стартовый baseline, накопление итога и пересечение порога τ",
    "card_balance": "круговая диаграмма: не поддерживает разнонаправленные (положительные и отрицательные) величины",
    "sim_hexmap": "обычный scatter plot: при десятках тысяч карточек точки сливаются в нечитаемое пятно и тормозят браузер (методичка, 6.1)",
    "group_violin": "столбцы средних: скрывают бимодальность и разброс вкладов внутри похожих карточек",
    "beeswarm": "один рейтинг mean|SHAP|: показывает силу признака, но скрывает направление влияния и нелинейность",
    "global_dot": "сгруппированные столбцы: втрое больше «чернил», длинные имена полей SOC хуже читаются (методичка, 6.3)",
    "ale_pdp_ice": "один PDP без ALE: при сильной мультиколлинеарности признаков SOC искажает маргинальный эффект (методичка, 5.3)",
    "cat_gpdp": "обычный PDP по OHE-колонке: создаёт невозможные карточки с двумя активными категориями одновременно",
    "drift_dumbbell": "сгруппированные столбцы: хуже читается величина и знак сдвига, удваивается визуальная нагрузка",
    "drift_dist": "одно число PSI без профиля: показывает масштаб сдвига, но не объясняет, какие именно уровни или пропуски выросли",
    "verify_wilson": "один Waterfall без статистики данных: показывает логику модели, но не отвечает, подтверждается ли она реальностью",
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


def base(fig: go.Figure, height: int = 380, legend: bool = False, **kw) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=8, r=20, t=16, b=10), showlegend=legend)
    if kw:
        fig.update_layout(**kw)
    return fig


def empty(msg: str = "Нет данных для выбранных параметров", height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base(fig, height)


def _short(s: str, n: int = 40) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ================================================================ 1. ПОЧЕМУ ТАК? (КАРТОЧКА)
def fig_card_waterfall(slc: dict, tau: float) -> go.Figure:
    """Горизонтальный каскад SHAP (log-odds): Baseline -> Top-N -> Прочие -> Итог + линия порога τ."""
    top = slc["top_df"]
    if top.empty:
        return empty()

    base_v = slc["base"]
    raw_v = slc["raw_score"]
    logit_tau = dd.logit(tau)

    labels, starts, widths, colors, texts, hovers = [], [], [], [], [], []
    cur = base_v
    cum_pts = [base_v, logit_tau, raw_v]

    for r in top.itertuples(index=False):
        dv = float(r.shap_val)
        labels.append(_short(r.display_name, 38))
        starts.append(cur)
        widths.append(dv)
        colors.append(ORANGE if dv >= 0 else BLUE)
        texts.append(dd.fmt_signed(dv, 2))
        hovers.append(f"<b>{r.display_name}</b><br>Значение: {_short(str(r.hover_val), 80)}<br>"
                      f"Вклад SHAP: {dd.fmt_signed(dv, 3)} log-odds<br>После шага: {cur + dv:.2f} (p={dd.fmt_proba(dd.sigmoid(cur + dv))})")
        cur += dv
        cum_pts.append(cur)

    # Шаг «Прочие»
    ov = float(slc["other_val"])
    labels.append(slc["other_label"])
    starts.append(cur)
    widths.append(ov)
    colors.append(GRAY if abs(ov) < 0.05 else (ORANGE_LIGHT if ov > 0 else BLUE_LIGHT))
    texts.append(dd.fmt_signed(ov, 2))
    hovers.append(f"<b>{slc['other_label']}</b><br>Суммарный вклад: {dd.fmt_signed(ov, 3)} log-odds")
    cur += ov
    cum_pts.append(cur)

    # Инвертируем порядок, чтобы Top-1 был сверху
    labels = labels[::-1]
    starts = starts[::-1]
    widths = widths[::-1]
    colors = colors[::-1]
    texts = texts[::-1]
    hovers = hovers[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=widths, base=starts, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=texts, textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=INK_2),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))

    # Соединительные тонкие линии каскада
    for i in range(len(labels) - 1):
        end_x = starts[i + 1] + widths[i + 1]
        fig.add_shape(type="line", x0=end_x, x1=end_x, y0=i + 0.35, y1=i + 0.65,
                      line=dict(color=GRAY_LIGHT, width=1))

    # Опорные вертикали: Baseline, Порог τ, Итог
    fig.add_vline(x=base_v, line=dict(color=GRAY, width=1.5, dash="dot"),
                  annotation_text=f"baseline {dd.fmt_num(base_v, 2)} ({dd.fmt_proba(dd.sigmoid(base_v))})",
                  annotation_position="bottom left", annotation_font=dict(size=10, color=MUTED))
    fig.add_vline(x=logit_tau, line=dict(color=MISS, width=1.8, dash="dash"),
                  annotation_text=f"порог τ={dd.fmt_proba(tau)} ({dd.fmt_num(logit_tau, 2)})",
                  annotation_position="top right", annotation_font=dict(size=11, color=MISS))
    fig.add_vline(x=raw_v, line=dict(color=INK, width=1.8, dash="solid"),
                  annotation_text=f"итог {dd.fmt_num(raw_v, 2)} ({dd.fmt_proba(slc['proba'])})",
                  annotation_position="top left", annotation_font=dict(size=11, color=INK))

    # Легенда направлений
    for name, col in [("в сторону инцидента (+)", ORANGE), ("в сторону FP (−)", BLUE)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, marker=dict(size=10, symbol="square", color=col)))

    lo, hi = min(cum_pts), max(cum_pts)
    pad = max(1.2, (hi - lo) * 0.16)
    fig.update_xaxes(title_text="Логарифм шансов (log-odds · SHAP аддитивен в этой шкале)", range=[lo - pad, hi + pad],
                     showgrid=True, gridcolor=GRID, showline=True, linecolor=AXIS)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    h = max(360, 120 + 28 * len(labels))
    return base(fig, h, legend=True, margin=dict(l=8, r=36, t=42, b=28))


def fig_card_balance(slc: dict) -> go.Figure:
    """Дивергентная полоса баланса: сумма «за инцидент» (+) vs сумма «за FP» (−) для всех полей и Top-N."""
    cats = [f"Top-{slc['top_n']} полей", "Все 73 поля"]
    pos_v = [slc["sum_top_pos"], slc["sum_all_pos"]]
    neg_v = [slc["sum_top_neg"], slc["sum_all_neg"]]
    net_v = [slc["sum_top_net"], slc["sum_all_net"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=neg_v, orientation="h", name="сумма к FP (−)",
        marker=dict(color=BLUE, line=dict(width=0)),
        text=[dd.fmt_signed(v, 2) for v in neg_v], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=BLUE),
        hovertemplate="%{y}<br>Сумма вкладов к FP: %{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=cats, x=pos_v, orientation="h", name="сумма к инциденту (+)",
        marker=dict(color=ORANGE, line=dict(width=0)),
        text=[dd.fmt_signed(v, 2) for v in pos_v], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=ORANGE),
        hovertemplate="%{y}<br>Сумма вкладов к инциденту: +%{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=cats, x=net_v, mode="markers+text", name="чистый итог ΣSHAP",
        marker=dict(size=11, symbol="diamond", color=INK, line=dict(color=SURFACE, width=1.5)),
        text=[f"Σ={dd.fmt_signed(v, 2)}" for v in net_v], textposition="top center",
        textfont=dict(size=11, color=INK),
        hovertemplate="%{y}<br>Чистая сумма вкладов: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.5))
    lim = max(abs(min(neg_v)), max(pos_v), 1.0) * 1.28
    fig.update_xaxes(range=[-lim, lim], title_text="Суммарный вклад (log-odds)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=INK_2))
    return base(fig, 210, legend=True, barmode="overlay", margin=dict(l=8, r=24, t=36, b=8))


# ================================================================ 2. ПОХОЖИЕ КАРТОЧКИ
def fig_sim_hexmap(
    store: dd.Store,
    eid: int,
    space_label: str,
    color_mode: str,
    tau: float,
    show_neighbors: bool = True,
) -> go.Figure:
    """Двухслойная карта: Hexbin фон (~41.7k карточек) + маркеры 8 карточек team=5 + 20-NN."""
    if "t-SNE" in space_label:
        hx, xcol, ycol, sp_key = store.hex_tsne, "x_tsne", "y_tsne", "shap"
        xlab, ylab = "t-SNE 1 (пространство SHAP)", "t-SNE 2 (пространство SHAP)"
    elif "PCA" in space_label or "SVD" in space_label:
        hx, xcol, ycol, sp_key = store.hex_pca, "x_pca", "y_pca", "pca"
        ax = store.meta.get("map_axes", {}).get("pca", {})
        xlab = f"SVD-1 признаков ({ax.get('PC1', {}).get('var_ratio', 0):.0%} дисп.)"
        ylab = f"SVD-2 признаков ({ax.get('PC2', {}).get('var_ratio', 0):.0%} дисп.)"
    else:
        hx, xcol, ycol, sp_key = store.hex_shap, "x_shap", "y_shap", "shap"
        ax = store.meta.get("map_axes", {}).get("shap", {})
        t1 = ", ".join(ax.get("PC1", {}).get("top_features", [])[:2])
        t2 = ", ".join(ax.get("PC2", {}).get("top_features", [])[:2])
        xlab = f"PCA-1 SHAP ({ax.get('PC1', {}).get('var_ratio', 0):.0%}): {_short(t1, 44)}"
        ylab = f"PCA-2 SHAP ({ax.get('PC2', {}).get('var_ratio', 0):.0%}): {_short(t2, 44)}"

    if hx.empty:
        return empty("Проекция t-SNE не рассчитана (перезапустите prepare.py без --skip-tsne)")

    fig = go.Figure()

    # Слой 1: гексагональный фон
    c_cfg = {
        "Доля инцидентов (факт)": ("inc_rate", [[0, BLUE_LIGHT], [0.5, GRAY_LIGHT], [1, ORANGE]], [0, 1], "Доля<br>инцидентов"),
        "Решение модели (приоритет)": ("pred_rate", [[0, "#dce8f8"], [0.5, BLUE], [1, ORANGE]], [0, 1], "Доля<br>приоритета"),
        "Доля ошибок модели": ("err_rate", [[0, "#eef0f4"], [0.4, ORANGE_LIGHT], [1, MISS]], [0, 0.5], "Доля<br>ошибок"),
        "Кластеры (MiniBatchKMeans)": ("dominant_cluster", "Portland", [0, 11], "Кластер"),
    }[color_mode]
    c_col, cscale, crange, ctitle = c_cfg

    sz = 7 + 11 * (np.log1p(hx["n"].to_numpy()) / max(np.log1p(hx["n"].max()), 1.0))
    fig.add_trace(go.Scatter(
        x=hx["x"], y=hx["y"], mode="markers", showlegend=False,
        marker=dict(
            symbol="hexagon", size=sz, opacity=0.78,
            color=hx[c_col].fillna(0), colorscale=cscale, cmin=crange[0], cmax=crange[1],
            colorbar=dict(title=ctitle, thickness=12, len=0.72, tickformat=".0%" if crange[1] <= 1 else None),
            line=dict(width=0.4, color="rgba(255,255,255,0.6)"),
        ),
        customdata=np.c_[hx["n"], hx["inc_rate"].fillna(-1), hx["pred_rate"], hx["err_rate"].fillna(-1),
                         hx["dominant_cluster"], hx["top_rule"]],
        hovertemplate=(
            "Ячейка Hexbin: <b>%{customdata[0]:,}</b> карточек<br>"
            "Доля инцидентов: %{customdata[1]:.1%} · В приоритете: %{customdata[2]:.1%}<br>"
            "Ошибки: %{customdata[3]:.1%} · Кластер #%{customdata[4]}<br>"
            "Частое правило: %{customdata[5]}<extra></extra>"
        ),
    ))

    # Слой 2: 20 ближайших соседей (только для PCA/SVD, где координаты совпадают)
    foc = store.focus_team5.merge(store.team5_cards[["EntityId", "rank_in_team", "y_true", "proba_calc", "rule"]], on="EntityId")
    sel_row = foc[foc["EntityId"] == eid].iloc[0]
    if show_neighbors and "t-SNE" not in space_label:
        nbr = store.neighbors_team5[(store.neighbors_team5["target_id"] == eid) & (store.neighbors_team5["space"] == sp_key)]
        if len(nbr):
            lx, ly = [], []
            for nr in nbr.itertuples(index=False):
                lx += [float(sel_row[xcol]), float(nr.x), None]
                ly += [float(sel_row[ycol]), float(nr.y), None]
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=_rgba(INK, 0.25), width=1),
                                     hoverinfo="skip", name="связи к 20-NN"))
            fig.add_trace(go.Scatter(
                x=nbr["x"], y=nbr["y"], mode="markers", name="20 ближайших соседей",
                marker=dict(size=7, color=np.where(nbr["y_true"] == 1, ORANGE, BLUE),
                            line=dict(color=INK, width=1)),
                customdata=np.c_[nbr["neighbor_id"], nbr["cosine_sim"], nbr["outcome_at_tau"], nbr["rule"]],
                hovertemplate="Сосед #%{customdata[0]} (сходство %{customdata[1]:.3f})<br>Исход: %{customdata[2]} · %{customdata[3]}<extra></extra>",
            ))

    # Слой 3: 8 карточек команды 5
    style_map = {
        "TP": (ORANGE, "circle"),
        "FN": (MISS, "diamond"),
        "FP": (BLUE, "triangle-up"),
        "TN": (GRAY, "square"),
    }
    foc["out_dyn"] = [dd.outcome_of(int(y), float(p), tau) for y, p in zip(foc["y_true"], foc["proba_calc"])]
    for out_k, (col, sym) in style_map.items():
        sub = foc[foc["out_dyn"] == out_k]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[xcol], y=sub[ycol], mode="markers+text", name=f"team 5 · {out_k}",
            text=[f"#{int(r)}" for r in sub["rank_in_team"]], textposition="top center",
            textfont=dict(size=10, color=INK),
            marker=dict(size=13, symbol=sym, color=col, line=dict(color=SURFACE, width=2)),
            customdata=np.c_[sub["EntityId"], sub["rank_in_team"], sub["proba_calc"], sub["rule"]],
            hovertemplate=f"<b>Команда 5 · {out_k} (#%{{customdata[1]}})</b><br>EntityId %{{customdata[0]}} · p=%{{customdata[2]:.4%}}<br>%{{customdata[3]}}<extra></extra>",
        ))

    # Выделяем текущую карточку контрастным кольцом
    fig.add_trace(go.Scatter(
        x=[sel_row[xcol]], y=[sel_row[ycol]], mode="markers", name="выбранная карточка",
        marker=dict(size=22, symbol="circle-open", color=INK, line=dict(color=INK, width=2.5)),
        hoverinfo="skip",
    ))

    fig.update_xaxes(title_text=xlab, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text=ylab, showgrid=True, gridcolor=GRID)
    return base(fig, 470, legend=True, margin=dict(l=8, r=16, t=42, b=12))


def fig_group_violin(grp: dict, eid: int) -> go.Figure:
    """Violin + точки распределения SHAP в группе похожих карточек с выделением текущей карточки."""
    labels = grp["labels"][::-1]
    mat = grp["group_mat"][:, ::-1]
    cvals = grp["card_vals"][::-1]
    if mat.size == 0:
        return empty()

    fig = go.Figure()
    for i, lbl in enumerate(labels):
        vals = mat[:, i]
        fig.add_trace(go.Violin(
            x=vals, y=[_short(lbl, 34)] * len(vals), orientation="h",
            side="positive", width=0.85, points="all", jitter=0.18, pointpos=-0.22,
            fillcolor=_rgba(BLUE, 0.22), line=dict(color=BLUE, width=1.2),
            marker=dict(size=4, color=_rgba(INK_2, 0.45)),
            box_visible=True, meanline_visible=False,
            name="группа похожих", showlegend=(i == 0),
            hovertemplate=f"{lbl}<br>SHAP в группе: %{{x:+.3f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=cvals, y=[_short(l, 34) for l in labels], mode="markers", name=f"карточка {eid}",
        marker=dict(size=12, symbol="star", color=MISS, line=dict(color=SURFACE, width=1.2)),
        hovertemplate="%{y}<br>Вклад выбранной карточки: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.2, dash="dash"))
    fig.update_xaxes(title_text="Вклад SHAP (log-odds)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 440, legend=True, margin=dict(l=8, r=16, t=38, b=8))


# ================================================================ 3. ВЕСЬ ПОТОК (ГЛОБАЛЬНЫЙ XAI)
def fig_beeswarm(store: dd.Store, sample: str = "Эксплуатация", top_k: int = 15) -> go.Figure:
    """SHAP Summary Plot (Beeswarm) по топ-15 полям (WebGL Scattergl для плавности)."""
    sub = store.beeswarm[store.beeswarm["part"] == sample]
    if sub.empty:
        return empty()

    imp = store.global_imp[store.global_imp["sample"] == sample].sort_values("rank").head(top_k)
    cols = imp["source_column"].tolist()[::-1]
    labels = [_short(REGISTRY[c].label_ru, 34) for c in cols]

    rng = np.random.default_rng(42)
    xs, ys, cs, hovers = [], [], [], []
    for yi, col in enumerate(cols):
        sv = sub[col].to_numpy(dtype=float)
        vc = sub[f"val__{col}"].to_numpy(dtype=float)
        # Плотность по x задаёт ширину «роя»
        hist, edges = np.histogram(sv, bins=30)
        bi = np.clip(np.searchsorted(edges, sv, side="right") - 1, 0, len(hist) - 1)
        dens = hist[bi] / max(hist.max(), 1)
        jitter = rng.uniform(-0.34, 0.34, size=len(sv)) * np.sqrt(dens)
        xs.append(sv)
        ys.append(yi + jitter)
        cs.append(vc)
        hovers.append(np.full(len(sv), REGISTRY[col].label_ru))

    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers",
        marker=dict(
            size=4.5, opacity=0.72, color=np.concatenate(cs),
            colorscale=[[0, BLUE], [0.5, GRAY_LIGHT], [1, ORANGE]], cmin=0, cmax=1,
            colorbar=dict(title="Значение<br>признака", tickvals=[0.05, 0.95], ticktext=["низкое / пусто", "высокое / есть"],
                          thickness=12, len=0.7),
        ),
        text=np.concatenate(hovers),
        hovertemplate="%{text}<br>SHAP: %{x:+.3f} log-odds<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.2, dash="dash"))
    fig.update_yaxes(tickvals=list(range(len(labels))), ticktext=labels, showgrid=False, tickfont=dict(size=11, color=INK_2))
    fig.update_xaxes(title_text="Вклад признака в скор модели (SHAP, log-odds)", showgrid=True, gridcolor=GRID)
    return base(fig, 480, margin=dict(l=8, r=16, t=16, b=10))


def fig_global_dot(store: dd.Store, top_k: int = 15) -> go.Figure:
    """2 панели: (1) Dot Plot доли mean|SHAP| в Приёмке / Эксплуатации / Практике 320; (2) Ablation Δproba."""
    imp = store.global_imp
    order = imp[imp["sample"] == "Эксплуатация"].sort_values("rank").head(top_k)["source_column"].tolist()[::-1]
    labels = [_short(REGISTRY[c].label_ru, 34) for c in order]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.66, 0.34], horizontal_spacing=0.06,
                        subplot_titles=["Доля в сумме mean|SHAP| по выборкам", "Ablation: Δ вероятности при выключении"])

    # Направляющие линии Dot Plot
    for yi, lbl in enumerate(labels):
        fig.add_shape(type="line", x0=0, x1=0.001, y0=lbl, y1=lbl, line=dict(color=GRID, width=1), row=1, col=1)

    for samp, col, sym in [("Приёмка", VAL_COLOR, "circle"), ("Эксплуатация", OPER_COLOR, "circle"), ("Практика 320", ORANGE, "diamond")]:
        s = imp[imp["sample"] == samp].set_index("source_column").reindex(order)
        fig.add_trace(go.Scatter(
            x=s["share_abs_shap"], y=labels, mode="markers", name=samp,
            marker=dict(size=9, symbol=sym, color=col, line=dict(color=SURFACE, width=1.5)),
            customdata=s["mean_abs_shap"],
            hovertemplate=f"{samp} · %{{y}}<br>Доля |SHAP|: %{{x:.1%}} (mean|SHAP|=%{{customdata:.3f}})<extra></extra>",
        ), row=1, col=1)

    abl = imp[imp["sample"] == "Эксплуатация"].set_index("source_column").reindex(order)
    fig.add_trace(go.Bar(
        x=abl["ablation_delta_proba"], y=labels, orientation="h", showlegend=False,
        marker=dict(color=GRAY, line=dict(width=0)),
        text=[dd.fmt_pct(v, 2) for v in abl["ablation_delta_proba"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK_2),
        hovertemplate="%{y}<br>Ablation |Δp|: %{x:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Доля вклада", tickformat=".0%", showgrid=True, gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(title_text="Среднее |Δ вероятности|", tickformat=".1%", showgrid=True, gridcolor=GRID,
                     range=[0, float(abl["ablation_delta_proba"].max()) * 1.28 + 1e-4], row=1, col=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 480, legend=True, margin=dict(l=8, r=24, t=42, b=10))


def fig_ale_pdp_ice(store: dd.Store, source_col: str, period: str, eid: int, show_ice: bool = True) -> go.Figure:
    """2 вертикальные панели: (1) ALE (основной) + PDP + ICE; (2) плотность наблюдений (бины)."""
    cn = store.curves_num[(store.curves_num["source_column"] == source_col) & (store.curves_num["period"] == period)].sort_values("grid_idx")
    if cn.empty:
        return empty(f"Для «{source_col}» недостаточно данных для ALE/PDP")

    meta = REGISTRY[source_col]
    warn = bool(cn["collinear_warn"].iloc[0])
    partner = str(cn["corr_partner"].iloc[0])
    max_r = float(cn["max_spearman"].iloc[0])

    # Смещаем ALE на средний уровень PDP для визуального сопоставления масштаба, оставляя приращение в hover
    pdp_base = float(cn["pdp"].mean())
    ale_shifted = cn["ale"].to_numpy() + pdp_base

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.05)

    # ICE (фон + выбранная карточка команды 5)
    if show_ice and period == "Эксплуатация":
        ice = store.curves_ice[store.curves_ice["source_column"] == source_col]
        bg = ice[ice["curve_kind"] == "bg"]
        if len(bg):
            ix, iy = [], []
            for _, g in bg.groupby("curve_id", sort=False):
                g = g.sort_values("grid_idx")
                ix += g["raw_value"].tolist() + [None]
                iy += g["proba"].tolist() + [None]
            fig.add_trace(go.Scatter(x=ix, y=iy, mode="lines", name="ICE (40 карточек)",
                                     line=dict(color=_rgba(GRAY, 0.25), width=1), hoverinfo="skip"), row=1, col=1)
        t5 = ice[(ice["curve_kind"] == "team5") & (ice["curve_id"] == str(eid))].sort_values("grid_idx")
        if len(t5):
            fig.add_trace(go.Scatter(
                x=t5["raw_value"], y=t5["proba"], mode="lines+markers", name=f"ICE карточки {eid}",
                line=dict(color=MISS, width=2, dash="dot"), marker=dict(size=5, color=MISS),
                hovertemplate=f"Карточка {eid}: при значении %{{x:.4g}} → p=%{{y:.2%}}<extra></extra>",
            ), row=1, col=1)
            cv = float(t5["card_raw_value"].iloc[0])
            if np.isfinite(cv):
                fig.add_vline(x=cv, line=dict(color=MISS, width=1.2, dash="dot"),
                              annotation_text=f"факт {eid}: {dd.fmt_num(cv, 2)}", annotation_position="top right",
                              annotation_font=dict(size=10, color=MISS), row=1, col=1)

    # PDP (прерывистая, если есть мультиколлинеарность)
    fig.add_trace(go.Scatter(
        x=cn["raw_value"], y=cn["pdp"], mode="lines+markers",
        name="PDP (искажён при |ρ|≥0,7)" if warn else "PDP (среднее)",
        line=dict(color=GRAY if warn else BLUE, width=1.8, dash="dash" if warn else "solid"),
        marker=dict(size=5),
        hovertemplate="PDP при %{x:.4g}: средняя оценка %{y:.2%}<extra></extra>",
    ), row=1, col=1)

    # ALE — главный метод (жирная оранжевая линия)
    fig.add_trace(go.Scatter(
        x=cn["raw_value"], y=ale_shifted, mode="lines+markers", name="ALE (основной, устойчив к коллинеарности)",
        line=dict(color=ORANGE, width=3), marker=dict(size=7, color=ORANGE, line=dict(color=SURFACE, width=1.5)),
        customdata=cn["ale"],
        hovertemplate="ALE при %{x:.4g}: эффект %{customdata:+.2%} (уровень %{y:.2%})<extra></extra>",
    ), row=1, col=1)

    if warn:
        fig.add_annotation(
            x=0.01, y=0.98, xref="x domain", yref="y domain", xanchor="left", yanchor="top", showarrow=False,
            text=f"⚠ Сильная корреляция |ρ|={dd.fmt_num(max_r, 2)} с «{_short(partner, 24)}»: ориентируйтесь на ALE, не на PDP",
            font=dict(size=11, color=MISS), bgcolor="#fef4f3", bordercolor=MISS, borderwidth=1, borderpad=4,
        )

    # Нижняя панель — плотность данных по сетке
    fig.add_trace(go.Bar(
        x=cn["raw_value"], y=cn["bin_n"], showlegend=False, marker=dict(color=GRAY_LIGHT, line=dict(width=0)),
        hovertemplate="Значение %{x:.4g}: %{y:,} карточек<extra></extra>",
    ), row=2, col=1)

    fig.update_yaxes(title_text="Оценка вероятности", tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text="Карточек", tickformat=",d", nticks=3, row=2, col=1)
    fig.update_xaxes(title_text=f"{meta.label_ru} ({meta.source_column})", row=2, col=1)
    return base(fig, 440, legend=True, margin=dict(l=8, r=16, t=38, b=10))


def fig_cat_gpdp(store: dd.Store, cat_col: str) -> go.Figure:
    """group-PDP (what-if) для 12 категориальных полей: точечная диаграмма Приёмка vs Эксплуатация."""
    sub = store.curves_cat[store.curves_cat["source_column"] == cat_col]
    if sub.empty:
        return empty()

    oper = sub[sub["period"] == "Эксплуатация"].sort_values("mean_proba", ascending=True)
    levels = oper["level"].tolist()
    val = sub[sub["period"] == "Приёмка"].set_index("level").reindex(levels).reset_index()

    fig = go.Figure()
    base_o = float(oper["base_proba"].iloc[0])
    fig.add_vline(x=base_o, line=dict(color=MUTED, width=1.4, dash="dash"),
                  annotation_text=f"среднее по потоку {dd.fmt_pct(base_o, 1)}",
                  annotation_position="top right", annotation_font=dict(size=11, color=INK_2))

    for lv, vo, vv in zip(levels, oper["mean_proba"], val["mean_proba"]):
        if np.isfinite(vv):
            fig.add_shape(type="line", x0=vv, x1=vo, y0=_short(lv, 38), y1=_short(lv, 38),
                          line=dict(color=GRAY_LIGHT, width=2.5), layer="below")

    y_lbls = [_short(lv, 38) for lv in levels]
    fig.add_trace(go.Scatter(
        x=val["mean_proba"], y=y_lbls, mode="markers", name="приёмка",
        marker=dict(size=10, color=VAL_COLOR, line=dict(color=SURFACE, width=1.5)),
        hovertemplate="%{y}<br>приёмка (group-PDP): %{x:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=oper["mean_proba"], y=y_lbls, mode="markers", name="эксплуатация",
        marker=dict(size=11, color=OPER_COLOR, line=dict(color=SURFACE, width=1.5)),
        customdata=np.c_[oper["delta_proba"], oper["n_observed"]],
        hovertemplate="%{y}<br>эксплуатация (group-PDP): %{x:.2%} (Δ %{customdata[0]:+.2%}, n=%{customdata[1]:,})<extra></extra>",
    ))
    fig.update_xaxes(title_text="Средняя оценка при подстановке уровня (group-PDP)", tickformat=".1%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, max(320, 80 + 28 * len(levels)), legend=True, margin=dict(l=8, r=20, t=38, b=10))


# ================================================================ 4. ДРЕЙФ ПРИЁМКА -> ЭКСПЛУАТАЦИЯ
def fig_drift_dumbbell(store: dd.Store, sort_by: str = "По сдвигу |SHAP|", top_k: int = 15) -> go.Figure:
    """Гантели сдвига mean|SHAP| между приёмкой и эксплуатацией + метка PSI."""
    ds = store.drift_summary.copy()
    if sort_by == "По PSI":
        ds = ds.sort_values("psi", ascending=False)
    elif sort_by == "По росту пропусков":
        ds = ds.sort_values("delta_null", ascending=False)
    else:
        ds["abs_d"] = ds["delta_abs_shap"].abs()
        ds = ds.sort_values("abs_d", ascending=False)

    top = ds.head(top_k).iloc[::-1]
    names = [_short(r.label_ru, 34) for r in top.itertuples(index=False)]

    fig = go.Figure()
    for nm, a, b in zip(names, top["mean_abs_shap_val"], top["mean_abs_shap_oper"]):
        fig.add_shape(type="line", x0=a, x1=b, y0=nm, y1=nm,
                      line=dict(color=ORANGE_LIGHT if b > a else BLUE_LIGHT, width=3), layer="below")

    fig.add_trace(go.Scatter(
        x=top["mean_abs_shap_val"], y=names, mode="markers", name="приёмка",
        marker=dict(size=11, color=VAL_COLOR, line=dict(color=SURFACE, width=1.8)),
        hovertemplate="%{y}<br>приёмка mean|SHAP|: %{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=top["mean_abs_shap_oper"], y=names, mode="markers+text", name="эксплуатация",
        marker=dict(size=11, color=OPER_COLOR, line=dict(color=SURFACE, width=1.8)),
        text=[f"PSI {dd.fmt_num(p, 2)}" for p in top["psi"]],
        textposition=["middle right" if b >= a else "middle left" for a, b in zip(top["mean_abs_shap_val"], top["mean_abs_shap_oper"])],
        textfont=dict(size=10, color=[MISS if p >= 0.25 else MUTED for p in top["psi"]]),
        cliponaxis=False,
        customdata=np.c_[top["psi"], top["null_val"], top["null_oper"]],
        hovertemplate="%{y}<br>эксплуатация mean|SHAP|: %{x:.3f}<br>PSI: %{customdata[0]:.2f} · пропуски: %{customdata[1]:.0%} → %{customdata[2]:.0%}<extra></extra>",
    ))
    xmax = float(max(top["mean_abs_shap_val"].max(), top["mean_abs_shap_oper"].max())) * 1.22
    fig.update_xaxes(title_text="Средний абсолютный вклад mean|SHAP| (log-odds)", range=[0, xmax], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 480, legend=True, margin=dict(l=8, r=60, t=38, b=10))


def fig_drift_dist(store: dd.Store, source_col: str) -> go.Figure:
    """Сравнение распределения выбранного поля в приёмке и эксплуатации."""
    bins = store.drift_bins[store.drift_bins["source_column"] == source_col].sort_values("bin_idx")
    if bins.empty:
        return empty(f"Нет бинов распределения для «{source_col}»")

    is_num = REGISTRY[source_col].last_step == "QuantileTransformer"
    fig = go.Figure()
    if is_num:
        fig.add_trace(go.Bar(
            x=bins["bin_label"], y=bins["val_share"], name="приёмка",
            marker=dict(color=VAL_COLOR, opacity=0.8),
            customdata=bins["val_n"],
            hovertemplate="Интервал %{x}<br>Приёмка: %{y:.1%} (%{customdata:,})<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=bins["bin_label"], y=bins["oper_share"], name="эксплуатация",
            marker=dict(color=OPER_COLOR, opacity=0.85),
            customdata=bins["oper_n"],
            hovertemplate="Интервал %{x}<br>Эксплуатация: %{y:.1%} (%{customdata:,})<extra></extra>",
        ))
        fig.update_yaxes(title_text="Доля карточек", tickformat=".0%")
        fig.update_xaxes(title_text=REGISTRY[source_col].label_ru, tickangle=-25)
        return base(fig, 400, legend=True, barmode="group", margin=dict(l=8, r=16, t=38, b=20))

    # Категории / текст — горизонтальные гантели долей (чище столбцов для длинных имён)
    b = bins.iloc[::-1]
    lbls = [_short(s, 36) for s in b["bin_label"]]
    for nm, a, c in zip(lbls, b["val_share"], b["oper_share"]):
        fig.add_shape(type="line", x0=a, x1=c, y0=nm, y1=nm, line=dict(color=GRAY_LIGHT, width=3), layer="below")
    fig.add_trace(go.Scatter(
        x=b["val_share"], y=lbls, mode="markers", name="приёмка",
        marker=dict(size=10, color=VAL_COLOR, line=dict(color=SURFACE, width=1.5)),
        hovertemplate="%{y}<br>приёмка: %{x:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=b["oper_share"], y=lbls, mode="markers+text", name="эксплуатация",
        marker=dict(size=11, color=OPER_COLOR, line=dict(color=SURFACE, width=1.5)),
        text=[dd.fmt_pct(v, 0) for v in b["oper_share"]],
        textposition=["middle right" if c >= a else "middle left" for a, c in zip(b["val_share"], b["oper_share"])],
        textfont=dict(size=10, color=INK_2), cliponaxis=False,
        hovertemplate="%{y}<br>эксплуатация: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Доля карточек в потоке", tickformat=".0%",
                     range=[0, min(1.08, float(max(b["val_share"].max(), b["oper_share"].max())) * 1.18 + 0.03)],
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, max(340, 70 + 30 * len(b)), legend=True, margin=dict(l=8, r=36, t=38, b=10))


# ================================================================ 5. ПРОВЕРКА ОБЪЯСНЕНИЙ (КЕЙСЫ)
def fig_verify_wilson(store: dd.Store, eid: int, period: str = "Эксплуатация") -> go.Figure:
    """Две панели: (1) SHAP-вклад топ-12 признаков кейса; (2) фактическая доля инцидентов + 95 % ДИ Уилсона."""
    df = store.verification[(store.verification["EntityId"] == eid) & (store.verification["period"] == period)].sort_values("rank_abs")
    if df.empty:
        return empty("Нет статистики верификации для карточки")

    d = df.iloc[::-1]
    labels = [f"{_short(r.label_ru, 28)} ({_short(r.slice_desc, 16)})" for r in d.itertuples(index=False)]
    base_rate = float(d["base_inc_rate"].iloc[0])

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.44, 0.56], horizontal_spacing=0.05,
                        subplot_titles=["Вклад SHAP модели (log-odds)", f"Фактическая доля инцидентов и 95 % ДИ ({period})"])

    # Панель 1: SHAP
    fig.add_trace(go.Bar(
        x=d["shap_val"], y=labels, orientation="h", showlegend=False,
        marker=dict(color=[ORANGE if v >= 0 else BLUE for v in d["shap_val"]], line=dict(width=0)),
        text=[dd.fmt_signed(v, 2) for v in d["shap_val"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK_2),
        hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.2), row=1, col=1)

    # Панель 2: Доля инцидентов с 95 % ДИ Уилсона
    st_color = {
        "Подтверждено": ORANGE,
        "Противоречие": MISS,
        "Нейтрально": GRAY,
        "Мало данных": GRAY_LIGHT,
    }
    colors = [st_color.get(s, GRAY) for s in d["status"]]
    fig.add_vline(x=base_rate, line=dict(color=MUTED, width=1.4, dash="dash"),
                  annotation_text=f"среднее {dd.fmt_pct(base_rate, 1)}", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=INK_2), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=d["inc_rate"], y=labels, mode="markers", showlegend=False,
        error_x=dict(type="data", symmetric=False,
                     array=d["ci_hi"] - d["inc_rate"], arrayminus=d["inc_rate"] - d["ci_lo"],
                     color=AXIS, thickness=1.5, width=0),
        marker=dict(size=11, color=colors, line=dict(color=SURFACE, width=1.8)),
        customdata=np.c_[d["k"], d["n"], d["ci_lo"], d["ci_hi"], d["status"]],
        hovertemplate="%{y}<br>Инцидентов: %{customdata[0]} из %{customdata[1]} (%{x:.1%}; ДИ %{customdata[2]:.1%}–%{customdata[3]:.1%})<br><b>%{customdata[4]}</b><extra></extra>",
    ), row=1, col=2)

    for st_name, col in [("Подтверждено", ORANGE), ("Противоречие", MISS), ("Нейтрально", GRAY), ("Мало данных", GRAY_LIGHT)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=st_name, marker=dict(size=9, color=col)), row=1, col=2)

    smax = max(float(d["shap_val"].abs().max()), 0.5) * 1.28
    fig.update_xaxes(range=[-smax, smax], title_text="SHAP (log-odds)", showgrid=True, gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(range=[0, 1.04], title_text="Фактическая доля инцидентов", tickformat=".0%", showgrid=True, gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 480, legend=True, margin=dict(l=8, r=20, t=42, b=10))