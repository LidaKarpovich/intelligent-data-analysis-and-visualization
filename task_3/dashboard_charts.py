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
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde

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
                "в компактную сетку занятых шестиугольных ячеек без перегрузки Plotly и наслоения точек, а поверх "
                "показывает 8 карточек команды 5 и истинных 20-NN из многомерного пространства.",
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
    "corr_heatmap": "Кластеризованная тепловая карта корреляций Спирмена (только поля со связью |ρ| ≥ 0,5) "
                    "группирует мультиколлинеарные признаки в видимые блоки, сразу объясняя, для каких полей PDP искажается.",
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
    "corr_heatmap": "матрица всех 43×43 полей без сортировки: превращается в нечитаемую сетку из ~1 800 ячеек с преобладанием нулей",
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


def _fmt_compact(v: float) -> str:
    """Компактное форматирование больших чисел для подписей осей (41200000 -> 41,2M)."""
    av = abs(v)
    if av >= 1e6:
        return f"{v / 1e6:.1f}M".replace(".", ",")
    if av >= 1e3:
        return f"{v / 1e3:.1f}k".replace(".", ",")
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.3g}".replace(".", ",")


def _pca_axis_title(axis_name: str, info: dict, max_feats: int = 2) -> str:
    """Двухстрочная подпись оси PCA без обрезки через многоточие."""
    vr = info.get("var_ratio", 0.0)
    feats = info.get("top_features", [])[:max_feats]
    if not feats:
        return f"{axis_name} SHAP ({vr:.0%} дисп.)"
    cleaned = []
    for f in feats:
        parts = f.rsplit(" (", 1)
        name = parts[0].split(" (", 1)[0].strip()
        coef = f" ({parts[1]}" if len(parts) == 2 else ""
        cleaned.append(f"{name}{coef}")
    # Если для вертикальной оси два длинных названия не помещаются, оставляем главный фактор
    if max_feats == 1 or len(", ".join(cleaned)) > 58:
        line2 = cleaned[0]
    else:
        line2 = ", ".join(cleaned)
    return f"{axis_name} SHAP ({vr:.0%} дисп.)<br>{line2}"


# ---------------------------------------------------------------- хелперы для плотного Hexbin
def _infer_hex_radii(hx: pd.DataFrame) -> tuple[float, float]:
    """Точно восстанавливает полуоси (rx, ry) плоских шестиугольников из координат центров."""
    xs = np.sort(np.unique(np.round(hx["x"].to_numpy(dtype=float), 7)))
    dxs = np.diff(xs)
    dxs = dxs[dxs > 1e-6]
    rx = (float(np.min(dxs)) / 1.5) if len(dxs) else 0.1

    col_diffs = []
    for _, g in hx.assign(_xr=np.round(hx["x"], 6)).groupby("_xr"):
        if len(g) >= 2:
            dys = np.diff(np.sort(g["y"].to_numpy(dtype=float)))
            dys = dys[dys > 1e-6]
            if len(dys):
                col_diffs.append(float(np.min(dys)))
    if col_diffs:
        ry = float(np.min(col_diffs)) / np.sqrt(3.0)
    else:
        ys = np.sort(np.unique(np.round(hx["y"].to_numpy(dtype=float), 7)))
        dys = np.diff(ys)
        dys = dys[dys > 1e-6]
        ry = (float(np.min(dys)) * 2.0 / np.sqrt(3.0)) if len(dys) else rx
    return rx, ry


def _interp_color(t: float, scale: list[tuple[float, str]]) -> str:
    """Линейная интерполяция hex-цвета по шкале [(pos, '#rrggbb'), ...]."""
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(scale) - 1):
        p0, c0 = scale[i]
        p1, c1 = scale[i + 1]
        if t <= p1 or i == len(scale) - 2:
            w = 0.0 if p1 == p0 else (t - p0) / (p1 - p0)
            rgb0 = [int(c0.lstrip("#")[j:j + 2], 16) for j in (0, 2, 4)]
            rgb1 = [int(c1.lstrip("#")[j:j + 2], 16) for j in (0, 2, 4)]
            rgb = [int(round(a + (b - a) * w)) for a, b in zip(rgb0, rgb1)]
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    return scale[-1][1]


def _hex_polys(xs: np.ndarray, ys: np.ndarray, rx: float, ry: float) -> tuple[list, list]:
    """Строит замкнутые полигоны стыкующихся шестиугольников (разделитель None) для одного трейса."""
    ang = np.linspace(0.0, 2.0 * np.pi, 7)
    dx = rx * np.cos(ang)
    dy = ry * np.sin(ang)
    px, py = [], []
    for xc, yc in zip(xs, ys):
        px.extend((xc + dx).tolist())
        px.append(None)
        py.extend((yc + dy).tolist())
        py.append(None)
    return px, py


# Качественная (дискретная) палитра Okabe-Ito для топ-5 кластеров — без радуги и безопасна при дальтонизме
QUAL_5 = ["#2a78d6", "#eb6834", "#009e73", "#cc79a7", "#e69f00"]


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
        hovers.append(
            f"<b>{r.display_name}</b><br>Значение: {_short(str(r.hover_val), 80)}<br>"
            f"Вклад SHAP: {dd.fmt_signed(dv, 3)} log-odds<br>"
            f"После шага: {cur + dv:.2f} (p={dd.fmt_proba(dd.sigmoid(cur + dv))})"
        )
        cur += dv
        cum_pts.append(cur)

    ov = float(slc["other_val"])
    labels.append(slc["other_label"])
    starts.append(cur)
    widths.append(ov)
    colors.append(GRAY if abs(ov) < 0.05 else (ORANGE_LIGHT if ov > 0 else BLUE_LIGHT))
    texts.append(dd.fmt_signed(ov, 2))
    hovers.append(f"<b>{slc['other_label']}</b><br>Суммарный вклад: {dd.fmt_signed(ov, 3)} log-odds")
    cur += ov
    cum_pts.append(cur)

    labels, starts, widths = labels[::-1], starts[::-1], widths[::-1]
    colors, texts, hovers = colors[::-1], texts[::-1], hovers[::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=widths, base=starts, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=texts, textposition="outside", cliponaxis=False,
        textfont=dict(size=10 if len(labels) > 15 else 11, color=INK_2),
        hovertext=hovers, hoverinfo="text", showlegend=False,
    ))

    for i in range(len(labels) - 1):
        end_x = starts[i + 1] + widths[i + 1]
        fig.add_shape(type="line", x0=end_x, x1=end_x, y0=i + 0.35, y1=i + 0.65,
                      line=dict(color=GRAY_LIGHT, width=1))

    fig.add_vline(x=base_v, line=dict(color=GRAY, width=1.5, dash="dot"),
                  annotation_text=f"baseline {dd.fmt_num(base_v, 2)} ({dd.fmt_proba(dd.sigmoid(base_v))})",
                  annotation_position="bottom left", annotation_font=dict(size=10, color=MUTED))
    fig.add_vline(x=logit_tau, line=dict(color=MISS, width=1.8, dash="dash"),
                  annotation_text=f"порог τ={dd.fmt_proba(tau)} ({dd.fmt_num(logit_tau, 2)})",
                  annotation_position="top right", annotation_font=dict(size=11, color=MISS))
    fig.add_vline(x=raw_v, line=dict(color=INK, width=1.8, dash="solid"),
                  annotation_text=f"итог {dd.fmt_num(raw_v, 2)} ({dd.fmt_proba(slc['proba'])})",
                  annotation_position="top left", annotation_font=dict(size=11, color=INK))

    for name, col in [("в сторону инцидента (+)", ORANGE), ("в сторону FP (−)", BLUE)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=10, symbol="square", color=col)))

    lo, hi = min(cum_pts), max(cum_pts)
    pad = max(1.2, (hi - lo) * 0.16)
    fig.update_xaxes(title_text="Логарифм шансов (log-odds · SHAP аддитивен в этой шкале)",
                     range=[lo - pad, hi + pad], showgrid=True, gridcolor=GRID, showline=True, linecolor=AXIS)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10 if len(labels) > 15 else 11, color=INK_2))
    # Фиксированная высота 600px совпадает с контейнером 620px в dashboard.py — при N=5..20 ничего не обрезается
    return base(fig, 600, legend=True, legend_y=1.03, margin=dict(l=8, r=42, t=46, b=46))


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
    """Двухслойная карта: плотная сотовая сетка Hexbin (~41.7k карточек) + команда 5 + 20-NN."""
    if "t-SNE" in space_label:
        hex_key, xcol, ycol, sp_key = "tsne", "x_tsne", "y_tsne", "shap"
        xlab, ylab = "t-SNE 1 (пространство SHAP)", "t-SNE 2 (пространство SHAP)"
    elif "SVD" in space_label or "Признаки" in space_label:
        hex_key, xcol, ycol, sp_key = "pca", "x_pca", "y_pca", "pca"
        ax = store.meta.get("map_axes", {}).get("pca", {})
        xlab = f"SVD-1 признаков ({ax.get('PC1', {}).get('var_ratio', 0):.0%} дисп.)"
        ylab = f"SVD-2 признаков ({ax.get('PC2', {}).get('var_ratio', 0):.0%} дисп.)"
    else:
        hex_key, xcol, ycol, sp_key = "shap", "x_shap", "y_shap", "shap"
        ax = store.meta.get("map_axes", {}).get("shap", {})
        xlab = _pca_axis_title("PCA-1", ax.get("PC1", {}), max_feats=2)
        ylab = _pca_axis_title("PCA-2", ax.get("PC2", {}), max_feats=1)

    hx, is_live_tau = dd.hex_at_threshold(store, hex_key, tau)
    if hx.empty:
        return empty("Проекция t-SNE не рассчитана (перезапустите prepare.py без --skip-tsne)")

    foc = store.focus_team5.merge(
        store.team5_cards[["EntityId", "rank_in_team", "y_true", "proba_calc", "rule"]], on="EntityId"
    )
    sel_row = foc[foc["EntityId"] == eid].iloc[0]
    sel_cluster = int(sel_row["cluster_pca"] if sp_key == "pca" else sel_row["cluster_shap"])

    rx, ry = _infer_hex_radii(hx)
    fig = go.Figure()
    hx_x = hx["x"].to_numpy(dtype=float)
    hx_y = hx["y"].to_numpy(dtype=float)

    tau_note = ""
    if not is_live_tau and abs(tau - dd.DEFAULT_TAU) > 1e-6:
        tau_note = "<br>(τ=0,0193)"
    elif is_live_tau:
        tau_note = f"<br>(τ={dd.fmt_num(tau, 4)})"

    is_discrete_cluster = color_mode in ("Кластер выбранной карточки", "Топ-5 кластеров (дискретно)")
    if color_mode == "Кластер выбранной карточки":
        cl_arr = hx["dominant_cluster"].to_numpy(dtype=int)
        for mask, col, name in [
            (cl_arr != sel_cluster, "#e6e8ec", "остальные кластеры"),
            (cl_arr == sel_cluster, _rgba(BLUE, 0.72), f"кластер #{sel_cluster} (выбранной карточки)"),
        ]:
            if mask.any():
                px, py = _hex_polys(hx_x[mask], hx_y[mask], rx, ry)
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="lines", fill="toself", fillcolor=col,
                    line=dict(color="rgba(255,255,255,0.65)", width=0.5),
                    name=name, hoverinfo="skip",
                ))
        cb_vals, cb_scale, cb_range, cb_title, cb_fmt = None, None, None, None, None

    elif color_mode == "Топ-5 кластеров (дискретно)":
        cl_info = store.clusters[store.clusters["space"] == sp_key].sort_values("n", ascending=False)
        top5 = cl_info.head(5)["cluster"].astype(int).tolist()
        rule_map = dict(zip(cl_info["cluster"].astype(int), cl_info["top_rule"]))
        cl_arr = hx["dominant_cluster"].to_numpy(dtype=int)

        other_m = ~np.isin(cl_arr, top5)
        if other_m.any():
            px, py = _hex_polys(hx_x[other_m], hx_y[other_m], rx, ry)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", fill="toself", fillcolor="#e2e4e8",
                line=dict(color="rgba(255,255,255,0.65)", width=0.5),
                name="прочие кластеры", hoverinfo="skip",
            ))
        for idx_c, cid in enumerate(top5):
            m = cl_arr == cid
            if not m.any():
                continue
            px, py = _hex_polys(hx_x[m], hx_y[m], rx, ry)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", fill="toself", fillcolor=_rgba(QUAL_5[idx_c], 0.72),
                line=dict(color="rgba(255,255,255,0.65)", width=0.5),
                name=f"#{cid}: {_short(rule_map.get(cid, '—'), 20)}", hoverinfo="skip",
            ))
        cb_vals, cb_scale, cb_range, cb_title, cb_fmt = None, None, None, None, None

    else:
        cont_cfg = {
            "Плотность карточек (классический Hexbin)": (
                np.log10(np.maximum(hx["n"].to_numpy(dtype=float), 1.0)),
                [(0.0, "#eaf2fc"), (0.5, "#6aa4e8"), (1.0, "#14488a")],
                [0.0, float(np.log10(max(int(hx["n"].max()), 10)))],
                "log10(N)", ".1f",
            ),
            "Доля инцидентов (факт)": (
                hx["inc_rate"].fillna(0.0).to_numpy(dtype=float),
                [(0.0, "#cfe2f9"), (0.5, "#e6e5e0"), (1.0, ORANGE)],
                [0.0, 1.0], "Доля<br>инцидентов", ".0%",
            ),
            "Решение модели (приоритет)": (
                hx["pred_rate"].fillna(0.0).to_numpy(dtype=float),
                [(0.0, "#e3eefb"), (0.5, BLUE), (1.0, ORANGE)],
                [0.0, 1.0], f"Доля<br>приоритета{tau_note}", ".0%",
            ),
            "Доля ошибок модели": (
                hx["err_rate"].fillna(0.0).to_numpy(dtype=float),
                [(0.0, "#eef0f4"), (0.45, ORANGE_LIGHT), (1.0, MISS)],
                [0.0, 0.5], f"Доля<br>ошибок{tau_note}", ".0%",
            ),
        }[color_mode]
        cb_vals, cb_scale, cb_range, cb_title, cb_fmt = cont_cfg
        n_bins = 16
        t_norm = np.clip((cb_vals - cb_range[0]) / max(cb_range[1] - cb_range[0], 1e-9), 0.0, 1.0)
        b_ids = np.clip((t_norm * n_bins).astype(int), 0, n_bins - 1)
        for bi in range(n_bins):
            m = b_ids == bi
            if not m.any():
                continue
            col = _interp_color((bi + 0.5) / n_bins, cb_scale)
            px, py = _hex_polys(hx_x[m], hx_y[m], rx, ry)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", fill="toself", fillcolor=col,
                line=dict(color="rgba(255,255,255,0.55)", width=0.5),
                showlegend=False, hoverinfo="skip",
            ))

    marker_kw = dict(size=10, opacity=0.001)
    if not is_discrete_cluster and cb_vals is not None:
        marker_kw.update(
            color=cb_vals,
            colorscale=[[p, c] for p, c in cb_scale],
            cmin=cb_range[0], cmax=cb_range[1], showscale=True,
            colorbar=dict(title=cb_title, thickness=12, len=0.72, tickformat=cb_fmt),
        )
    fig.add_trace(go.Scatter(
        x=hx_x, y=hx_y, mode="markers", showlegend=False, marker=marker_kw,
        customdata=np.c_[
            hx["n"], hx["inc_rate"].fillna(0.0), hx["pred_rate"], hx["err_rate"].fillna(0.0),
            hx["dominant_cluster"], hx["top_rule"],
        ],
        hovertemplate=(
            "Ячейка Hexbin: <b>%{customdata[0]:,}</b> карточек<br>"
            "Доля инцидентов: %{customdata[1]:.1%} · В приоритете: %{customdata[2]:.1%}<br>"
            "Ошибки: %{customdata[3]:.1%} · Домин. кластер #%{customdata[4]}<br>"
            "Частое правило: %{customdata[5]}<extra></extra>"
        ),
    ))

    if show_neighbors and "t-SNE" not in space_label:
        nbr = store.neighbors_team5[(store.neighbors_team5["target_id"] == eid) & (store.neighbors_team5["space"] == sp_key)]
        if len(nbr):
            lx, ly = [], []
            for nr in nbr.itertuples(index=False):
                lx += [float(sel_row[xcol]), float(nr.x), None]
                ly += [float(sel_row[ycol]), float(nr.y), None]
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=_rgba(INK, 0.28), width=1),
                                     hoverinfo="skip", name="связи к 20-NN"))
            fig.add_trace(go.Scatter(
                x=nbr["x"], y=nbr["y"], mode="markers", name="20 ближайших соседей",
                marker=dict(size=7, color=np.where(nbr["y_true"] == 1, ORANGE, BLUE),
                            line=dict(color=INK, width=1)),
                customdata=np.c_[nbr["neighbor_id"], nbr["cosine_sim"], nbr["outcome_at_tau"], nbr["rule"]],
                hovertemplate="Сосед #%{customdata[0]} (сходство %{customdata[1]:.3f})<br>Исход: %{customdata[2]} · %{customdata[3]}<extra></extra>",
            ))

    style_map = {"TP": (ORANGE, "circle"), "FN": (MISS, "diamond"), "FP": (BLUE, "triangle-up"), "TN": (GRAY, "square")}
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

    fig.add_trace(go.Scatter(
        x=[sel_row[xcol]], y=[sel_row[ycol]], mode="markers", name="выбранная карточка",
        marker=dict(size=22, symbol="circle-open", color=INK, line=dict(color=INK, width=2.5)),
        hoverinfo="skip",
    ))

    fig.update_xaxes(title_text=xlab, title_standoff=8, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text=ylab, title_standoff=8, showgrid=True, gridcolor=GRID)
    return base(
        fig, 520, legend=True,
        legend_y=1.04,
        margin=dict(l=12, r=16, t=72 if is_discrete_cluster else 56, b=66),
    )


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
    """2 панели: (1) Dot Plot доли mean|SHAP| по выборкам; (2) Ablation |Δp|."""
    imp = store.global_imp
    order = imp[imp["sample"] == "Эксплуатация"].sort_values("rank").head(top_k)["source_column"].tolist()[::-1]
    labels = [_short(REGISTRY[c].label_ru, 34) for c in order]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
        subplot_titles=["Доля вклада SHAP", "Ablation (|Δp|)"],
    )

    for samp, col, sym in [("Приёмка", VAL_COLOR, "circle"), ("Эксплуатация", OPER_COLOR, "circle"), ("Практика 320", ORANGE, "diamond")]:
        s = imp[imp["sample"] == samp].set_index("source_column").reindex(order)
        fig.add_trace(go.Scatter(
            x=s["share_abs_shap"], y=labels, mode="markers", name=samp,
            marker=dict(size=9, symbol=sym, color=col, line=dict(color=SURFACE, width=1.5)),
            customdata=s["mean_abs_shap"],
            hovertemplate=f"{samp} · %{{y}}<br>Доля SHAP: %{{x:.1%}} (mean|SHAP|=%{{customdata:.3f}})<extra></extra>",
        ), row=1, col=1)

    abl = imp[imp["sample"] == "Эксплуатация"].set_index("source_column").reindex(order)
    fig.add_trace(go.Bar(
        x=abl["ablation_delta_proba"], y=labels, orientation="h", showlegend=False,
        marker=dict(color=GRAY, line=dict(width=0)),
        text=[dd.fmt_pct(v, 2) for v in abl["ablation_delta_proba"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK_2),
        hovertemplate="%{y}<br>Ablation |Δp|: %{x:.4f}<extra></extra>",
    ), row=1, col=2)

    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color=INK_2), yshift=4)

    fig.update_xaxes(title_text="Доля вклада", tickformat=".0%", showgrid=True, gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(title_text="Среднее |Δ вероятности|", tickformat=".1%", showgrid=True, gridcolor=GRID,
                     range=[0, float(abl["ablation_delta_proba"].max()) * 1.32 + 1e-4], row=1, col=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(
        fig, 500, legend=True,
        legend_y=1.10,
        margin=dict(l=8, r=28, t=74, b=10),
    )


def _trim_outlier_tails(arr: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.95) -> tuple[np.ndarray, int, int, float, float]:
    """Обрезает экстремальные выбросы слева и справа, если они растягивают ось более чем на 50% ядра."""
    v = arr[np.isfinite(arr)]
    if len(v) < 20 or len(np.unique(v)) <= 10:
        return v, 0, 0, float(v.min()) if len(v) else 0.0, float(v.max()) if len(v) else 1.0
    vmin, vmax = float(v.min()), float(v.max())
    qlo, qhi = float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    span_core = max(qhi - qlo, 1e-9)
    hi_cut = qhi if (vmax - qhi) > 0.5 * span_core else vmax
    lo_cut = qlo if (qlo - vmin) > 0.5 * span_core else vmin
    left_n = int((v < lo_cut).sum())
    right_n = int((v > hi_cut).sum())
    core = v[(v >= lo_cut) & (v <= hi_cut)]
    if len(core) < 10 or float(np.std(core)) < 1e-12:
        return v, 0, 0, vmin, vmax
    return core, left_n, right_n, lo_cut, hi_cut


def fig_ale_pdp_ice(
    store: dd.Store,
    source_col: str,
    period: str,
    eid: int,
    show_ice: bool = True,
    y_mode: str = "Автофокус (ALE/PDP)",
    dist_mode: str = "Гистограмма (равные бины)",
    kde_bw: float = 1.0,
) -> go.Figure:
    """ALE + PDP + ICE + нижняя панель распределения (равные бины гистограммы или динамический KDE)."""
    cn = store.curves_num[(store.curves_num["source_column"] == source_col) & (store.curves_num["period"] == period)].sort_values("grid_idx").reset_index(drop=True)
    if cn.empty:
        return empty(f"Для «{source_col}» недостаточно данных для ALE/PDP")

    meta = REGISTRY[source_col]
    warn = bool(cn["collinear_warn"].iloc[0])
    partner = str(cn["corr_partner"].iloc[0])
    max_r = float(cn["max_spearman"].iloc[0])
    is_discrete = bool(cn["is_discrete"].iloc[0])

    # Берём сырые значения выборки (если есть numeric_dist_sample.parquet) для честного диапазона и KDE
    raw_sample = np.array([], dtype=float)
    if store.numeric_dist is not None and source_col in store.numeric_dist.columns:
        sub_d = store.numeric_dist[store.numeric_dist["period"] == period]
        raw_sample = pd.to_numeric(sub_d[source_col], errors="coerce").dropna().to_numpy(dtype=float)

    left_n, right_n = 0, 0
    if len(raw_sample) >= 20 and not is_discrete:
        core_vals, left_n, right_n, x_lo, x_hi = _trim_outlier_tails(raw_sample)
        cn_vis = cn[(cn["raw_value"] >= x_lo - 1e-9) & (cn["raw_value"] <= x_hi + 1e-9)].reset_index(drop=True)
        if len(cn_vis) < 2:
            cn_vis = cn.copy()
            core_vals = raw_sample
            x_lo, x_hi, left_n, right_n = float(cn["raw_value"].min()), float(cn["raw_value"].max()), 0, 0
    else:
        cn_vis = cn.copy()
        core_vals = raw_sample
        x_lo, x_hi = float(cn_vis["raw_value"].min()), float(cn_vis["raw_value"].max())

    raw_vis = cn_vis["raw_value"].to_numpy(dtype=float)
    vis_gids = set(cn_vis["grid_idx"].astype(int).tolist())
    pdp_base = float(cn["pdp"].mean())
    ale_shifted = cn_vis["ale"].to_numpy(dtype=float) + pdp_base
    pdp_vis = cn_vis["pdp"].to_numpy(dtype=float)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.07)

    # --- Верхняя панель: ICE + PDP + ALE ---
    ice_focus_hi, n_clipped_ice, t5_max = 0.0, 0, 0.0
    bg_max_per_curve = np.array([])
    if show_ice and period == "Эксплуатация":
        ice = store.curves_ice[(store.curves_ice["source_column"] == source_col) & (store.curves_ice["grid_idx"].isin(vis_gids))]
        bg = ice[ice["curve_kind"] == "bg"]
        if len(bg):
            piv = bg.pivot(index="curve_id", columns="grid_idx", values="proba").reindex(columns=cn_vis["grid_idx"].tolist())
            mat = piv.to_numpy(dtype=float)
            if mat.size:
                q25, q75 = np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)
                ice_focus_hi = float(np.nanmax(q75))
                bg_max_per_curve = np.nanmax(mat, axis=1)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([raw_vis, raw_vis[::-1]]),
                    y=np.concatenate([q75, q25[::-1]]),
                    fill="toself", fillcolor=_rgba(BLUE, 0.11), line=dict(width=0),
                    name="ICE (25–75 %)", hoverinfo="skip",
                ), row=1, col=1)
                ix, iy = [], []
                for row_i in range(min(15, len(mat))):
                    ix += raw_vis.tolist() + [None]
                    iy += mat[row_i].tolist() + [None]
                fig.add_trace(go.Scatter(
                    x=ix, y=iy, mode="lines", name="ICE (примеры)",
                    line=dict(color="rgba(71, 84, 103, 0.34)", width=1.1), hoverinfo="skip",
                ), row=1, col=1)

        t5 = ice[(ice["curve_kind"] == "team5") & (ice["curve_id"] == str(eid))].sort_values("grid_idx")
        if len(t5):
            t5_x = t5["raw_value"].to_numpy(dtype=float)
            t5_y = t5["proba"].to_numpy(dtype=float)
            t5_max = float(np.nanmax(t5_y))
            fig.add_trace(go.Scatter(
                x=t5_x, y=t5_y, mode="lines+markers", name=f"ICE {eid}",
                line=dict(color=MISS, width=2.2, dash="dot"), marker=dict(size=5, color=MISS),
                hovertemplate=f"Карточка {eid}: при %{{x:.4g}} → p=%{{y:.2%}}<extra></extra>",
            ), row=1, col=1)
            cv = float(t5["card_raw_value"].iloc[0])
            if np.isfinite(cv) and x_lo <= cv <= x_hi:
                fig.add_vline(x=cv, line=dict(color=MISS, width=1.3, dash="dot"),
                              annotation_text=f"факт {eid}: {_fmt_compact(cv)}", annotation_position="top right",
                              annotation_font=dict(size=10, color=MISS), row=1, col=1, exclude_empty_subplots=False)

    fig.add_trace(go.Scatter(
        x=raw_vis, y=pdp_vis, mode="lines+markers",
        name="PDP (искажён при |ρ|≥0,7)" if warn else "PDP (среднее)",
        line=dict(color=INK_2 if warn else BLUE, width=1.8, dash="dash" if warn else "solid"),
        marker=dict(size=5),
        hovertemplate="PDP при %{x:.4g}: средняя оценка %{y:.2%}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=raw_vis, y=ale_shifted, mode="lines+markers", name="ALE (основной)",
        line=dict(color=ORANGE, width=3), marker=dict(size=7, color=ORANGE, line=dict(color=SURFACE, width=1.5)),
        customdata=cn_vis["ale"],
        hovertemplate="ALE при %{x:.4g}: эффект %{customdata:+.2%} (уровень %{y:.2%})<extra></extra>",
    ), row=1, col=1)

    if y_mode.startswith("Автофокус"):
        focus_top = max(float(ale_shifted.max()), float(pdp_vis.max()), t5_max, ice_focus_hi)
        y_hi = min(1.03, max(focus_top * 1.30 + 0.02, 0.08))
        y_lo = max(-0.01, min(float(ale_shifted.min()), float(pdp_vis.min()), 0.0) - 0.01)
        if len(bg_max_per_curve):
            n_clipped_ice = int((bg_max_per_curve > y_hi).sum())
    else:
        y_lo, y_hi = -0.02, 1.04

    notes = []
    if warn:
        p_ru = REGISTRY[partner].label_ru if partner in REGISTRY else partner
        notes.append(f"⚠ |ρ|={dd.fmt_num(max_r, 2)} с «{_short(p_ru, 22)}» — смотрите ALE")
    if n_clipped_ice > 0:
        notes.append(f"фокус Y до {y_hi:.0%} ({n_clipped_ice} ICE выше)")
    if notes:
        fig.add_annotation(
            x=0.01, y=0.99, xref="x domain", yref="y domain", xanchor="left", yanchor="top", showarrow=False,
            text=" · ".join(notes), font=dict(size=10, color=MISS if warn else MUTED),
            bgcolor="rgba(254,244,243,0.92)" if warn else "rgba(255,255,255,0.88)",
            bordercolor=MISS if warn else GRAY_LIGHT, borderwidth=1, borderpad=3,
        )

    # --- Нижняя панель: честная гистограмма (равные бины) ИЛИ динамический KDE ---
    dist_y_title = "Карточек"
    use_kde = dist_mode.startswith("KDE") and not is_discrete and len(core_vals) >= 20 and len(np.unique(core_vals)) > 10
    if len(core_vals) >= 10:
        if use_kde:
            bw = float(np.clip(kde_bw, 0.2, 2.5))
            kde = gaussian_kde(core_vals, bw_method=lambda s, b=bw: max(s.scotts_factor() * b, 1e-3))
            xg = np.linspace(x_lo, x_hi, 220)
            yg = kde(xg)
            fig.add_trace(go.Scatter(
                x=xg, y=yg, mode="lines", fill="tozeroy",
                fillcolor=_rgba(BLUE, 0.25), line=dict(color=BLUE, width=2),
                showlegend=False,
                hovertemplate="Значение %{x:.4g}<br>Плотность KDE: %{y:.4g}<extra></extra>",
            ), row=2, col=1)
            dist_y_title = f"Плотность (bw={dd.fmt_num(bw, 1)})"
        elif is_discrete or len(np.unique(core_vals)) <= 12:
            uv, cnt = np.unique(core_vals, return_counts=True)
            step = float(np.min(np.diff(uv))) if len(uv) >= 2 else 1.0
            fig.add_trace(go.Bar(
                x=uv, y=cnt, width=max(step * 0.5, 0.25), showlegend=False,
                marker=dict(color="#5c8ec9", opacity=0.88, line=dict(color=BLUE, width=0.6)),
                hovertemplate="Значение %{x:.4g}: %{y:,} карт.<extra></extra>",
            ), row=2, col=1)
            if dist_mode.startswith("KDE"):
                fig.add_annotation(
                    x=0.99, y=0.95, xref="x2 domain", yref="y2 domain", xanchor="right", yanchor="top", showarrow=False,
                    text="Дискретный признак: показаны точные частоты вместо KDE",
                    font=dict(size=10, color=MUTED), bgcolor="rgba(255,255,255,0.88)",
                )
        else:
            cnt, edges = np.histogram(core_vals, bins=32, range=(x_lo, x_hi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = (edges[1] - edges[0]) * 0.88
            hovers = [f"Интервал [{_fmt_compact(edges[i])}; {_fmt_compact(edges[i + 1])}]: {dd.fmt_int(cnt[i])} карт." for i in range(len(cnt))]
            fig.add_trace(go.Bar(
                x=centers, y=cnt, width=width, showlegend=False,
                marker=dict(color="#5c8ec9", opacity=0.88, line=dict(color=BLUE, width=0.5)),
                hovertext=hovers, hoverinfo="text",
            ), row=2, col=1)
    else:
        # Запасной вариант, если numeric_dist_sample.parquet ещё не сгенерирован
        fig.add_trace(go.Bar(x=raw_vis, y=cn_vis["bin_n"], showlegend=False, marker=dict(color="#5c8ec9")), row=2, col=1)

    # Явные подписи скрытых выбросов-хвостов
    tail_parts = []
    if left_n > 0:
        tail_parts.append(f"← левый хвост < {_fmt_compact(x_lo)}: {dd.fmt_int(left_n)} карт.")
    if right_n > 0:
        tail_parts.append(f"правый хвост > {_fmt_compact(x_hi)}: {dd.fmt_int(right_n)} карт. →")
    if tail_parts:
        fig.add_annotation(
            x=0.99, y=0.95, xref="x2 domain", yref="y2 domain", xanchor="right", yanchor="top", showarrow=False,
            text=" · ".join(tail_parts), font=dict(size=10, color=INK_2),
            bgcolor="rgba(243,242,238,0.92)", bordercolor=AXIS, borderwidth=1, borderpad=3,
        )

    fig.update_yaxes(title_text="Оценка вероятности", range=[y_lo, y_hi], tickformat=".0%",
                     showgrid=False, showline=True, linecolor=AXIS, row=1, col=1)
    fig.update_yaxes(title_text=dist_y_title, nticks=3, showgrid=True, gridcolor=GRID, row=2, col=1)
    fig.update_xaxes(title_text=f"{meta.label_ru} ({meta.source_column})", range=[x_lo - (x_hi - x_lo) * 0.02, x_hi + (x_hi - x_lo) * 0.02], row=2, col=1)
    return base(fig, 500, legend=True, legend_y=1.05, margin=dict(l=8, r=16, t=54, b=14))


def fig_corr_heatmap(store: dd.Store, min_abs_corr: float = 0.50) -> go.Figure:
    """Кластеризованная тепловая карта корреляций Спирмена (только признаки со связью |ρ| ≥ min_abs_corr)."""
    if store.corr_numeric is None or store.corr_numeric.empty:
        return empty("Перезапустите prepare.py для расчёта полной матрицы корреляций (corr_numeric.parquet)")

    piv = store.corr_numeric.pivot(index="feat_a", columns="feat_b", values="spearman").fillna(0.0)
    cols = list(piv.columns)
    mat = piv.to_numpy(dtype=float).copy()
    np.fill_diagonal(mat, 0.0)
    keep = np.flatnonzero(np.max(np.abs(mat), axis=1) >= min_abs_corr)
    if len(keep) < 2:
        return empty(f"Нет пар числовых признаков с |ρ| ≥ {dd.fmt_num(min_abs_corr, 2)}")

    sub_cols = [cols[i] for i in keep]
    sub_mat = piv.loc[sub_cols, sub_cols].to_numpy(dtype=float)

    # Иерархическая сортировка, чтобы коррелированные группы образовали яркие блоки у диагонали
    if len(sub_cols) >= 3:
        dist = np.clip(1.0 - np.abs(sub_mat), 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        order = leaves_list(linkage(squareform(dist), method="average"))
        sub_cols = [sub_cols[i] for i in order]
        sub_mat = sub_mat[np.ix_(order, order)]

    labels = [_short(REGISTRY[c].label_ru if c in REGISTRY else c, 28) for c in sub_cols]
    text = [[f"{v:+.2f}" if (i != j and abs(v) >= min_abs_corr) else "" for j, v in enumerate(row)] for i, row in enumerate(sub_mat)]

    fig = go.Figure(go.Heatmap(
        z=sub_mat, x=labels, y=labels, text=text, texttemplate="%{text}",
        textfont=dict(size=9, color=INK),
        colorscale=[[0.0, BLUE], [0.5, "#f7f7f7"], [1.0, ORANGE]], zmin=-1.0, zmax=1.0,
        colorbar=dict(title="Спирмен ρ", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Спирмен ρ = %{z:+.2f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10, color=INK_2), side="bottom")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10, color=INK_2), showgrid=False)
    h = max(420, min(680, 160 + 24 * len(sub_cols)))
    return base(fig, h, margin=dict(l=8, r=16, t=16, b=100))


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
def fig_drift_dumbbell(
    store: dd.Store,
    sort_by: str = "По сдвигу |SHAP|",
    only_high_psi: bool = True,
    page_idx: int = 0,
    page_size: int = dd.DRIFT_PAGE_SIZE,
) -> go.Figure:
    """Гантели сдвига mean|SHAP| с пагинацией и общим масштабом оси X по всему отфильтрованному списку."""
    ds = dd.drift_filtered(store, sort_by, only_high_psi)
    if ds.empty:
        return empty("Нет полей, подходящих под выбранный фильтр дрейфа")

    # Единый максимум оси X по всем страницам, чтобы длины гантелей были сравнимы при перелистывании
    xmax = float(max(ds["mean_abs_shap_val"].max(), ds["mean_abs_shap_oper"].max())) * 1.28 + 1e-4

    start = page_idx * page_size
    page_df = ds.iloc[start:start + page_size]
    if page_df.empty:
        page_df = ds.iloc[:page_size]

    top = page_df.iloc[::-1]
    names = [_short(r.label_ru, 34) for r in top.itertuples(index=False)]
    v_arr = top["mean_abs_shap_val"].to_numpy(dtype=float)
    o_arr = top["mean_abs_shap_oper"].to_numpy(dtype=float)

    fig = go.Figure()
    for nm, a, b in zip(names, v_arr, o_arr):
        fig.add_shape(type="line", x0=a, x1=b, y0=nm, y1=nm,
                      line=dict(color=ORANGE_LIGHT if b > a else BLUE_LIGHT, width=3), layer="below")

    fig.add_trace(go.Scatter(
        x=v_arr, y=names, mode="markers", name="приёмка",
        marker=dict(size=11, color=VAL_COLOR, line=dict(color=SURFACE, width=1.8)),
        hovertemplate="%{y}<br>приёмка mean|SHAP|: %{x:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=o_arr, y=names, mode="markers", name="эксплуатация",
        marker=dict(size=11, color=OPER_COLOR, line=dict(color=SURFACE, width=1.8)),
        customdata=np.c_[top["psi"], top["null_val"], top["null_oper"]],
        hovertemplate="%{y}<br>эксплуатация mean|SHAP|: %{x:.3f}<br>PSI: %{customdata[0]:.2f} · пропуски: %{customdata[1]:.0%} → %{customdata[2]:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=np.maximum(v_arr, o_arr) + xmax * 0.015, y=names, mode="text", showlegend=False,
        text=[f"PSI {dd.fmt_num(p, 2)}" for p in top["psi"]],
        textposition="middle right",
        textfont=dict(size=10, color=[MISS if p >= 0.25 else MUTED for p in top["psi"]]),
        hoverinfo="skip",
    ))

    fig.update_xaxes(title_text="Средний абсолютный вклад mean|SHAP| (log-odds)",
                     range=[-xmax * 0.02, xmax], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, 440, legend=True, legend_y=1.04, margin=dict(l=8, r=70, t=42, b=36))


def fig_drift_dist(store: dd.Store, source_col: str) -> go.Figure:
    """Сравнение распределения выбранного поля в приёмке и эксплуатации (без наложения процентов на ось Y)."""
    bins = store.drift_bins[store.drift_bins["source_column"] == source_col].sort_values("bin_idx")
    if bins.empty:
        return empty(f"Нет бинов распределения для «{source_col}»")

    is_num = REGISTRY[source_col].last_step == "QuantileTransformer"
    fig = go.Figure()
    if is_num:
        fig.add_trace(go.Bar(
            x=bins["bin_label"], y=bins["val_share"], name="приёмка",
            marker=dict(color=VAL_COLOR, opacity=0.8), customdata=bins["val_n"],
            hovertemplate="Интервал %{x}<br>Приёмка: %{y:.1%} (%{customdata:,})<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=bins["bin_label"], y=bins["oper_share"], name="эксплуатация",
            marker=dict(color=OPER_COLOR, opacity=0.85), customdata=bins["oper_n"],
            hovertemplate="Интервал %{x}<br>Эксплуатация: %{y:.1%} (%{customdata:,})<extra></extra>",
        ))
        fig.update_yaxes(title_text="Доля карточек", tickformat=".0%")
        fig.update_xaxes(title_text=REGISTRY[source_col].label_ru, tickangle=-25)
        return base(fig, 420, legend=True, barmode="group", margin=dict(l=8, r=16, t=42, b=20))

    b = bins.iloc[::-1]
    lbls = [_short(s, 36) for s in b["bin_label"]]
    va = b["val_share"].to_numpy(dtype=float)
    oa = b["oper_share"].to_numpy(dtype=float)
    xmax = min(1.10, float(max(va.max(), oa.max())) * 1.22 + 0.06)

    for nm, a, c in zip(lbls, va, oa):
        fig.add_shape(type="line", x0=a, x1=c, y0=nm, y1=nm, line=dict(color=GRAY_LIGHT, width=3), layer="below")

    fig.add_trace(go.Scatter(
        x=va, y=lbls, mode="markers", name="приёмка",
        marker=dict(size=10, color=VAL_COLOR, line=dict(color=SURFACE, width=1.5)),
        hovertemplate="%{y}<br>приёмка: %{x:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=oa, y=lbls, mode="markers", name="эксплуатация",
        marker=dict(size=11, color=OPER_COLOR, line=dict(color=SURFACE, width=1.5)),
        hovertemplate="%{y}<br>эксплуатация: %{x:.1%}<extra></extra>",
    ))

    # Умное размещение процентов: слева от синей точки только если есть запас >= 10 п.п. от оси Y,
    # иначе — справа от всей гантели (max(val, oper)), чтобы не перекрывать ни подписи, ни серую точку.
    tx, tpos = [], []
    pad = xmax * 0.022
    for a, c in zip(va, oa):
        if c < a and c >= 0.10:
            tx.append(c - pad)
            tpos.append("middle left")
        else:
            tx.append(max(a, c) + pad)
            tpos.append("middle right")

    fig.add_trace(go.Scatter(
        x=tx, y=lbls, mode="text", showlegend=False,
        text=[dd.fmt_pct(v, 0) for v in oa],
        textposition=tpos, textfont=dict(size=10, color=INK_2), hoverinfo="skip",
    ))

    fig.update_xaxes(title_text="Доля карточек в потоке", tickformat=".0%",
                     range=[-0.025, xmax], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(fig, max(360, 74 + 30 * len(b)), legend=True, margin=dict(l=8, r=40, t=42, b=10))


# ================================================================ 5. ПРОВЕРКА ОБЪЯСНЕНИЙ (КЕЙСЫ)
def fig_verify_wilson(
    store: dd.Store,
    eid: int,
    period: str = "Эксплуатация",
    scale_mode: str = "Автофокус по ДИ",
    show_ci: bool = True,
) -> go.Figure:
    """Две панели: (1) SHAP-вклад топ-12 признаков кейса; (2) фактическая доля инцидентов + 95 % ДИ Уилсона."""
    df = store.verification[(store.verification["EntityId"] == eid) & (store.verification["period"] == period)].sort_values("rank_abs")
    if df.empty:
        return empty("Нет статистики верификации для карточки")

    d = df.iloc[::-1]
    labels = [f"{_short(r.label_ru, 28)} ({_short(r.slice_desc, 16)})" for r in d.itertuples(index=False)]
    base_rate = float(d["base_inc_rate"].iloc[0])

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.42, 0.58], horizontal_spacing=0.06,
        subplot_titles=["Вклад SHAP модели (log-odds)", f"Фактическая доля инцидентов и 95 % ДИ ({period})"],
    )

    # Панель 1: горизонтальные столбцы SHAP
    fig.add_trace(go.Bar(
        x=d["shap_val"], y=labels, orientation="h", showlegend=False,
        marker=dict(color=[ORANGE if v >= 0 else BLUE for v in d["shap_val"]], line=dict(width=0)),
        text=[dd.fmt_signed(v, 2) for v in d["shap_val"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=INK_2),
        hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.2), row=1, col=1, exclude_empty_subplots=False)

    # Панель 2: точки фактической доли + контрастные усы 95 % ДИ Уилсона
    st_color = {"Подтверждено": ORANGE, "Противоречие": MISS, "Нейтрально": GRAY, "Мало данных": GRAY_LIGHT}
    colors = [st_color.get(s, GRAY) for s in d["status"]]

    err_kw = (
        dict(
            type="data", symmetric=False,
            array=d["ci_hi"] - d["inc_rate"], arrayminus=d["inc_rate"] - d["ci_lo"],
            color=INK_2, thickness=2.0, width=4,
        )
        if show_ci else None
    )
    fig.add_trace(go.Scatter(
        x=d["inc_rate"], y=labels, mode="markers", showlegend=False,
        error_x=err_kw,
        marker=dict(size=9, color=colors, line=dict(color=SURFACE, width=1.6)),
        customdata=np.c_[d["k"], d["n"], d["ci_lo"], d["ci_hi"], d["status"]],
        hovertemplate="%{y}<br>Инцидентов: %{customdata[0]} из %{customdata[1]} (%{x:.1%}; ДИ %{customdata[2]:.1%}–%{customdata[3]:.1%})<br><b>%{customdata[4]}</b><extra></extra>",
    ), row=1, col=2)

    # Вычисляем диапазон правой оси (автофокус делает узкие ДИ Уилсона хорошо различимыми)
    max_hi = float(np.nanmax(np.r_[d["ci_hi"].to_numpy(dtype=float), [base_rate]]))
    if scale_mode.startswith("Автофокус"):
        x_right = min(1.08, max(max_hi * 1.28 + 0.04, base_rate * 2.2))
    else:
        x_right = 1.12

    # Компактные подписи «доля (n=...)» справа от уса — сразу понятно, почему при большом n усы узкие
    txt_x = np.where(np.isfinite(d["ci_hi"]) & show_ci, d["ci_hi"], d["inc_rate"]) + x_right * 0.015
    fig.add_trace(go.Scatter(
        x=txt_x, y=labels, mode="text", showlegend=False,
        text=[f"{dd.fmt_pct(r.inc_rate, 1)} (n={dd.fmt_int(r.n)})" if np.isfinite(r.inc_rate) else "нет данных"
              for r in d.itertuples(index=False)],
        textposition="middle right", textfont=dict(size=10, color=MUTED), hoverinfo="skip",
    ), row=1, col=2)

    # Пунктир среднего по потоку — ПОСЛЕ add_trace и с exclude_empty_subplots=False (теперь всегда виден!)
    fig.add_vline(
        x=base_rate, line=dict(color=INK_2, width=1.4, dash="dash"),
        annotation_text=f"среднее {dd.fmt_pct(base_rate, 1)}",
        annotation_position="top right", annotation_font=dict(size=10, color=INK_2),
        row=1, col=2, exclude_empty_subplots=False,
    )

    for st_name, col in [("Подтверждено", ORANGE), ("Противоречие", MISS), ("Нейтрально", GRAY), ("Мало данных", GRAY_LIGHT)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=st_name,
                                 marker=dict(size=9, color=col)), row=1, col=2)

    for ann in fig.layout.annotations:
        if ann.text in ("Вклад SHAP модели (log-odds)", f"Фактическая доля инцидентов и 95 % ДИ ({period})"):
            ann.update(font=dict(size=12, color=INK_2), yshift=4)

    smax = max(float(d["shap_val"].abs().max()), 0.5) * 1.28
    fig.update_xaxes(range=[-smax, smax], title_text="SHAP (log-odds)", showgrid=True, gridcolor=GRID, row=1, col=1)
    fig.update_xaxes(range=[-x_right * 0.03, x_right], title_text="Фактическая доля инцидентов",
                     tickformat=".0%", showgrid=True, gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=INK_2))
    return base(
        fig, 500, legend=True,
        legend_x=0.45, legend_y=1.11,
        margin=dict(l=8, r=24, t=76, b=10),
    )