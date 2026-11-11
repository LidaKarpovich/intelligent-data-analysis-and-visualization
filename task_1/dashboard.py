"""
dashboard.py — дашборд Taipy «SOC · триаж алертов SIEM»: поток алертов, исходы карточек,
скорость реакции и распределения (аналитика без ML-модели).

Запуск из папки проекта:
    python dashboard.py
и открыть http://localhost:5002

Страницы (на каждой — KPI-плитки и графики; выводы текстом — в REPORT.md, см. report.py):
    Обзор              — ключевые цифры, динамика и состав исходов
    Сравнение          — правила SIEM, категории IRP, нагрузка по часам, таблица
    Распределения      — время реакции и решения, интенсивность потока, признаки
    Потоки и плотность — категория → вердикт, изменение потока, плотность, источники, правила
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from taipy.gui import Gui, navigate

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_insights as ins
import dashboard_kpis as kp
from dashboard_charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страниц)
from dashboard_data import CRIT_ORDER, DAY_TYPES, PERIODS, SOURCES  # noqa: F401

STORE = dd.load()

# ---------------------------------------------------------------- состояние (значения по умолчанию)
period, day_type, source, crit_sel = PERIODS[0], DAY_TYPES[0], dd.ALL, list(CRIT_ORDER)
HIST_LOV = [dd.ALL] + CRIT_ORDER
hist_crit = "Критическая"
qq_log = False
KDE_LOV = list(ch.KDE_FEATURES)
kde_feature = KDE_LOV[0]
VIOLIN_LOV = ch.VIOLIN_GROUPS
violin_by = VIOLIN_LOV[0]
SANKEY_LOV = list(ch.SANKEY_LEFT)
sankey_left = SANKEY_LOV[0]
HEX_LOV = ch.HEX_MODES
hex_mode = HEX_LOV[0]

NAV = [("/overview", "Обзор"), ("/compare", "Сравнение"), ("/eda", "Распределения"),
       ("/flows", "Потоки и плотность")]


def _filters(s) -> dd.Filters:
    return dd.Filters(s.period, s.day_type, s.source, list(s.crit_sel or []))


# ================================================================ расчёт представлений
# На листах — только графики и KPI-плитки. Текстовые выводы собирает report.py.
def view_header(d: pd.DataFrame, f: dd.Filters, dates: pd.DatetimeIndex | None = None) -> dict:
    return dict(period_caption=ins.kpis(d, f, dates)["period_caption"])


def view_overview(store: dd.Store, f: dd.Filters, d: pd.DataFrame) -> dict:
    dates = dd.scope_dates(store, f)
    out = dict(fig_daily=ch.fig_daily(d, dates), fig_mix=ch.fig_class_mix(d), fig_verdicts=ch.fig_verdicts(d))
    out.update(kp.overview(d, dates))
    return out


def view_compare(store: dd.Store, f: dd.Filters, d: pd.DataFrame) -> dict:
    dates = dd.scope_dates(store, f)
    out = dict(fig_top_rules=ch.fig_top_rules(d), fig_fp_dev=ch.fig_fp_deviation(d),
               fig_heat=ch.fig_heatmap_hour(d, dates), rules_tbl=ch.rules_table(d))
    out.update(kp.compare(d))
    return out


def view_hist(d: pd.DataFrame, crit: str) -> dict:
    sub = d if crit == dd.ALL else d[d["crit"] == crit]
    return dict(fig_hist=ch.fig_histogram(sub))


def view_qq(store: dd.Store, f: dd.Filters, log: bool) -> dict:
    fig, _, _ = ch.fig_qq(dd.hourly_counts(store, f), log)
    return dict(fig_qq=fig)


def view_kde(d: pd.DataFrame, feature: str) -> dict:
    fig, _ = ch.fig_kde(d, feature)
    return dict(fig_kde=fig)


def view_violin(d: pd.DataFrame, by: str) -> dict:
    return dict(fig_violin=ch.fig_violin(d, by))


def view_eda(store: dd.Store, f: dd.Filters, d: pd.DataFrame, ctl: dict) -> dict:
    out = dict(fig_rain=ch.fig_raincloud(d))
    out.update(view_hist(d, ctl["hist_crit"]))
    out.update(view_qq(store, f, ctl["qq_log"]))
    out.update(view_kde(d, ctl["kde_feature"]))
    out.update(view_violin(d, ctl["violin_by"]))
    out.update(kp.eda(d, dd.hourly_counts(store, f)))
    return out


def view_sankey(d: pd.DataFrame, left: str) -> dict:
    return dict(fig_sankey=ch.fig_sankey(d, left))


def view_hex(store: dd.Store, f: dd.Filters, d: pd.DataFrame, mode: str) -> dict:
    hourly = None if mode == ch.HEX_MODES[0] else dd.hourly_counts(store, f)
    fig, _ = ch.fig_hexbin(d, mode, hourly)
    return dict(fig_hex=fig, hex_title=ch.HEX_QUESTIONS[mode])


def view_flows(store: dd.Store, d: pd.DataFrame, f: dd.Filters, left: str, hexm: str) -> dict:
    prev, cur, n_days = dd.previous_window(store, f)
    out = dict(fig_wf=ch.fig_waterfall(prev, cur, n_days), fig_bubble=ch.fig_bubble(d))
    out.update(view_sankey(d, left))
    out.update(view_hex(store, f, d, hexm))
    out.update(kp.flows(d, prev, cur, n_days, store.top_src))
    return out


def compute_all(store: dd.Store, f: dd.Filters, ctl: dict) -> dict:
    d = dd.apply_filters(store, f)
    dates = dd.scope_dates(store, f)
    out = {}
    out.update(view_header(d, f, dates))
    out.update(view_overview(store, f, d))
    out.update(view_compare(store, f, d))
    out.update(view_eda(store, f, d, ctl))
    out.update(view_flows(store, d, f, ctl["sankey_left"], ctl["hex_mode"]))
    return out


_ctl0 = dict(hist_crit=hist_crit, qq_log=qq_log, kde_feature=kde_feature, violin_by=violin_by,
             sankey_left=sankey_left, hex_mode=hex_mode)
globals().update(compute_all(STORE, dd.Filters(), _ctl0))   # начальные значения фигур и выводов для Taipy
fig_dot = ch.fig_dotplot(STORE.top_src)            # агрегат etl.py за весь период — фильтры к нему не применимы


# ================================================================ обработчики
def _assign(state, values: dict):
    with state:                                    # одно обновление интерфейса вместо десятка
        for k, v in values.items():
            setattr(state, k, v)


def _ctl(state) -> dict:
    return {k: getattr(state, k) for k in _ctl0}


def on_filter(state, var_name=None, value=None):
    _assign(state, compute_all(STORE, _filters(state), _ctl(state)))


def on_reset(state, *args):
    with state:
        state.period, state.day_type, state.source, state.crit_sel = PERIODS[0], DAY_TYPES[0], dd.ALL, list(CRIT_ORDER)
    on_filter(state)


def _d(state) -> pd.DataFrame:
    return dd.apply_filters(STORE, _filters(state))


def on_hist(state, var_name, value):
    _assign(state, view_hist(_d(state), state.hist_crit))


def on_qq(state, var_name, value):
    _assign(state, view_qq(STORE, _filters(state), state.qq_log))


def on_kde(state, var_name, value):
    _assign(state, view_kde(_d(state), state.kde_feature))


def on_violin(state, var_name, value):
    _assign(state, view_violin(_d(state), state.violin_by))


def on_sankey(state, var_name, value):
    _assign(state, view_sankey(_d(state), state.sankey_left))


def on_hex(state, var_name, value):
    _assign(state, view_hex(STORE, _filters(state), _d(state), state.hex_mode))


def on_init(state):
    navigate(state, "overview")


# ================================================================ разметка страниц
root_page = """
<|part|class_name=topbar|
<|layout|columns=1 auto|class_name=topbar-row|
<|part|class_name=brand|
<|SOC · триаж алертов SIEM|text|class_name=brand-title|>
<|{period_caption}|text|class_name=brand-sub|>
|>
<|navbar|lov={NAV}|class_name=nav|>
|>

<|layout|columns=1 1 1 2 auto|gap=12px|class_name=filters|
<|{period}|selector|lov={PERIODS}|dropdown|label=Период|on_change=on_filter|>

<|{day_type}|selector|lov={DAY_TYPES}|dropdown|label=Тип дня|on_change=on_filter|>

<|{source}|selector|lov={SOURCES}|dropdown|label=Источник инцидента|on_change=on_filter|>

<|{crit_sel}|selector|lov={CRIT_ORDER}|dropdown|multiple|label=Критичность|on_change=on_filter|>

<|Сбросить фильтры|button|on_action=on_reset|class_name=reset|>
|>
|>

<|part|class_name=page|
<|content|>
|>
"""

overview_page = """
""" + kp.markup("overview") + """

<|part|class_name=card|
<|Сколько карточек поступает каждый день|text|class_name=card-title|>
<|chart|figure={fig_daily}|plot_config={PLOT_CONFIG}|height=340px|>
|>

<|layout|columns=3 2|gap=16px|
<|part|class_name=card|
<|Как меняется исход карточек|text|class_name=card-title|>
<|chart|figure={fig_mix}|plot_config={PLOT_CONFIG}|height=380px|>
|>
<|part|class_name=card|
<|Какой вердикт получает карточка?|text|class_name=card-title|>
<|chart|figure={fig_verdicts}|plot_config={PLOT_CONFIG}|height=380px|>
|>
|>
"""

compare_page = """
""" + kp.markup("compare") + """

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|Какие правила SIEM дают больше всего карточек|text|class_name=card-title|>
<|chart|figure={fig_top_rules}|plot_config={PLOT_CONFIG}|height=400px|>
|>
<|part|class_name=card|
<|Где «шума» больше среднего|text|class_name=card-title|>
<|chart|figure={fig_fp_dev}|plot_config={PLOT_CONFIG}|height=400px|>
|>
|>

<|part|class_name=card|
<|Когда поступают алерты: день недели × час|text|class_name=card-title|>
<|chart|figure={fig_heat}|plot_config={PLOT_CONFIG}|height=300px|>
|>

<|part|class_name=card|
<|У каких правил объём, исход и скорость реакции не совпадают?|text|class_name=card-title|>
<|{rules_tbl}|table|show_all|downloadable|class_name=rules-table|>
|>
"""

eda_page = """
""" + kp.markup("eda") + """

<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Сколько ждут карточки до взятия в работу|text|class_name=card-title|>

<|{hist_crit}|selector|lov={HIST_LOV}|dropdown|label=Критичность|on_change=on_hist|>
|>
<|chart|figure={fig_hist}|plot_config={PLOT_CONFIG}|height=360px|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Подходит ли «правило трёх сигм» для порогов SIEM|text|class_name=card-title|>

<|{qq_log}|toggle|label=ln(1 + x)|on_change=on_qq|>
|>
<|chart|figure={fig_qq}|plot_config={PLOT_CONFIG}|height=360px|>
|>
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Чем реальные инциденты отличаются от False Positive|text|class_name=card-title|>

<|{kde_feature}|selector|lov={KDE_LOV}|dropdown|label=Признак|on_change=on_kde|>
|>
<|chart|figure={fig_kde}|plot_config={PLOT_CONFIG}|height=360px|>
|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Какие группы ждут взятия в работу дольше?|text|class_name=card-title|>

<|{violin_by}|selector|lov={VIOLIN_LOV}|dropdown|label=Группировка|on_change=on_violin|>
|>
<|chart|figure={fig_violin}|plot_config={PLOT_CONFIG}|height=400px|>
|>
<|part|class_name=card|
<|Насколько дольше живёт карточка после эскалации на L2?|text|class_name=card-title|>
<|chart|figure={fig_rain}|plot_config={PLOT_CONFIG}|height=400px|>
|>
|>
"""

flows_page = """
""" + kp.markup("flows") + """

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Куда уходят карточки: категория → вердикт|text|class_name=card-title|>

<|{sankey_left}|selector|lov={SANKEY_LOV}|dropdown|label=Слева|on_change=on_sankey|>
|>
<|chart|figure={fig_sankey}|plot_config={PLOT_CONFIG}|height=480px|>
|>
<|part|class_name=card|
<|За счёт чего изменился поток карточек|text|class_name=card-title|>
<|chart|figure={fig_wf}|plot_config={PLOT_CONFIG}|height=480px|>
|>
|>

<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|{hex_title}|text|class_name=card-title|>

<|{hex_mode}|selector|lov={HEX_LOV}|dropdown|label=Оси|on_change=on_hex|>
|>
<|chart|figure={fig_hex}|plot_config={PLOT_CONFIG}|height=460px|>
|>

<|layout|columns=2 3|gap=16px|
<|part|class_name=card|
<|Какие адреса-источники порождают больше всего карточек?|text|class_name=card-title|>
<|chart|figure={fig_dot}|plot_config={PLOT_CONFIG}|height=520px|>
|>
<|part|class_name=card|
<|Какие правила долго разбирают и при этом находят реальные инциденты?|text|class_name=card-title|>
<|chart|figure={fig_bubble}|plot_config={PLOT_CONFIG}|height=520px|>
|>
|>
"""

pages = {"/": root_page, "overview": overview_page, "compare": compare_page, "eda": eda_page,
         "flows": flows_page}

stylekit = {
    "color_primary": "#2a78d6", "color_secondary": "#52514e",
    "color_background_light": "#f6f7f9", "color_paper_light": "#ffffff",
    "font_family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "border_radius": 10, "input_button_height": "40px", "root_margin": "0",
}

# Gui создаётся на уровне модуля: Taipy берёт переменные страниц из кадра, где вызван Gui()
gui = Gui(pages=pages, css_file="dashboard.css")

if __name__ == "__main__":
    gui.run(title="SOC · триаж алертов SIEM", dark_mode=False, stylekit=stylekit, port=5002,
            watermark="", use_reloader=False)
