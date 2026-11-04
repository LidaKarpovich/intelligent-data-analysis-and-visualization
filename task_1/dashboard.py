"""
dashboard.py — дашборд Taipy «SOC · триаж алертов SIEM»: поток алертов, исходы карточек,
скорость реакции и распределения (аналитика без ML-модели).

Запуск из папки проекта:
    python dashboard.py
и открыть http://localhost:5002

Страницы:
    Обзор              — ключевые цифры, главные выводы, динамика и состав исходов
    Сравнение          — правила SIEM, категории IRP, нагрузка по часам, таблица
    Распределения      — время реакции и решения, интенсивность потока, признаки
    Потоки и плотность — категория → вердикт, изменение потока, плотность, источники, правила
    О данных           — как читать дашборд и оговорки
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from taipy.gui import Gui, navigate

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_insights as ins
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
       ("/flows", "Потоки и плотность"), ("/about", "О данных")]


def _filters(s) -> dd.Filters:
    return dd.Filters(s.period, s.day_type, s.source, list(s.crit_sel or []))


# ================================================================ расчёт представлений
def view_header(d: pd.DataFrame, f: dd.Filters) -> dict:
    if d.empty:
        return dict(period_caption="Нет карточек для выбранных фильтров", kpi_n="0", kpi_n_note="",
                    kpi_fp="—", kpi_fp_note="", kpi_real="—", kpi_real_note="", kpi_none="—",
                    kpi_none_note="", kpi_react="—", kpi_react_note="")
    n = len(d)
    cls = d["cls"].value_counts()
    days = d["date"].nunique()
    react = d["react_min"].dropna()
    parts = [f"{dd.fmt_date(d['date'].min())} — {dd.fmt_date(d['date'].max())}"]
    if f.day_type != DAY_TYPES[0]:
        parts.append(f.day_type.lower())
    if f.source != dd.ALL:
        parts.append(f.source)
    if f.crit and len(f.crit) < len(CRIT_ORDER):
        parts.append("критичность: " + ", ".join(c.lower() for c in CRIT_ORDER if c in f.crit))
    return dict(
        period_caption=" · ".join(parts),
        kpi_n=dd.fmt_int(n), kpi_n_note=f"≈ {dd.fmt_int(n / max(days, 1))} в день, {dd.fmt_int(days)} дн.",
        kpi_fp=dd.fmt_pct(cls.get(dd.CLS_FP, 0) / n), kpi_fp_note=f"{dd.fmt_int(cls.get(dd.CLS_FP, 0))} карточек",
        kpi_real=dd.fmt_int(cls.get(dd.CLS_REAL, 0)), kpi_real_note=f"{dd.fmt_pct(cls.get(dd.CLS_REAL, 0) / n)} потока",
        kpi_none=dd.fmt_pct(cls.get(dd.CLS_NONE, 0) / n), kpi_none_note=f"{dd.fmt_int(cls.get(dd.CLS_NONE, 0))} без вердикта",
        kpi_react=dd.fmt_minutes(react.median()) if len(react) else "—",
        kpi_react_note=f"90 % — быстрее {dd.fmt_minutes(react.quantile(0.9))}" if len(react) else "",
    )


def view_overview(store: dd.Store, f: dd.Filters, d: pd.DataFrame) -> dict:
    dates = dd.scope_dates(store, f)
    return dict(fig_daily=ch.fig_daily(d, dates), fig_mix=ch.fig_class_mix(d), fig_verdicts=ch.fig_verdicts(d),
                ins_summary=ins.summary(d), ins_daily=ins.daily(d, dates),
                ins_mix=ins.mix(d, ch.grain_label(d)) if not d.empty else "", ins_verdicts=ins.verdicts(d))


def view_compare(store: dd.Store, f: dd.Filters, d: pd.DataFrame) -> dict:
    dates = dd.scope_dates(store, f)
    out = dict(fig_top_rules=ch.fig_top_rules(d), fig_fp_dev=ch.fig_fp_deviation(d),
               fig_heat=ch.fig_heatmap_hour(d, dates), rules_tbl=ch.rules_table(d))
    if d.empty:
        out.update(ins_top="", ins_dev="", ins_heat="")
    else:
        out.update(ins_top=ins.top_rules(d), ins_dev=ins.fp_deviation(d), ins_heat=ins.heatmap(d))
    return out


def view_hist(d: pd.DataFrame, crit: str) -> dict:
    sub = d if crit == dd.ALL else d[d["crit"] == crit]
    return dict(fig_hist=ch.fig_histogram(sub), ins_hist=ins.histogram(d, sub, crit))


def view_qq(store: dd.Store, f: dd.Filters, log: bool) -> dict:
    counts = dd.hourly_counts(store, f)
    fig, r2, n = ch.fig_qq(counts, log)
    if not np.isfinite(r2):
        return dict(fig_qq=fig, ins_qq="")
    return dict(fig_qq=fig, ins_qq=ins.qq(counts.to_numpy(float), r2, log))


def view_kde(d: pd.DataFrame, feature: str) -> dict:
    fig, ns = ch.fig_kde(d, feature)
    col, is_log = ch.KDE_FEATURES[feature]
    return dict(fig_kde=fig, ins_kde=ins.kde(d, col, is_log, ns))


def view_violin(d: pd.DataFrame, by: str) -> dict:
    col, groups = ch.violin_groups(d, by)
    return dict(fig_violin=ch.fig_violin(d, by), ins_violin=ins.violin(d, col, groups) if not d.empty else "")


def view_rain(d: pd.DataFrame) -> dict:
    return dict(fig_rain=ch.fig_raincloud(d), ins_rain=ins.raincloud(d))


def view_sankey(d: pd.DataFrame, left: str) -> dict:
    return dict(fig_sankey=ch.fig_sankey(d, left), ins_sankey=ins.sankey(d, ch.SANKEY_LEFT[left]))


def view_hex(store: dd.Store, f: dd.Filters, d: pd.DataFrame, mode: str) -> dict:
    hourly = None if mode == ch.HEX_MODES[0] else dd.hourly_counts(store, f)
    fig, n = ch.fig_hexbin(d, mode, hourly)
    return dict(fig_hex=fig, ins_hex=ins.hexbin(d, mode == ch.HEX_MODES[0], n, hourly))


def view_flows(store: dd.Store, d: pd.DataFrame, f: dd.Filters, left: str, hexm: str) -> dict:
    prev, cur, n_days = dd.previous_window(store, f)
    out = dict(fig_wf=ch.fig_waterfall(prev, cur, n_days),
               ins_wf=ins.waterfall(prev, cur, n_days, f.period not in store.period_dates),
               fig_bubble=ch.fig_bubble(d), ins_bubble=ins.bubble(d) if not d.empty else "")
    out.update(view_sankey(d, left))
    out.update(view_hex(store, f, d, hexm))
    return out


def compute_all(store: dd.Store, f: dd.Filters, ctl: dict) -> dict:
    d = dd.apply_filters(store, f)
    out = {}
    out.update(view_header(d, f))
    out.update(view_overview(store, f, d))
    out.update(view_compare(store, f, d))
    out.update(view_hist(d, ctl["hist_crit"]))
    out.update(view_qq(store, f, ctl["qq_log"]))
    out.update(view_kde(d, ctl["kde_feature"]))
    out.update(view_violin(d, ctl["violin_by"]))
    out.update(view_rain(d))
    out.update(view_flows(store, d, f, ctl["sankey_left"], ctl["hex_mode"]))
    return out


_ctl0 = dict(hist_crit=hist_crit, qq_log=qq_log, kde_feature=kde_feature, violin_by=violin_by,
             sankey_left=sankey_left, hex_mode=hex_mode)
globals().update(compute_all(STORE, dd.Filters(), _ctl0))   # начальные значения фигур и выводов для Taipy
fig_dot = ch.fig_dotplot(STORE.top_src)            # агрегат etl.py за весь период — фильтры к нему не применимы
ins_dot = ins.dotplot(STORE.top_src)


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
<|layout|columns=1 1 1 1 1|gap=12px|class_name=kpis|
<|part|class_name=kpi|
<|Карточек|text|class_name=kpi-label|>
<|{kpi_n}|text|class_name=kpi-value|>
<|{kpi_n_note}|text|class_name=kpi-note|>
|>
<|part|class_name=kpi kpi-fp|
<|Доля False Positive|text|class_name=kpi-label|>
<|{kpi_fp}|text|class_name=kpi-value|>
<|{kpi_fp_note}|text|class_name=kpi-note|>
|>
<|part|class_name=kpi kpi-real|
<|Реальных инцидентов («Решён»)|text|class_name=kpi-label|>
<|{kpi_real}|text|class_name=kpi-value|>
<|{kpi_real_note}|text|class_name=kpi-note|>
|>
<|part|class_name=kpi|
<|Без вердикта («Не указан»)|text|class_name=kpi-label|>
<|{kpi_none}|text|class_name=kpi-value|>
<|{kpi_none_note}|text|class_name=kpi-note|>
|>
<|part|class_name=kpi|
<|Медиана взятия в работу|text|class_name=kpi-label|>
<|{kpi_react}|text|class_name=kpi-value|>
<|{kpi_react_note}|text|class_name=kpi-note|>
|>
|>

<|part|class_name=card summary|
<|Главное за период|text|class_name=card-title|>
<|{ins_summary}|text|mode=md|class_name=insight summary-list|>
|>

<|part|class_name=card|
<|Сколько карточек поступает каждый день|text|class_name=card-title|>
<|{ins_daily}|text|mode=md|class_name=insight|>
<|chart|figure={fig_daily}|plot_config={PLOT_CONFIG}|height=340px|>
<|Серая линия — карточки за день, синяя — среднее за 7 дней: сглаживает недельный цикл и показывает тренд.|text|class_name=hint|>
|>

<|layout|columns=3 2|gap=16px|
<|part|class_name=card|
<|Как меняется исход карточек|text|class_name=card-title|>
<|{ins_mix}|text|mode=md|class_name=insight|>
<|chart|figure={fig_mix}|plot_config={PLOT_CONFIG}|height=360px|>
<|Каждый столбец — 100 % карточек интервала. Сравнивайте высоту оранжевого и синего сегментов.|text|class_name=hint|>
|>
<|part|class_name=card|
<|Состав вердиктов|text|class_name=card-title|>
<|{ins_verdicts}|text|mode=md|class_name=insight|>
<|chart|figure={fig_verdicts}|plot_config={PLOT_CONFIG}|height=360px|>
<|Доля от всех карточек за период и число карточек.|text|class_name=hint|>
|>
|>
"""

compare_page = """
<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|Какие правила SIEM дают больше всего карточек|text|class_name=card-title|>
<|{ins_top}|text|mode=md|class_name=insight|>
<|chart|figure={fig_top_rules}|plot_config={PLOT_CONFIG}|height=380px|>
<|Число карточек по правилу корреляции. Наведите курсор, чтобы увидеть полное имя правила.|text|class_name=hint|>
|>
<|part|class_name=card|
<|Где «шума» больше среднего|text|class_name=card-title|>
<|{ins_dev}|text|mode=md|class_name=insight|>
<|chart|figure={fig_fp_dev}|plot_config={PLOT_CONFIG}|height=380px|>
<|На сколько процентных пунктов доля False Positive в категории выше (красный) или ниже (синий) средней. Категории с менее чем 100 карточками скрыты.|text|class_name=hint|>
|>
|>

<|part|class_name=card|
<|Когда поступают алерты: день недели × час|text|class_name=card-title|>
<|{ins_heat}|text|mode=md|class_name=insight|>
<|chart|figure={fig_heat}|plot_config={PLOT_CONFIG}|height=300px|>
<|Цвет — среднее число карточек в этот час такого дня недели (время московское).|text|class_name=hint|>
|>

<|part|class_name=card|
<|Топ-15 правил: объём, исход и скорость реакции|text|class_name=card-title|>
<|Точные значения по самым массовым правилам: сортировка по столбцам и выгрузка в CSV.|text|class_name=insight|>
<|{rules_tbl}|table|show_all|downloadable|class_name=rules-table|>
|>
"""

eda_page = """
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Сколько ждут карточки до взятия в работу|text|class_name=card-title|>

<|{hist_crit}|selector|lov={HIST_LOV}|dropdown|label=Критичность|on_change=on_hist|>
|>
<|{ins_hist}|text|mode=md|class_name=insight|>
<|chart|figure={fig_hist}|plot_config={PLOT_CONFIG}|height=360px|>
<|Шкала логарифмическая: каждое деление — в 10 раз больше времени. Вертикальные линии — медиана и 90-й процентиль.|text|class_name=hint|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Подходит ли «правило трёх сигм» для порогов SIEM|text|class_name=card-title|>

<|{qq_log}|toggle|label=ln(1 + x)|on_change=on_qq|>
|>
<|{ins_qq}|text|mode=md|class_name=insight|>
<|chart|figure={fig_qq}|plot_config={PLOT_CONFIG}|height=340px|>
<|Если точки лежат на прямой, число карточек в час распределено нормально. Отрыв вверх справа — всплески сильнее ожидаемых.|text|class_name=hint|>
|>
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Чем реальные инциденты отличаются от False Positive|text|class_name=card-title|>

<|{kde_feature}|selector|lov={KDE_LOV}|dropdown|label=Признак|on_change=on_kde|>
|>
<|{ins_kde}|text|mode=md|class_name=insight|>
<|chart|figure={fig_kde}|plot_config={PLOT_CONFIG}|height=340px|>
<|Площадь под каждой кривой равна 1: сравнивается форма распределений, а не число карточек.|text|class_name=hint|>
|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Время взятия в работу по группам|text|class_name=card-title|>

<|{violin_by}|selector|lov={VIOLIN_LOV}|dropdown|label=Группировка|on_change=on_violin|>
|>
<|{ins_violin}|text|mode=md|class_name=insight|>
<|chart|figure={fig_violin}|plot_config={PLOT_CONFIG}|height=380px|>
<|Ширина контура — сколько карточек с таким временем; шкала логарифмическая.|text|class_name=hint|>
|>
<|part|class_name=card|
<|Время решения: L1, L2 и ГИБ|text|class_name=card-title|>
<|{ins_rain}|text|mode=md|class_name=insight|>
<|chart|figure={fig_rain}|plot_config={PLOT_CONFIG}|height=380px|>
<|Облако — плотность, ящик — медиана и квартили, точки — до 400 случайных карточек группы.|text|class_name=hint|>
|>
|>
"""

flows_page = """
<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Куда уходят карточки: категория → вердикт|text|class_name=card-title|>

<|{sankey_left}|selector|lov={SANKEY_LOV}|dropdown|label=Слева|on_change=on_sankey|>
|>
<|{ins_sankey}|text|mode=md|class_name=insight|>
<|chart|figure={fig_sankey}|plot_config={PLOT_CONFIG}|height=460px|>
<|Слева — категория, справа — вердикт. Наведите на ленту, чтобы увидеть точное число карточек.|text|class_name=hint|>
|>
<|part|class_name=card|
<|За счёт чего изменился поток карточек|text|class_name=card-title|>
<|{ins_wf}|text|mode=md|class_name=insight|>
<|chart|figure={fig_wf}|plot_config={PLOT_CONFIG}|height=460px|>
<|Серые столбцы — итог окна, красные — рост категории, синие — снижение. Окно — выбранный период против такого же предыдущего.|text|class_name=hint|>
|>
|>

<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Плотность вместо диаграммы рассеяния|text|class_name=card-title|>

<|{hex_mode}|selector|lov={HEX_LOV}|dropdown|label=Оси|on_change=on_hex|>
|>
<|{ins_hex}|text|mode=md|class_name=insight|>
<|chart|figure={fig_hex}|plot_config={PLOT_CONFIG}|height=460px|>
<|Цвет соты — сколько точек в неё попало.|text|class_name=hint|>
|>

<|layout|columns=2 3|gap=16px|
<|part|class_name=card|
<|Топ-20 адресов-источников (src)|text|class_name=card-title|>
<|{ins_dot}|text|mode=md|class_name=insight|>
<|chart|figure={fig_dot}|plot_config={PLOT_CONFIG}|height=520px|>
<|Число карточек, в которых встречается адрес-источник.|text|class_name=hint|>
|>
<|part|class_name=card|
<|Правила: скорость решения и доля реальных инцидентов|text|class_name=card-title|>
<|{ins_bubble}|text|mode=md|class_name=insight|>
<|chart|figure={fig_bubble}|plot_config={PLOT_CONFIG}|height=480px|>
<|X — медиана времени решения, Y — доля карточек «Решён», размер — число карточек. Средняя критичность правила — в подсказке.|text|class_name=hint|>
|>
|>
"""

about_page = """
<|part|class_name=card prose|
### Как читать дашборд

Дашборд описывает **операционную работу SOC** по карточкам инцидентов `iavd_incidents_raw`. Все графики, кроме топа адресов-источников, пересчитываются по фильтрам в шапке: **период**, **тип дня** (рабочие / выходные и праздники), **источник** и **критичность**. Под каждым графиком — вывод с цифрами для текущих фильтров.

**Исход карточки** — вердикт аналитика: «Решён» — реальный инцидент, «False Positive» — ложное срабатывание, «Не указан», «Дубликат» и редкие вердикты — прочее. Цвета исходов одинаковы на всех страницах: **оранжевый** — реальный инцидент, **синий** — False Positive, **серый** — прочее.

### Оговорки

- **Доля False Positive — это решение аналитика по карточке**, а не доля трудозатрат SOC: короткая FP-карточка обходится дешевле расследования.
- **Время** («Период взятия в работу», «Время решения») хранится в **миллисекундах** — это проверено по разнице дат «Создан» и «Дата взятия в работу». Отрицательные значения и значения длиннее периода наблюдения считаются ошибками и исключены.
- **Около 28 % карточек берутся в работу «мгновенно»** (< 0,6 с) — их закрывает автоматика. На логарифмической шкале ноль не показать, поэтому они исключены из гистограмм и «скрипок», а их доля указана в выводах.
- **Около 23 % карточек без вердикта.** Для периодов, где таких карточек много, доли FP и реальных инцидентов занижены.
- Время суток и дни недели — по Москве (Europe/Moscow).
|>
"""

pages = {"/": root_page, "overview": overview_page, "compare": compare_page, "eda": eda_page,
         "flows": flows_page, "about": about_page}

stylekit = {
    "color_primary": "#2a78d6", "color_secondary": "#52514e",
    "color_background_light": "#f5f5f2", "color_paper_light": "#ffffff",
    "font_family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "border_radius": 10, "input_button_height": "40px", "root_margin": "0",
}

# Gui создаётся на уровне модуля: Taipy берёт переменные страниц из кадра, где вызван Gui()
gui = Gui(pages=pages, css_file=str(dd.BASE / "dashboard.css"))

if __name__ == "__main__":
    gui.run(title="SOC · триаж алертов SIEM", dark_mode=False, stylekit=stylekit, port=5002,
            watermark="", use_reloader=False)
