"""
dashboard.py — дашборд Taipy «SOC: триаж алертов SIEM» (EDA и BI, без ML-части).

Запуск из папки проекта (после etl.py):
    python dashboard.py
и открыть http://localhost:5002

Страницы построены по «Рекомендациям по выбору диаграмм v1.0»:
    Обзор         — §2: изменение во времени, изменение состава, состав вердиктов (доли)
    Сравнение     — §2: ранжирование, отклонение от нормы, тепловая карта (большой объём данных)
    Распределения — §4: гистограмма, QQ, KDE, violin, raincloud
    Потоки        — §2/§6: Sankey, Waterfall, Hexbin, Dot Plot, Bubble Chart
    О данных      — как читать дашборд, ограничения и соответствие методичке
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from taipy.gui import Gui, navigate

import dashboard_charts as ch
import dashboard_data as dd
from dashboard_charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страниц)
from dashboard_data import CRIT_ORDER, DAY_TYPES, PERIODS, SOURCES  # noqa: F401

STORE = dd.load()

# ---------------------------------------------------------------- состояние (значения по умолчанию)
period, day_type, source, crit_sel = PERIODS[0], DAY_TYPES[0], dd.ALL, list(CRIT_ORDER)
HIST_LOV = [dd.ALL] + CRIT_ORDER
hist_crit = "Критическая"                       # методичка §4.1: «время реакции на критические инциденты»
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


def view_overview(d: pd.DataFrame) -> dict:
    out = dict(fig_daily=ch.fig_daily(d), fig_mix=ch.fig_class_mix(d), fig_verdicts=ch.fig_verdicts(d))
    if d.empty:
        out.update(ins_daily="", ins_mix="", ins_verdicts="")
        return out
    s = d.groupby("date").size()
    out["ins_daily"] = (f"В среднем **{dd.fmt_int(s.mean())}** карточек в день; пик — "
                        f"**{dd.fmt_int(s.max())}** ({dd.fmt_date(s.idxmax())}).")
    fp_share = (d["cls"] == dd.CLS_FP).mean()
    out["ins_mix"] = ("Как меняется исход карточек: доля False Positive, реальных инцидентов и карточек без вердикта "
                      "по интервалам. Интервалы с < 30 карточками скрыты.")
    out["ins_verdicts"] = (f"**{dd.fmt_pct(fp_share)}** карточек закрыты как False Positive. Это операционный вердикт "
                           "аналитика, а не FPR модели и не доля трудозатрат SOC.")
    return out


def view_compare(d: pd.DataFrame) -> dict:
    out = dict(fig_top_rules=ch.fig_top_rules(d), fig_fp_dev=ch.fig_fp_deviation(d),
               fig_heat=ch.fig_heatmap_hour(d), rules_tbl=ch.rules_table(d))
    if d.empty:
        out.update(ins_top="", ins_dev="", ins_heat="")
        return out
    vc = d["rule"].value_counts()
    out["ins_top"] = (f"10 правил из {dd.fmt_int(len(vc))} дают **{dd.fmt_pct(vc.head(10).sum() / len(d))}** всех "
                      f"карточек — с них и стоит начинать настройку SIEM.")
    g = d.groupby("irp")["cls"].agg(["size", lambda s: (s == dd.CLS_FP).mean()])
    g.columns = ["n", "fp"]
    g = g[g["n"] >= 100]
    if len(g) >= 2:
        avg = (d["cls"] == dd.CLS_FP).mean()
        hi, lo = g["fp"].idxmax(), g["fp"].idxmin()
        out["ins_dev"] = (f"Больше всего «шума» — **{hi}** ({dd.fmt_pct(g.loc[hi, 'fp'])} FP при среднем "
                          f"{dd.fmt_pct(avg)}); меньше всего — **{lo}** ({dd.fmt_pct(g.loc[lo, 'fp'])}).")
    else:
        out["ins_dev"] = "Недостаточно категорий для сравнения."
    t = pd.crosstab(d["dow"], d["hour"])
    dow, hour = np.unravel_index(np.argmax(t.values), t.shape)
    night = d["hour"].between(0, 5).mean()
    out["ins_heat"] = (f"Пик нагрузки — **{dd.WEEKDAYS[t.index[dow]]}, {t.columns[hour]:02d}:00**; "
                       f"ночью (00–06) поступает {dd.fmt_pct(night)} карточек — SOC нужен круглосуточно.")
    return out


def view_hist(d: pd.DataFrame, crit: str) -> dict:
    sub = d if crit == dd.ALL else d[d["crit"] == crit]
    react = sub["react_min"].dropna()
    fig = ch.fig_histogram(sub)
    if len(react) < 10:
        return dict(fig_hist=fig, ins_hist="Слишком мало карточек с заполненным временем взятия в работу — "
                                           "проверьте фильтр «Критичность» в шапке.")
    lv = dd.log_minutes(react)
    tail = (react > 24 * 60).mean()
    ins = (f"{dd.fmt_int(len(react))} карточек. Медиана — **{dd.fmt_minutes(react.median())}**, "
           f"но {dd.fmt_pct(tail)} ждут дольше суток — длинный правый хвост. "
           f"**{dd.fmt_pct(dd.instant_share(react))}** взяты «мгновенно» (< 0,6 с, автоматика) — "
           f"на логарифмической шкале их не показать, они исключены.")
    return dict(fig_hist=fig if len(lv) else ch.empty(), ins_hist=ins)


def view_qq(d: pd.DataFrame, log: bool) -> dict:
    fig, r2, n = ch.fig_qq(d, log)
    if not np.isfinite(r2):
        return dict(fig_qq=fig, ins_qq="")
    x = d.set_index("created").resample("h").size().to_numpy(float)
    mu, sd = x.mean(), x.std()
    above = (x > mu + 3 * sd).mean()
    word = "близко к прямой" if r2 >= 0.97 else "заметно отклоняется от прямой"
    ins = (f"{dd.fmt_int(n)} часовых интервалов, R² = **{dd.fmt_num(r2, 3)}** — распределение {word}. "
           f"Часов выше порога «среднее + 3σ»: **{dd.fmt_pct(above, 2)}** при ожидаемых 0,13 % для нормального "
           f"закона, поэтому статический порог по «3 сигмам» будет срабатывать чаще расчётного.")
    return dict(fig_qq=fig, ins_qq=ins)


def view_kde(d: pd.DataFrame, feature: str) -> dict:
    fig, ns = ch.fig_kde(d, feature)
    col, is_log = ch.KDE_FEATURES[feature]
    meds, zeros = [], []
    for cls in [dd.CLS_REAL, dd.CLS_FP]:
        v = pd.to_numeric(d.loc[d["cls"] == cls, col], errors="coerce").dropna()
        name = cls.split(" (")[0]
        if is_log and len(v):
            zeros.append(f"{name} — {dd.fmt_pct((v <= 0).mean())}")
            v = v[v > 0]                           # медиана — по тем же значениям, что и на графике
        if len(v):
            m = v.median()
            val = (dd.fmt_minutes(m) if col.endswith("_min") else f"{dd.fmt_int(m)} символов" if col == "cmd_len"
                   else f"{dd.fmt_num(m, 2)} бит/символ")
            meds.append(f"{name} — **{val}**")
    ins = ("Медианы: " + "; ".join(meds) + ".") if meds else ""
    if is_log and any(not z.endswith(" 0,0\u00a0%") for z in zeros):
        ins += " Нулевые значения на лог-шкале не показать, их доля: " + "; ".join(zeros) + "."
    small = [c for c, n in ns.items() if n < 50]
    if small:
        ins += " В классе «" + "», «".join(c.split(" (")[0] for c in small) + "» меньше 50 значений — кривая не строится."
    return dict(fig_kde=fig, ins_kde=ins)


def view_violin(d: pd.DataFrame, by: str) -> dict:
    return dict(fig_violin=ch.fig_violin(d, by))


def view_rain(d: pd.DataFrame) -> dict:
    meds = []
    for g in dd.GROUP_ORDER:
        v = d.loc[d["group"] == g, "solve_min"].dropna()
        if len(v) >= 5:
            meds.append(f"{g} — **{dd.fmt_minutes(v.median())}**")
    ins = ("Медиана времени решения: " + "; ".join(meds) + ". Эскалированные карточки (L2, ГИБ) живут на порядки "
           "дольше — это сложные расследования, а не медленная работа.") if meds else ""
    return dict(fig_rain=ch.fig_raincloud(d), ins_rain=ins)


def view_flows(store: dd.Store, d: pd.DataFrame, f: dd.Filters, left: str, hexm: str) -> dict:
    out = dict(fig_sankey=ch.fig_sankey(d, left), fig_bubble=ch.fig_bubble(d))
    out.update(view_hex(d, hexm))
    prev, cur, n_days = dd.previous_window(store, f)
    out["fig_wf"] = ch.fig_waterfall(prev, cur, n_days)
    if len(prev) or len(cur):
        delta = len(cur) - len(prev)
        a, b = prev["irp"].value_counts(), cur["irp"].value_counts()
        dlt = b.reindex(a.index.union(b.index), fill_value=0) - a.reindex(a.index.union(b.index), fill_value=0)
        main = dlt.abs().idxmax() if len(dlt) else "—"
        rel = f" ({'+' if delta >= 0 else '−'}{dd.fmt_pct(abs(delta) / len(prev))})" if len(prev) else ""
        scope = "" if f.period in store.period_dates else " Для «Весь период» взято окно «30 дней» из таблицы периодов."
        out["ins_wf"] = (f"Поток {'вырос' if delta >= 0 else 'снизился'} на **{dd.fmt_int(abs(delta))}** карточек{rel} "
                         f"против предыдущих {n_days} дн.; главный вклад — **{main}** "
                         f"({'+' if dlt[main] >= 0 else '−'}{dd.fmt_int(abs(dlt[main]))}).{scope}")
    else:
        out["ins_wf"] = ""
    if not d.empty:
        t = pd.crosstab(d[ch.SANKEY_LEFT[left]].astype(str), d["cls"])
        t = t[t.sum(axis=1) >= max(30, 0.01 * len(d))]
        if len(t):
            share = t.get(dd.CLS_FP, 0) / t.sum(axis=1)
            k = share.idxmax()
            out["ins_sankey"] = (f"Ширина ленты = число карточек. Самая «шумная» категория — **{ch._short(k, 50)}**: "
                                 f"{dd.fmt_pct(share[k])} её карточек закрыты как False Positive.")
        else:
            out["ins_sankey"] = ""
    else:
        out["ins_sankey"] = ""
    return out


def view_hex(d: pd.DataFrame, mode: str) -> dict:
    fig, n = ch.fig_hexbin(d, mode)
    if mode == ch.HEX_MODES[0]:
        ins = (f"{dd.fmt_int(n)} карточек: на диаграмме рассеяния они слились бы в сплошное пятно, соты показывают, "
               "где сосредоточена масса. Диагональ — карточки, решённые сразу после взятия в работу.")
    else:
        ins = (f"{dd.fmt_int(n)} часовых интервалов (точка = один час): видно, в какие часы бывают всплески "
               "потока, а не только среднюю нагрузку. Час смещён случайно в пределах ±0,5 для раскладки по сотам.")
    return dict(fig_hex=fig, ins_hex=ins)


def compute_all(store: dd.Store, f: dd.Filters, ctl: dict) -> dict:
    d = dd.apply_filters(store, f)
    out = {}
    out.update(view_header(d, f))
    out.update(view_overview(d))
    out.update(view_compare(d))
    out.update(view_hist(d, ctl["hist_crit"]))
    out.update(view_qq(d, ctl["qq_log"]))
    out.update(view_kde(d, ctl["kde_feature"]))
    out.update(view_violin(d, ctl["violin_by"]))
    out.update(view_rain(d))
    out.update(view_flows(store, d, f, ctl["sankey_left"], ctl["hex_mode"]))
    return out


_ctl0 = dict(hist_crit=hist_crit, qq_log=qq_log, kde_feature=kde_feature, violin_by=violin_by,
             sankey_left=sankey_left, hex_mode=hex_mode)
_init = compute_all(STORE, dd.Filters(), _ctl0)
globals().update(_init)                            # начальные значения всех фигур и подписей для Taipy
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
    _assign(state, view_qq(_d(state), state.qq_log))


def on_kde(state, var_name, value):
    _assign(state, view_kde(_d(state), state.kde_feature))


def on_violin(state, var_name, value):
    _assign(state, view_violin(_d(state), state.violin_by))


def on_sankey(state, var_name, value):
    _assign(state, view_flows(STORE, _d(state), _filters(state), state.sankey_left, state.hex_mode))


on_hex = on_sankey


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

<|part|class_name=card|
<|Сколько карточек поступает каждый день|text|class_name=card-title|>
<|{ins_daily}|text|mode=md|class_name=insight|>
<|chart|figure={fig_daily}|plot_config={PLOT_CONFIG}|height=340px|>
<|§2 Сравнение → изменение во времени: линейный график. Серая линия — дни, синяя — сглаживание за 7 дней.|text|class_name=method|>
|>

<|layout|columns=3 2|gap=16px|
<|part|class_name=card|
<|Как меняется исход карточек|text|class_name=card-title|>
<|{ins_mix}|text|mode=md|class_name=insight|>
<|chart|figure={fig_mix}|plot_config={PLOT_CONFIG}|height=340px|>
<|§2 Состав → изменение состава во времени: 100-процентная столбчатая с накоплением.|text|class_name=method|>
|>
<|part|class_name=card|
<|Состав вердиктов|text|class_name=card-title|>
<|{ins_verdicts}|text|mode=md|class_name=insight|>
<|chart|figure={fig_verdicts}|plot_config={PLOT_CONFIG}|height=340px|>
<|§2 Состав → простые доли: столбцы долей (круговая не подходит — 7 категорий, есть очень малые).|text|class_name=method|>
|>
|>
"""

compare_page = """
<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|Какие правила SIEM дают больше всего карточек|text|class_name=card-title|>
<|{ins_top}|text|mode=md|class_name=insight|>
<|chart|figure={fig_top_rules}|plot_config={PLOT_CONFIG}|height=380px|>
<|§2 Сравнение → ранжирование: линейчатая диаграмма, один цвет — сравниваем длину.|text|class_name=method|>
|>
<|part|class_name=card|
<|Где «шума» больше среднего|text|class_name=card-title|>
<|{ins_dev}|text|mode=md|class_name=insight|>
<|chart|figure={fig_fp_dev}|plot_config={PLOT_CONFIG}|height=380px|>
<|§2 Сравнение → отклонение от нормы: дивергентная столбчатая. Красный — FP чаще среднего, синий — реже. Категории с < 100 карточками скрыты.|text|class_name=method|>
|>
|>

<|part|class_name=card|
<|Когда поступают алерты: день недели × час|text|class_name=card-title|>
<|{ins_heat}|text|mode=md|class_name=insight|>
<|chart|figure={fig_heat}|plot_config={PLOT_CONFIG}|height=300px|>
<|§2 Связь → большой объём данных: тепловая карта. Цвет — среднее число карточек в этот час такого дня недели.|text|class_name=method|>
|>

<|part|class_name=card|
<|Топ-15 правил: объём, исход и скорость реакции|text|class_name=card-title|>
<|Табличное представление тех же данных: точные значения, сортировка по столбцам и выгрузка в CSV.|text|class_name=insight|>
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
<|§4.1 Гистограмма: форма распределения, асимметрия и хвосты. Не подходит, когда нужна гладкая плотность. Бины равной ширины на логарифмической шкале.|text|class_name=method|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Подходит ли «правило трёх сигм» для порогов SIEM|text|class_name=card-title|>

<|{qq_log}|toggle|label=ln(1 + x)|on_change=on_qq|>
|>
<|{ins_qq}|text|mode=md|class_name=insight|>
<|chart|figure={fig_qq}|plot_config={PLOT_CONFIG}|height=340px|>
<|§4.2 QQ Plot: близость к эталонному (нормальному) закону. Не для сравнения категорий и поиска корреляций.|text|class_name=method|>
|>
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Чем реальные инциденты отличаются от False Positive|text|class_name=card-title|>

<|{kde_feature}|selector|lov={KDE_LOV}|dropdown|label=Признак|on_change=on_kde|>
|>
<|{ins_kde}|text|mode=md|class_name=insight|>
<|chart|figure={fig_kde}|plot_config={PLOT_CONFIG}|height=340px|>
<|§4.3 KDE: сглаженная плотность (площадь под каждой кривой = 1, сравниваем форму, а не объём). Не для малых выборок и точных частот.|text|class_name=method|>
|>
|>

<|layout|columns=1 1|gap=16px|
<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Время взятия в работу по группам|text|class_name=card-title|>

<|{violin_by}|selector|lov={VIOLIN_LOV}|dropdown|label=Группировка|on_change=on_violin|>
|>
<|Контур — плотность распределения, белый ящик — медиана и квартили. Группы с < 30 значениями скрыты.|text|class_name=insight|>
<|chart|figure={fig_violin}|plot_config={PLOT_CONFIG}|height=380px|>
<|§4.4 Violin = KDE + box plot: сравнение распределений по группам. При очень малых группах лучше Dot или Strip Plot.|text|class_name=method|>
|>
<|part|class_name=card|
<|Время решения: L1, L2 и ГИБ|text|class_name=card-title|>
<|{ins_rain}|text|mode=md|class_name=insight|>
<|chart|figure={fig_rain}|plot_config={PLOT_CONFIG}|height=380px|>
<|§4.5 Raincloud: облако (KDE) + зонтик (box plot) + дождь (до 400 случайных карточек на группу). Сложнее для бизнес-аудитории.|text|class_name=method|>
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
<|chart|figure={fig_sankey}|plot_config={PLOT_CONFIG}|height=440px|>
<|§6.2 Sankey: объёмы связей между категориями (статический срез, время не нужно). Не подходит, если известны только доли.|text|class_name=method|>
|>
<|part|class_name=card|
<|За счёт чего изменился поток карточек|text|class_name=card-title|>
<|{ins_wf}|text|mode=md|class_name=insight|>
<|chart|figure={fig_wf}|plot_config={PLOT_CONFIG}|height=400px|>
<|§2 Состав → вклады и изменения: Waterfall. Окно — выбранный период против такого же предыдущего; красный — рост, синий — снижение.|text|class_name=method|>
|>
|>

<|part|class_name=card|
<|layout|columns=1 auto|class_name=card-head|
<|Плотность вместо диаграммы рассеяния|text|class_name=card-title|>

<|{hex_mode}|selector|lov={HEX_LOV}|dropdown|label=Оси|on_change=on_hex|>
|>
<|{ins_hex}|text|mode=md|class_name=insight|>
<|chart|figure={fig_hex}|plot_config={PLOT_CONFIG}|height=440px|>
<|§6.1 Hexbin: цвет соты — число точек в ней. Не подходит, когда точек меньше 100 и важно видеть каждую.|text|class_name=method|>
|>

<|layout|columns=2 3|gap=16px|
<|part|class_name=card|
<|Топ-20 адресов-источников (src)|text|class_name=card-title|>
<|Агрегат из etl.py за весь период: фильтры дашборда к нему не применяются.|text|class_name=insight|>
<|chart|figure={fig_dot}|plot_config={PLOT_CONFIG}|height=520px|>
<|§6.3 Dot Plot: сравнение категорий без тяжёлых заливок. Не подходит, если важна именно длина столбца.|text|class_name=method|>
|>
<|part|class_name=card|
<|Правила: скорость решения и доля реальных инцидентов|text|class_name=card-title|>
<|Пузырёк — правило (топ-20 по объёму); размер — число карточек. Справа вверху — долгие и часто настоящие, внизу слева — быстрый «шум».|text|class_name=insight|>
<|chart|figure={fig_bubble}|plot_config={PLOT_CONFIG}|height=440px|>
<|§6.4 Bubble Chart: три переменные (X, Y, размер). Число правил в карточке и средняя критичность почти не различаются между правилами (не считываются размером), поэтому Y — доля «Решён», размер — объём; критичность — в подсказке.|text|class_name=method|>
|>
|>
"""

about_page = """
<|part|class_name=card prose|
### Как читать дашборд

Дашборд описывает **операционную работу SOC** по карточкам инцидентов `iavd_incidents_raw` — без ML-модели. Все графики, кроме Dot Plot, пересчитываются по фильтрам в шапке: **период** (таблица `dataframe.parquet`, как «остров периодов» в треке Qlik), **тип дня** (`calendar.parquet`), **источник** и **критичность**.

**Исход карточки** — вердикт аналитика: «Решён» = реальный инцидент (Fact_label = 1), «False Positive» = ложное срабатывание (Fact_label = 0), «Не указан», «Дубликат» и редкие вердикты — вне разметки (Fact_label = −1). Цвета исходов одинаковы на всех страницах: **оранжевый** — реальный инцидент, **синий** — False Positive, **серый** — прочее.

### Важные оговорки

- **Доля False Positive ≠ FPR модели.** Это доля карточек с операционным вердиктом FP, а не ошибка классификатора и не доля трудозатрат (для неё нужны длительности обработки).
- **Время** («Период взятия в работу», «Время решения») хранится в **миллисекундах** — проверено по разнице дат «Создан» и «Дата взятия в работу». Отрицательные значения и значения длиннее периода наблюдения считаются ошибками и исключены (`etl.py`).
- **Около 28 % карточек берутся в работу «мгновенно»** (< 0,6 с) — их закрывает автоматика. На логарифмической шкале ноль не показать, поэтому они исключены из гистограмм и скрипок, а их доля всегда подписана.
- Часовой пояс — Europe/Moscow.

### Соответствие методичке «Рекомендации по выбору диаграмм v1.0»

| Раздел | Вопрос к данным | Диаграмма | Страница |
|---|---|---|---|
| §2 Сравнение | Изменение во времени | Линейный график | Обзор |
| §2 Состав | Изменение состава во времени | 100-процентная столбчатая с накоплением | Обзор |
| §2 Состав | Простые доли (состав вердиктов) | Столбцы долей | Обзор |
| §2 Сравнение | Ранжирование | Линейчатая диаграмма | Сравнение |
| §2 Сравнение | Отклонение от нормы | Дивергентная столбчатая | Сравнение |
| §2 Связь | Большой объём данных | Тепловая карта | Сравнение |
| §4.1 | Форма распределения, хвосты | Гистограмма | Распределения |
| §4.2 | Близость к нормальному закону | QQ Plot | Распределения |
| §4.3 | Сглаженная плотность | KDE Plot | Распределения |
| §4.4 | Распределения по группам | Violin Plot | Распределения |
| §4.5 | Форма + статистика + сырые точки | Raincloud Plot | Распределения |
| §6.2 / §2 | Объёмы связей категорий | Sankey | Потоки и плотность |
| §2 Состав | Вклады в изменение итога | Waterfall | Потоки и плотность |
| §6.1 | Тысячи точек вместо scatter | Hexbin | Потоки и плотность |
| §6.3 | Компактное сравнение категорий | Dot Plot | Потоки и плотность |
| §6.4 | Три переменные сразу | Bubble Chart | Потоки и плотность |

Раздел §5 (ROC/PR, SHAP, PDP, Bias–Variance) относится к ML-модели и в этот дашборд намеренно не входит.

### Оформление (чек-лист 2.2)

Без 3D, теней и декоративных рамок; сетка бледная и только по оси значений; один фокусный цвет и нейтральная серая база; прямые подписи значений там, где серий мало; у каждой оси — подпись и единицы. Палитры проверены на различимость при дальтонизме.
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
