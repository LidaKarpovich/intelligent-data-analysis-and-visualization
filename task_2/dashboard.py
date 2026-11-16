"""
dashboard.py — дашборд Taipy «Можно ли доверять модели»: как модель-советник, которая сортирует
карточки SOC по вероятности инцидента, отработала в эксплуатации.

Запуск из папки task_2 (после prepare.py):
    python dashboard.py
и открыть http://localhost:5003

Пять страниц — пять вопросов руководителя SOC:
    1 · Итоги          — что модель дала за период эксплуатации?
    2 · Согласие       — как часто модель согласна с вердиктом аналитиков?
    3 · Пропуски       — сколько реальных инцидентов модель пропускает и какие?
    4 · Зоны ошибок    — ошибается случайно или систематически? где именно?
    5 · Можно ли верить — честна ли оценка модели и не устарела ли она?
На листах — только KPI-плитки и графики: важные числа упакованы в плитки, а выводы и ответ
«почему выбран именно этот график» (WHY в dashboard_charts.py) вынесены в REPORT.md (report.py).
"""
from __future__ import annotations

import pandas as pd
from taipy.gui import Gui, navigate

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_kpis as kp
from dashboard_charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страниц)
from dashboard_data import CRIT_ORDER  # noqa: F401

STORE = dd.load()
TARGET_MISS = dd.target_miss(STORE)          # доля пропусков на приёмке — уровень, с которым модель сдана

# ---------------------------------------------------------------- состояние (значения по умолчанию)
PERIODS = dd.period_options(STORE)
ORGS = dd.org_options(STORE)
POLICIES = list(STORE.policies)
period, org, crit_sel, policy = dd.PERIOD_ALL, dd.ALL, list(CRIT_ORDER), dd.default_policy(STORE)
SEG_LOV = list(dd.SEGMENTS)
seg_by = SEG_LOV[0]
METRIC_LOV = list(dd.METRICS)
seg_metric = METRIC_LOV[0]

NAV = [("/results", "1 · Итоги"), ("/agreement", "2 · Согласие"), ("/misses", "3 · Пропуски"),
       ("/errors", "4 · Зоны ошибок"), ("/trust", "5 · Можно ли верить")]


def _filters(s) -> dd.Filters:
    return dd.Filters(period=s.period, org=s.org, crit=list(s.crit_sel or []), policy=s.policy)


# ================================================================ расчёт представлений
def caption(store: dd.Store, f: dd.Filters, oper: pd.DataFrame) -> str:
    p = store.meta["periods"]
    span = (f"эксплуатация {dd.fmt_date(p['oper'][0])} — {dd.fmt_date(p['oper'][1])}" if f.period == dd.PERIOD_ALL
            else f"эксплуатация: {f.period.lower()}")
    n_lab = int((oper["y"] >= 0).sum())
    # метрики, каскад и разгрузка очереди считаются только по карточкам с вердиктом — показываем обе базы
    acc = f"приёмка: весь тест-сплит окна {dd.fmt_date(p['train'][0])} — {dd.fmt_date(p['train'][1])} без фильтров"
    if not dd.is_default_policy(store, f):
        acc += f", пересчитан при пороге {dd.fmt_num(dd.threshold_of(store, f), 4)} (не карточка модели)"
    return (f"{span} · {acc} "
            f"· порог {dd.fmt_num(dd.threshold_of(store, f), 4)} · {dd.fmt_int(len(oper))} "
            f"{dd.plural(len(oper), ('карточка', 'карточки', 'карточек'))}: с вердиктом {dd.fmt_int(n_lab)}, "
            f"без вердикта {dd.fmt_int(len(oper) - n_lab)} — вне метрик")


def view_segments(oper: pd.DataFrame, by: str, metric_label: str) -> dict:
    metric = dd.METRICS[metric_label]
    return dict(fig_seg=ch.fig_segments(dd.segment_table(oper, by, metric), metric))


def dumbbell_rows(cv: dd.Confusion, c: dd.Confusion) -> list:
    """Показатели, которые можно сравнивать между выборками с разной долей инцидентов (методичка, 5.1):
    recall и согласие на FP от доли инцидентов не зависят; precision зависит — её здесь нет.
    Доля инцидентов в потоке — контекст: показывает, насколько выборки различаются."""
    nan = float("nan")
    return [("Найдено реальных инцидентов (recall)", cv.recall, c.recall),
            ("Согласие с аналитиком на FP", cv.specificity, c.specificity),
            ("Очередь разгружена (низкий приоритет)", cv.low_share, c.low_share),
            ("Доля инцидентов в потоке", cv.pos / cv.n if cv.n else nan, c.pos / c.n if c.n else nan)]


def compute_all(store: dd.Store, f: dd.Filters, ctl: dict) -> dict:
    thr = dd.threshold_of(store, f)
    oper = dd.apply_filters(store, f)            # эксплуатация: фильтры шапки (месяц, организация, критичность)
    val = dd.acceptance(store, f)                # приёмка: всегда весь тест-сплит, фильтры шапки не действуют
    acc = dd.acceptance_label(store, f)          # «на приёмке» или «на приёмке при пороге …» для «что если»
    mon = dd.timeline(store, f)
    mon_oper = mon[mon["part"] == dd.OPER] if len(mon) else mon
    sel = dd.selected_month(store, f)
    c, cv = dd.confusion(oper), dd.confusion(val)
    tm = cv.miss if cv.pos else TARGET_MISS        # уровень приёмки при выбранном пороге

    out = dict(period_caption=caption(store, f, oper))
    # 1 · Итоги
    out |= dict(fig_waterfall=ch.fig_queue_waterfall(c), fig_gains=ch.fig_gains(val, oper, thr))
    out |= kp.results(oper, c, cv, tm, val, acc=acc)
    # 2 · Согласие
    out |= dict(fig_sankey=ch.fig_sankey(c), fig_agree=ch.fig_agreement_monthly(mon, sel))
    out |= kp.agreement(oper, c, cv, acc=acc)
    # 3 · Пропуски
    out |= dict(fig_miss=ch.fig_miss_monthly(mon, tm, sel, ref_label=acc), fig_missed_rules=ch.fig_missed_rules(oper))
    out |= kp.misses(oper, c, mon_oper, thr, tm, val, acc=acc)
    # 4 · Зоны ошибок
    out |= view_segments(oper, ctl["seg_by"], ctl["seg_metric"])
    out |= dict(fig_hist=ch.fig_score_hist(oper, thr))
    out |= kp.errors(oper, c, cv, acc=acc)
    # 5 · Можно ли верить
    val_title = "Приёмка" if dd.is_default_policy(store, f) else f"Приёмка, порог {dd.fmt_num(thr, 4)}"
    out |= dict(fig_rocpr=ch.fig_roc_pr(val, oper, thr, val_title),
                fig_cal=ch.fig_calibration(val, oper), fig_drift=ch.fig_drift(mon, sel),
                fig_dumbbell=ch.fig_dumbbell(dumbbell_rows(cv, c)))
    out |= kp.trust(oper, val, mon_oper, store.meta, acc=acc)
    return out


_ctl0 = dict(seg_by=seg_by, seg_metric=seg_metric)
globals().update(compute_all(STORE, dd.Filters(policy=policy), _ctl0))   # начальные значения для Taipy


# ================================================================ обработчики
def _assign(state, values: dict):
    with state:                                    # одно обновление интерфейса вместо десятков
        for k, v in values.items():
            setattr(state, k, v)


def _ctl(state) -> dict:
    return {k: getattr(state, k) for k in _ctl0}


def on_filter(state, var_name=None, value=None):
    _assign(state, compute_all(STORE, _filters(state), _ctl(state)))


def on_reset(state, *args):
    with state:
        state.period, state.org, state.crit_sel = dd.PERIOD_ALL, dd.ALL, list(CRIT_ORDER)
        state.policy = dd.default_policy(STORE)
    on_filter(state)


def on_segments(state, var_name, value):
    _assign(state, view_segments(dd.apply_filters(STORE, _filters(state)), state.seg_by, state.seg_metric))


def on_init(state):
    navigate(state, "results")


# ================================================================ разметка страниц
def card(title: str, fig: str, height: int, controls: str = "") -> str:
    """Карточка графика: вопрос-заголовок, при необходимости переключатели, сам график.
    Обоснование выбора графика — в REPORT.md, а не на листе."""
    n_ctl = controls.count("selector")
    cols = " ".join(["1"] + ["auto"] * n_ctl)
    head = (f"<|layout|columns={cols}|gap=12px|class_name=card-head|\n<|{title}|text|class_name=card-title|>\n\n{controls}\n|>"
            if controls else f"<|{title}|text|class_name=card-title|>")
    return (f"<|part|class_name=card|\n{head}\n"
            f"<|chart|figure={{{fig}}}|plot_config={{PLOT_CONFIG}}|height={height}px|>\n|>")


def two(a: str, b: str, cols: str = "1 1") -> str:
    return f"<|layout|columns={cols}|gap=16px|\n{a}\n{b}\n|>"


def question(text: str) -> str:
    return f"<|{text}|text|class_name=page-q|>"


root_page = """
<|part|class_name=topbar|
<|layout|columns=1 auto|class_name=topbar-row|
<|part|class_name=brand|
<|SOC · можно ли доверять модели курса lgbm_gate1|text|class_name=brand-title|>
<|{period_caption}|text|class_name=brand-sub|>
|>
<|navbar|lov={NAV}|class_name=nav|>
|>

<|layout|columns=1 1 2 2 auto|gap=12px|class_name=filters|
<|{period}|selector|lov={PERIODS}|dropdown|label=Период эксплуатации|on_change=on_filter|>

<|{org}|selector|lov={ORGS}|dropdown|label=Организация|on_change=on_filter|>

<|{crit_sel}|selector|lov={CRIT_ORDER}|dropdown|multiple|label=Критичность|on_change=on_filter|>

<|{policy}|selector|lov={POLICIES}|dropdown|label=Порог модели: рабочий или «что если»|on_change=on_filter|>

<|Сбросить фильтры|button|on_action=on_reset|class_name=reset|>
|>
|>

<|part|class_name=page|
<|content|>
|>
"""

results_page = "\n\n".join([
    question("Что модель дала SOC за период эксплуатации?"),
    kp.markup("results"),
    two(card("Карточки с вердиктом: сколько модель убрала из очереди и чем за это заплатили", "fig_waterfall", 380),
        card("Сколько очереди надо разобрать, чтобы найти инциденты", "fig_gains", 380)),
])

agreement_page = "\n\n".join([
    question("Как часто модель соглашается с вердиктом аналитиков SOC?"),
    kp.markup("agreement"),
    two(card("Вердикт аналитика → решение модели", "fig_sankey", 400),
        card("Когда модель начала расходиться с аналитиками", "fig_agree", 400)),
])

misses_page = "\n\n".join([
    question("Сколько реальных инцидентов модель пропускает?"),
    kp.markup("misses"),
    two(card("Доля пропущенных инцидентов по месяцам", "fig_miss", 400),
        card("На каких правилах SIEM модель пропускает инциденты", "fig_missed_rules", 400),
        cols="3 2"),
])

errors_page = "\n\n".join([
    question("Модель ошибается случайно или систематически? Где именно?"),
    kp.markup("errors"),
    card("В каких срезах ошибок больше среднего", "fig_seg", 440, controls=(
        "<|{seg_by}|selector|lov={SEG_LOV}|dropdown|label=Срез|on_change=on_segments|>\n\n"
        "<|{seg_metric}|selector|lov={METRIC_LOV}|dropdown|label=Ошибка|on_change=on_segments|>")),
    card("Насколько модель была уверена, когда ошиблась", "fig_hist", 480),
])

trust_page = "\n\n".join([
    question("Можно ли верить оценке модели и не устарела ли она?"),
    kp.markup("trust"),
    card("Насколько модель различает инциденты и FP: ROC и PR с рабочим порогом", "fig_rocpr", 640),
    two(card("Если модель пишет «30 %», сколько на самом деле инцидентов", "fig_cal", 400),
        card("Что изменилось между приёмкой и эксплуатацией", "fig_dumbbell", 400)),
    card("Когда модель начала устаревать", "fig_drift", 480),
])

pages = {"/": root_page, "results": results_page, "agreement": agreement_page, "misses": misses_page,
         "errors": errors_page, "trust": trust_page}

stylekit = {
    "color_primary": "#2a78d6", "color_secondary": "#52514e",
    "color_background_light": "#f6f7f9", "color_paper_light": "#ffffff",
    "font_family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "border_radius": 10, "input_button_height": "40px", "root_margin": "0",
}

# Gui создаётся на уровне модуля: Taipy берёт переменные страниц из кадра, где вызван Gui()
gui = Gui(pages=pages, css_file="dashboard.css")

if __name__ == "__main__":
    gui.run(title="SOC · доверие к модели", dark_mode=False, stylekit=stylekit, port=5003,
            watermark="", use_reloader=False)
