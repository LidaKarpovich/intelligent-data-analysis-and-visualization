"""
dashboard.py — дашборд Taipy для задания 3 (SHAP / XAI · команда team = 5).

Запуск из папки task_3 (после prepare.py):
    python dashboard.py
и открыть http://localhost:5004

Пять страниц — три уровня объяснений (карточка, группа похожих, весь поток), дрейф и проверка кейсов:
    1 · Почему так?       — локальный SHAP-водопад, баланс, Top-N, поля карточки, ручка порога τ
    2 · Похожие карточки  — Hexbin-карта похожести (SHAP / PCA / t-SNE), 20-NN и Violin по группе
    3 · Весь поток        — Beeswarm, глобальная важность + Ablation, ALE / PDP / ICE и group-PDP
    4 · Дрейф             — гантели сдвига |SHAP| + PSI и профили распределений Приёмка -> Прод
    5 · Проверка кейсов   — сверка топ-признаков с 95 % ДИ Уилсона: подтверждённые и сомнительные кейсы
"""
from __future__ import annotations

import numpy as np
from taipy.gui import Gui, navigate

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_kpis as kp
from dashboard_charts import PLOT_CONFIG  # noqa: F401
from feature_registry import CAT_FEATURES_12, FEATURE_ORDER_73, NUMERIC_FEATURES_43, REGISTRY

STORE = dd.load()

# ---------------------------------------------------------------- опции селекторов
NAV = [
    ("/card", "1 · Почему так?"),
    ("/similar", "2 · Похожие карточки"),
    ("/global", "3 · Весь поток"),
    ("/drift", "4 · Дрейф"),
    ("/cases", "5 · Проверка кейсов"),
]

OUTCOME_LOV = ["Все (8)", "FN", "FP", "TP", "TN"]
PRESET_LOV = list(dd.PRESETS.keys())
DETAIL_LOV = ["73 поля SOC", "1 099 фич модели"]
SPACE_LOV = ["SHAP 73D (PCA)", "Признаки (SVD-20)", "SHAP (t-SNE)"]
COLOR_LOV = [
    "Доля инцидентов (факт)",
    "Плотность карточек (классический Hexbin)",
    "Решение модели (приоритет)",
    "Доля ошибок модели",
    "Кластер выбранной карточки",
    "Топ-5 кластеров (дискретно)",
]
GROUP_LOV = ["40 ближайших по SHAP", "Тот же исход (практика 320)", "Весь срез практики (320)"]
SAMPLE_LOV = ["Эксплуатация", "Приёмка"]
DRIFT_SORT_LOV = ["По сдвигу |SHAP|", "По PSI", "По росту пропусков"]
CASE_LOV = list(dd.CURATED_CASES.keys())
VERIF_PER_LOV = ["Эксплуатация", "Окно обучения"]

NUM_LABEL_TO_COL = {f"{REGISTRY[c].label_ru} ({c})": c for c in NUMERIC_FEATURES_43}
NUM_LOV = list(NUM_LABEL_TO_COL.keys())
CAT_LABEL_TO_COL = {f"{REGISTRY[c].label_ru} ({c})": c for c in CAT_FEATURES_12}
CAT_LOV = list(CAT_LABEL_TO_COL.keys())
ALL_LABEL_TO_COL = {f"{REGISTRY[c].label_ru} ({c})": c for c in FEATURE_ORDER_73}
DRIFT_FEAT_LOV = list(ALL_LABEL_TO_COL.keys())

# ---------------------------------------------------------------- состояние по умолчанию
outcome_flt = OUTCOME_LOV[0]
tau_preset = PRESET_LOV[0]
log10_tau = float(np.log10(dd.DEFAULT_TAU))
CARD_LOV = dd.card_options(STORE, outcome_flt, dd.DEFAULT_TAU)
card_sel = CARD_LOV[0]                                # первая FN-карточка (#1 · 523480)
top_n = 10
detail_lvl = DETAIL_LOV[0]

map_space = SPACE_LOV[0]
map_color = COLOR_LOV[0]
group_mode = GROUP_LOV[0]

bw_sample = SAMPLE_LOV[0]
num_feat_lbl = next((k for k, v in NUM_LABEL_TO_COL.items() if v == "create_lag_s"), NUM_LOV[0])
curve_period = SAMPLE_LOV[0]
cat_feat_lbl = next((k for k, v in CAT_LABEL_TO_COL.items() if v == "Вендор из сием"), CAT_LOV[0])

drift_sort = DRIFT_SORT_LOV[0]
drift_feat_lbl = next((k for k, v in ALL_LABEL_TO_COL.items() if v == "Вендор из сием"), DRIFT_FEAT_LOV[0])

case_sel = CASE_LOV[1]                                # сомнительный FP #4 · 526930 по умолчанию для наглядности
verif_period = VERIF_PER_LOV[0]
case_summary_text = ""


# ---------------------------------------------------------------- расчёт представлений
def current_tau(preset: str, l10: float) -> float:
    p_val = dd.PRESETS.get(preset)
    if p_val is not None:
        return float(p_val)
    return float(np.clip(10.0 ** float(l10), 1e-5, 0.99))


def header_caption(store: dd.Store, tau: float) -> str:
    p = store.meta["counts"]
    cm_p = dd.confusion_from_proba(store.practice_320["y_true"].to_numpy(), store.practice_320["proba"].to_numpy(), tau)
    cm_t5 = dd.confusion_from_proba(store.team5_cards["y_true"].to_numpy(), store.team5_cards["proba_calc"].to_numpy(), tau)
    flag = "" if abs(tau - dd.DEFAULT_TAU) <= 1e-6 else " (НЕ эталон курса)"
    return (
        f"команда team = {store.meta['team']} (8 карточек) · модель lgbm_gate1__v001 (73 поля / 1 099 фич) · "
        f"τ = {dd.fmt_num(tau, 4)}{flag} · команда 5: TP {cm_t5['tp']}/FP {cm_t5['fp']}/FN {cm_t5['fn']}/TN {cm_t5['tn']} · "
        f"практика ({p['practice_total']}): TP {cm_p['tp']}/FP {cm_p['fp']}/FN {cm_p['fn']}/TN {cm_p['tn']}"
    )


def view_card(store: dd.Store, eid: int, tau: float, n_top: int, detail: str) -> dict:
    slc = dd.slice_top_n(store, eid, n_top, detail)
    return (
        kp.page_card(store, eid, tau, slc)
        | dict(
            fig_card_wf=ch.fig_card_waterfall(slc, tau),
            fig_card_bal=ch.fig_card_balance(slc),
            tbl_card_fields=dd.build_card_table(store, eid, n_top, only_top=False),
        )
    )


def view_similar(store: dd.Store, eid: int, space: str, color: str, grp_mode: str, tau: float) -> dict:
    grp = dd.similar_group_shap(store, eid, grp_mode)
    return (
        kp.page_similar(store, eid, space, tau)
        | dict(
            fig_hexmap=ch.fig_sim_hexmap(store, eid, space, color, tau, show_neighbors=True),
            fig_violin=ch.fig_group_violin(grp, eid),
            tbl_neighbors=dd.neighbors_table(store, eid, space, tau),
        )
    )


def view_global(store: dd.Store, sample: str, n_lbl: str, c_per: str, c_lbl: str, eid: int) -> dict:
    nf = NUM_LABEL_TO_COL[n_lbl]
    cf = CAT_LABEL_TO_COL[c_lbl]
    return (
        kp.page_global(store, sample, nf, c_per)
        | dict(
            fig_beeswarm=ch.fig_beeswarm(store, sample),
            fig_global_dot=ch.fig_global_dot(store),
            fig_ale=ch.fig_ale_pdp_ice(store, nf, c_per, eid, show_ice=True),
            fig_gpdp=ch.fig_cat_gpdp(store, cf),
        )
    )


def view_drift(store: dd.Store, sort_by: str, f_lbl: str) -> dict:
    col = ALL_LABEL_TO_COL[f_lbl]
    return (
        kp.page_drift(store)
        | dict(
            fig_drift_db=ch.fig_drift_dumbbell(store, sort_by),
            fig_drift_dist=ch.fig_drift_dist(store, col),
        )
    )


def view_cases(store: dd.Store, case_name: str, period: str, tau: float) -> dict:
    info = dd.CURATED_CASES[case_name]
    eid = int(info["EntityId"])
    txt = f"{info['verdict_tag']} — {info['why_case']} Рекомендация: {info['action']}."
    return (
        kp.page_cases(store, eid, period, tau)
        | dict(
            case_summary_text=txt,
            fig_verify=ch.fig_verify_wilson(store, eid, period),
            tbl_verify=dd.verification_table(store, eid, period),
        )
    )


def compute_all(
    store: dd.Store, eid: int, tau: float, n_top: int, detail: str,
    space: str, color: str, grp_mode: str, sample: str, n_lbl: str, c_per: str, c_lbl: str,
    d_sort: str, d_feat: str, c_name: str, v_per: str,
) -> dict:
    out = dict(
        period_caption=header_caption(store, tau),
        tau_display=f"τ = {dd.fmt_num(tau, 4)} ({dd.fmt_proba(tau)})",
    )
    out |= view_card(store, eid, tau, n_top, detail)
    out |= view_similar(store, eid, space, color, grp_mode, tau)
    out |= view_global(store, sample, n_lbl, c_per, c_lbl, eid)
    out |= view_drift(store, d_sort, d_feat)
    out |= view_cases(store, c_name, v_per, tau)
    return out


_eid0 = dd.parse_entity_id(card_sel, int(STORE.team5_cards["EntityId"].iloc[0]))
globals().update(compute_all(
    STORE, _eid0, dd.DEFAULT_TAU, top_n, detail_lvl,
    map_space, map_color, group_mode, bw_sample, num_feat_lbl, curve_period, cat_feat_lbl,
    drift_sort, drift_feat_lbl, case_sel, verif_period,
))


# ---------------------------------------------------------------- обработчики Taipy
def _assign(state, values: dict):
    with state:
        for k, v in values.items():
            setattr(state, k, v)


def _cur_eid(state) -> int:
    return dd.parse_entity_id(state.card_sel, int(STORE.team5_cards["EntityId"].iloc[0]))


def _refresh_card_and_similar(state, tau: float):
    eid = _cur_eid(state)
    vals = dict(
        period_caption=header_caption(STORE, tau),
        tau_display=f"τ = {dd.fmt_num(tau, 4)} ({dd.fmt_proba(tau)})",
    )
    vals |= view_card(STORE, eid, tau, int(state.top_n), state.detail_lvl)
    vals |= view_similar(STORE, eid, state.map_space, state.map_color, state.group_mode, tau)
    vals |= view_global(STORE, state.bw_sample, state.num_feat_lbl, state.curve_period, state.cat_feat_lbl, eid)
    vals |= view_cases(STORE, state.case_sel, state.verif_period, tau)
    _assign(state, vals)


def on_outcome_filter(state, var_name=None, value=None):
    tau = current_tau(state.tau_preset, state.log10_tau)
    opts = dd.card_options(STORE, state.outcome_flt, tau)
    sel = state.card_sel if state.card_sel in opts else opts[0]
    with state:
        state.CARD_LOV = opts
        state.card_sel = sel
    _refresh_card_and_similar(state, tau)


def on_card_change(state, var_name=None, value=None):
    tau = current_tau(state.tau_preset, state.log10_tau)
    _refresh_card_and_similar(state, tau)


def on_tau_preset(state, var_name=None, value=None):
    p_val = dd.PRESETS.get(state.tau_preset)
    if p_val is not None:
        with state:
            state.log10_tau = float(np.log10(p_val))
    tau = current_tau(state.tau_preset, state.log10_tau)
    opts = dd.card_options(STORE, state.outcome_flt, tau)
    eid = _cur_eid(state)
    matched = next((o for o in opts if str(eid) in o), opts[0])
    with state:
        state.CARD_LOV = opts
        state.card_sel = matched
    _refresh_card_and_similar(state, tau)


def on_tau_slider(state, var_name=None, value=None):
    custom_key = "Свой порог (слайдер)"
    tau = float(np.clip(10.0 ** float(state.log10_tau), 1e-5, 0.99))
    opts = dd.card_options(STORE, state.outcome_flt, tau)
    eid = _cur_eid(state)
    matched = next((o for o in opts if str(eid) in o), opts[0])
    with state:
        state.tau_preset = custom_key
        state.CARD_LOV = opts
        state.card_sel = matched
    _refresh_card_and_similar(state, tau)


def on_card_ctl(state, var_name=None, value=None):
    tau = current_tau(state.tau_preset, state.log10_tau)
    _assign(state, view_card(STORE, _cur_eid(state), tau, int(state.top_n), state.detail_lvl))


def on_similar_ctl(state, var_name=None, value=None):
    tau = current_tau(state.tau_preset, state.log10_tau)
    _assign(state, view_similar(STORE, _cur_eid(state), state.map_space, state.map_color, state.group_mode, tau))


def on_global_ctl(state, var_name=None, value=None):
    _assign(state, view_global(STORE, state.bw_sample, state.num_feat_lbl, state.curve_period, state.cat_feat_lbl, _cur_eid(state)))


def on_drift_ctl(state, var_name=None, value=None):
    _assign(state, view_drift(STORE, state.drift_sort, state.drift_feat_lbl))


def on_case_ctl(state, var_name=None, value=None):
    tau = current_tau(state.tau_preset, state.log10_tau)
    _assign(state, view_cases(STORE, state.case_sel, state.verif_period, tau))


def on_open_case_in_card(state, *args):
    eid = int(dd.CURATED_CASES[state.case_sel]["EntityId"])
    tau = current_tau(state.tau_preset, state.log10_tau)
    opts = dd.card_options(STORE, OUTCOME_LOV[0], tau)
    matched = next((o for o in opts if str(eid) in o), opts[0])
    with state:
        state.outcome_flt = OUTCOME_LOV[0]
        state.CARD_LOV = opts
        state.card_sel = matched
    _refresh_card_and_similar(state, tau)
    navigate(state, "card")


def on_reset(state, *args):
    opts = dd.card_options(STORE, OUTCOME_LOV[0], dd.DEFAULT_TAU)
    with state:
        state.outcome_flt = OUTCOME_LOV[0]
        state.tau_preset = PRESET_LOV[0]
        state.log10_tau = float(np.log10(dd.DEFAULT_TAU))
        state.CARD_LOV = opts
        state.card_sel = opts[0]
        state.top_n = 10
        state.detail_lvl = DETAIL_LOV[0]
    _refresh_card_and_similar(state, dd.DEFAULT_TAU)


def on_init(state):
    navigate(state, "card")


# ---------------------------------------------------------------- разметка страниц
def card(title: str, fig: str, height: int, controls: str = "", n_ctl: int = 0) -> str:
    # Экранируем вертикальную черту, чтобы Taipy не обрезал заголовки
    safe_title = title.replace("|", "∣")
    if controls:
        cols = " ".join(["1"] * max(n_ctl, 1))
        head = (
            f"<|{safe_title}|text|class_name=card-title|>\n"
            f"<|layout|columns={cols}|gap=12px|class_name=card-controls|\n{controls}\n|>"
        )
    else:
        head = f"<|{safe_title}|text|class_name=card-title|>"
    return (
        f"<|part|class_name=card|\n{head}\n"
        f"<|chart|figure={{{fig}}}|plot_config={{PLOT_CONFIG}}|height={height}px|>\n|>"
    )


def table_card(title: str, tbl_var: str, page_size: int = 15) -> str:
    return (
        f"<|part|class_name=card|\n<|{title}|text|class_name=card-title|>\n"
        f"<|{{{tbl_var}}}|table|page_size={page_size}|show_all=False|class_name=card-table|>\n|>"
    )


def two(a: str, b: str, cols: str = "1 1") -> str:
    return f"<|layout|columns={cols}|gap=16px|\n{a}\n{b}\n|>"


def question(text: str) -> str:
    return f"<|{text}|text|class_name=page-q|>"


root_page = """
<|part|class_name=topbar|
<|layout|columns=1 auto|class_name=topbar-row|
<|part|class_name=brand|
<|SOC · объяснение решений модели lgbm_gate1 (SHAP / XAI)|text|class_name=brand-title|>
<|{period_caption}|text|class_name=brand-sub|>
|>
<|navbar|lov={NAV}|class_name=nav|>
|>

<|layout|columns=1 2 2 2 auto|gap=12px|class_name=filters|
<|{outcome_flt}|selector|lov={OUTCOME_LOV}|dropdown|label=Фильтр исхода|on_change=on_outcome_filter|>

<|{card_sel}|selector|lov={CARD_LOV}|dropdown|label=Карточка команды 5 (team = 5)|on_change=on_card_change|>

<|{tau_preset}|selector|lov={PRESET_LOV}|dropdown|label=Порог модели τ (пресет)|on_change=on_tau_preset|>

<|part|class_name=slider-box|
<|Ручка порога (лог-шкала): {tau_display}|text|class_name=slider-label|>
<|{log10_tau}|slider|min=-5.0|max=0.0|step=0.05|continuous=False|on_change=on_tau_slider|>
|>

<|Сбросить|button|on_action=on_reset|class_name=reset|>
|>
|>

<|part|class_name=page|
<|content|>
|>
"""

card_page = "\n\n".join([
    question("Почему модель дала выбранной карточке именно такую оценку?"),
    kp.markup("card"),
    card(
        "Вклад признаков в решение модели (SHAP Waterfall · log-odds; объясняет модель, не причинность)",
        "fig_card_wf", 460,
        controls=(
            "<|part|class_name=slider-box-sm|\n"
            "<|Top-N признаков: {top_n}|text|class_name=slider-label|>\n"
            "<|{top_n}|slider|min=5|max=20|step=1|continuous=False|on_change=on_card_ctl|>\n|>\n\n"
            "<|{detail_lvl}|selector|lov={DETAIL_LOV}|dropdown|label=Уровень детализации|on_change=on_card_ctl|>"
        ),
        n_ctl=2,
    ),
    card("Баланс аргументов модели: суммарный вклад «за инцидент» (+) против «за FP» (−)", "fig_card_bal", 230),
    table_card("Все 73 поля карточки: значения, тип (поле карточки / сконструирован) и вклады SHAP", "tbl_card_fields", page_size=15),
])

similar_page = "\n\n".join([
    question("Насколько похожие вердикты получают похожие карточки в потоке?"),
    kp.markup("similar"),
    card(
        "Карта похожести карточек: гексагональная плотность области модели (~41,7 тыс.) + команда 5 и 20-NN",
        "fig_hexmap", 490,
        controls=(
            "<|{map_space}|selector|lov={SPACE_LOV}|dropdown|label=Пространство проекции|on_change=on_similar_ctl|>\n\n"
            "<|{map_color}|selector|lov={COLOR_LOV}|dropdown|label=Окраска ячеек Hexbin|on_change=on_similar_ctl|>"
        ),
        n_ctl=2,
    ),
    two(
        card(
            "Распределение вкладов SHAP в группе похожих карточек (звезда — выбранная карточка)",
            "fig_violin", 460,
            controls="<|{group_mode}|selector|lov={GROUP_LOV}|dropdown|label=Группа сравнения|on_change=on_similar_ctl|>",
            n_ctl=1,
        ),
        table_card("20 ближайших соседей в многомерном пространстве (косинусное сходство)", "tbl_neighbors", page_size=10),
        cols="3 2",
    ),
])

global_page = "\n\n".join([
    question("На какие сигналы модель опирается во всём потоке и как форма признаков влияет на скор?"),
    kp.markup("global"),
    two(
        card(
            "SHAP Summary Plot (Beeswarm): сила и направление топ-15 полей",
            "fig_beeswarm", 500,
            controls="<|{bw_sample}|selector|lov={SAMPLE_LOV}|dropdown|label=Выборка|on_change=on_global_ctl|>",
            n_ctl=1,
        ),
        card("Глобальная важность по выборкам (средний модуль SHAP) и проверка Ablation", "fig_global_dot", 500),
    ),
    two(
        card(
            "Числовой признак: ALE, PDP и индивидуальные ICE-кривые",
            "fig_ale", 460,
            controls=(
                "<|{num_feat_lbl}|selector|lov={NUM_LOV}|dropdown|label=Числовой признак (43)|on_change=on_global_ctl|>\n\n"
                "<|{curve_period}|selector|lov={SAMPLE_LOV}|dropdown|label=Период|on_change=on_global_ctl|>"
            ),
            n_ctl=2,
        ),
        card(
            "Категориальное поле: group-PDP (what-if подстановка уровня целиком)",
            "fig_gpdp", 460,
            controls="<|{cat_feat_lbl}|selector|lov={CAT_LOV}|dropdown|label=Категориальное поле (12)|on_change=on_global_ctl|>",
            n_ctl=1,
        ),
    ),
])

drift_page = "\n\n".join([
    question("Как изменились распределения признаков и их SHAP-вклады от приёмки к эксплуатации?"),
    kp.markup("drift"),
    two(
        card(
            "Сдвиг глобальной важности SHAP (Приёмка → Эксплуатация) и индекс дрейфа PSI",
            "fig_drift_db", 500,
            controls="<|{drift_sort}|selector|lov={DRIFT_SORT_LOV}|dropdown|label=Сортировка|on_change=on_drift_ctl|>",
            n_ctl=1,
        ),
        card(
            "Профиль распределения признака: Приёмка против Эксплуатации",
            "fig_drift_dist", 500,
            controls="<|{drift_feat_lbl}|selector|lov={DRIFT_FEAT_LOV}|dropdown|label=Поле (73)|on_change=on_drift_ctl|>",
            n_ctl=1,
        ),
    ),
])

cases_page = "\n\n".join([
    question("Где объяснение модели подтверждается данными, а где оно выглядит сомнительно?"),
    """
<|part|class_name=card|
<|layout|columns=2 1 auto|gap=12px|class_name=card-head|
<|{case_sel}|selector|lov={CASE_LOV}|dropdown|label=Кейс команды 5 для проверки|on_change=on_case_ctl|>

<|{verif_period}|selector|lov={VERIF_PER_LOV}|dropdown|label=Историческая база сверки|on_change=on_case_ctl|>

<|Открыть на вкладке «1 · Почему так?»|button|on_action=on_open_case_in_card|class_name=reset|>
|>
<|{case_summary_text}|text|class_name=case-summary|>
|>
""",
    kp.markup("cases"),
    card("Сверка топ-12 признаков кейса: вклад SHAP модели против фактической доли инцидентов (95 % ДИ Уилсона)", "fig_verify", 500),
    table_card("Детальная таблица верификации топ-12 признаков кейса", "tbl_verify", page_size=12),
])

pages = {
    "/": root_page,
    "card": card_page,
    "similar": similar_page,
    "global": global_page,
    "drift": drift_page,
    "cases": cases_page,
}

stylekit = {
    "color_primary": "#2a78d6",
    "color_secondary": "#52514e",
    "color_background_light": "#f6f7f9",
    "color_paper_light": "#ffffff",
    "font_family": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "border_radius": 10,
    "input_button_height": "40px",
    "root_margin": "0",
}

gui = Gui(pages=pages, css_file="dashboard.css")

if __name__ == "__main__":
    gui.run(
        title="SOC · объяснимость модели (SHAP / XAI)",
        dark_mode=False,
        stylekit=stylekit,
        port=5004,
        watermark="",
        use_reloader=False,
    )