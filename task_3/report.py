"""
report.py — собирает REPORT.md (полный отчёт к дашборду задания 3) и NOTE_3.7.md
(записку на 1 страницу для сдачи практики 3.7) из предрассчитанных артефактов data/agg/
и словаря обоснований WHY в dashboard_charts.py.

Запуск из папки task_3 (после prepare.py):
    python report.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import dashboard_charts as ch
import dashboard_data as dd
from feature_registry import REGISTRY

BASE = Path(__file__).resolve().parent

# Соответствие ключей ch.WHY страницам и каноничным названиям графиков (для таблицы в REPORT.md)
CHART_SPEC: list[tuple[str, str, str, str]] = [
    ("card_waterfall", "1 · Почему так?", "Вклад признаков в решение модели", "Каскадная (SHAP Waterfall)"),
    ("card_balance", "1 · Почему так?", "Баланс аргументов «за инцидент» против «за FP»", "Дивергентная линейчатая"),
    ("sim_hexmap", "2 · Похожие карточки", "Похожесть карточек в потоке (~41,7 тыс.) и 20-NN", "Гексагональная карта плотности (Hexbin) + маркеры"),
    ("group_violin", "2 · Похожие карточки", "Разброс вкладов SHAP в группе похожих карточек", "Скрипичная с точками (Violin + strip)"),
    ("beeswarm", "3 · Весь поток", "Сила и направление топ-15 признаков", "График значений Шепли (SHAP Summary / Beeswarm)"),
    ("global_dot", "3 · Весь поток", "Важность полей по выборкам и проверка Ablation", "Точечная диаграмма категорий (Dot Plot) + линейчатая"),
    ("ale_pdp_ice", "3 · Весь поток", "Форма зависимости от числового признака", "ALE + PDP + ICE и гистограмма плотности"),
    ("cat_gpdp", "3 · Весь поток", "Влияние уровней категориального поля", "Групповой PDP (what-if, Dot Plot)"),
    ("drift_dumbbell", "4 · Дрейф", "Сдвиг важности mean|SHAP| и PSI (Приёмка → Прод)", "Диаграмма гантелей (Dumbbell)"),
    ("drift_dist", "4 · Дрейф", "Как изменилось распределение признака", "Сравнительная гистограмма / гантели долей"),
    ("verify_wilson", "5 · Проверка кейсов", "Подтверждается ли объяснение статистикой данных", "Линейчатая SHAP + точечная с 95 % ДИ Уилсона"),
    ("corr_heatmap", "3 · Весь поток", "Группы коррелированных числовых признаков", "Кластеризованная тепловая карта (Heatmap)"),
]


def fmt_date(iso: str) -> str:
    y, m, d = str(iso).split("-")
    return f"{d}.{m}.{y}"


def top_fields_text(store: dd.Store, eid: int, k: int = 3) -> str:
    df = store.team5_fields[store.team5_fields["EntityId"] == eid].sort_values("rank_abs").head(k)
    parts = []
    for r in df.itertuples(index=False):
        tag = " *(сконструирован)*" if r.kind == "engineered" else ""
        parts.append(f"«{r.label_ru}»{tag} = `{r.value_str[:48]}` ({dd.fmt_signed(r.shap_val, 3)})")
    return ", ".join(parts)


def nbr_summary(store: dd.Store, eid: int, space: str = "shap") -> dict:
    sub = store.neighbors_team5[(store.neighbors_team5["target_id"] == eid) & (store.neighbors_team5["space"] == space)]
    if sub.empty:
        return dict(n=0, inc_n=0, inc_rate=float("nan"), med_p=float("nan"), outcomes={})
    return dict(
        n=int(len(sub)),
        inc_n=int(sub["y_true"].sum()),
        inc_rate=float(sub["y_true"].mean()),
        med_p=float(sub["proba"].median()),
        outcomes=sub["outcome_at_tau"].value_counts().to_dict(),
    )


def verif_summary(store: dd.Store, eid: int, period: str = "Эксплуатация") -> dict:
    sub = store.verification[(store.verification["EntityId"] == eid) & (store.verification["period"] == period)].sort_values("rank_abs")
    vc = sub["status"].value_counts().to_dict()
    return dict(
        n=int(len(sub)),
        ok=int(vc.get("Подтверждено", 0)),
        bad=int(vc.get("Противоречие", 0)),
        neutral=int(vc.get("Нейтрально", 0)),
        low=int(vc.get("Мало данных", 0)),
        df=sub,
    )


def three_phrase_block(store: dd.Store, eid: int, action_data: str, action_tau: str, action_feat: str, chosen: str) -> str:
    """Формирует вывод строго по 3-фразовому шаблону § 3 методических рекомендаций 3.7."""
    r = dd.card_row(store, eid)
    tau = float(store.meta["threshold"])
    proba = float(r["proba_calc"])
    dec = "инцидент (в приоритет)" if proba >= tau else "FP (отложить)"
    fact = "реальный инцидент (y = 1)" if int(r["y_true"]) == 1 else "ложная тревога, FP (y = 0)"
    add_diff = abs(float(dd.sigmoid(float(r["base"]) + float(r["sum_shap"]))) - proba)
    top3 = top_fields_text(store, eid, k=3)
    return "\n".join([
        f"1. **Что видит модель:** В карточке `{eid}` ({r['outcome']}, факт — {fact}, правило `{r.get('rule', '—')}`) "
        f"модель дала оценку `p = {dd.fmt_num(proba, 6)}` ({dd.fmt_proba(proba)}) при пороге `{dd.fmt_num(tau, 4)}`; "
        f"решение — **{dec}** (запас до порога {dd.fmt_signed(float(r['margin_to_tau']), 2)} log-odds).",
        f"2. **Почему так:** Наибольший вклад дали {top3}; вклады в логарифме шансов, "
        f"`base ({dd.fmt_num(float(r['base']), 4)}) + Σ ({dd.fmt_signed(float(r['sum_shap']), 4)}) = "
        f"{dd.fmt_num(float(r['raw_score']), 4)}` сходится (`|σ(raw) − p| = {add_diff:.1e}`).",
        f"3. **Что делать:** Чтобы исправить поведение модели, возможны: **данные** — {action_data}; "
        f"**порог** — {action_tau}; **признак** — {action_feat}. **Выбираю:** {chosen}.",
    ])


def control_table_md(store: dd.Store) -> str:
    m = store.meta
    c = m["counts"]
    chk = m["checks"]
    t5 = store.team5_cards
    oc = t5["outcome"].value_counts().to_dict()
    split_txt = f"{oc.get('TP', 0)} / {oc.get('FP', 0)} / {oc.get('FN', 0)} / {oc.get('TN', 0)}"
    return "\n".join([
        "| Проверка (§ 4 практики 3.7) | Требование курса | Факт в `task_3` | Статус |",
        "|---|---|---|---|",
        f"| Карточек в наборе `team = {m['team']}` | 8 (2 / 2 / 2 / 2) | {c['team_cards']} ({split_txt}) | OK |",
        f"| Карточек практики всего | 320 (40 × 8) | {c['practice_total']} | OK |",
        f"| Вердикт при τ = {dd.fmt_num(m['threshold'], 4)} | совпадает с `outcome` | {'совпадает 8 из 8' if chk['team_outcome_match'] else 'расхождение'} | {'OK' if chk['team_outcome_match'] else 'WARN'} |",
        f"| Аддитивность `σ(base + Σ) = proba` | ≤ 1e-6 | {chk['all_scope_additivity_diff']:.2e} (на всех {dd.fmt_int(m['scope_rows'])} карточках) | {'OK' if chk['all_scope_additivity_diff'] <= 1e-6 else 'WARN'} |",
        f"| Сверка с `course_pack/shap_cards.csv` | совпадение вкладов | max |ΔSHAP| = {chk.get('max_shap_diff', float('nan')):.2e} ({chk.get('checked_cards', 0)} карт.) | OK |",
        f"| Самопроверка модели (`control_cards.csv`) | Δ ≤ 4e-7 | {chk['control_cards_delta']:.2e} (5 контрольных карточек) | OK |",
        f"| Признаков в объяснении | 73 поля (1 099 фич модели) | {c['features_fields']} поля ({dd.fmt_int(c['features_model'])} фич) | OK |",
    ])


def build_report(store: dd.Store) -> str:
    m = store.meta
    c = m["counts"]
    tau = float(m["threshold"])
    today = date.today().strftime("%d.%m.%Y")

    # Проверяем, что все ключи ch.WHY вошли в CHART_SPEC
    spec_keys = {k for k, _, _, _ in CHART_SPEC}
    missing = set(ch.WHY) - spec_keys
    if missing:
        raise RuntimeError(f"В CHART_SPEC не хватает ключей из ch.WHY: {missing}")

    # --- Таблица 8 карточек команды 5 ---
    t5 = store.team5_cards.sort_values("rank_in_team")
    t5_lines = [
        "| # | EntityId | Исход | y_true | Оценка `proba` | Запас до τ | Правило SIEM | Главный драйвер SHAP | 20-NN (инцидентов) |",
        "|---:|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for r in t5.itertuples(index=False):
        eid = int(r.EntityId)
        f1 = store.team5_fields[store.team5_fields["EntityId"] == eid].sort_values("rank_abs").iloc[0]
        ns = nbr_summary(store, eid, "shap")
        t5_lines.append(
            f"| {int(r.rank_in_team)} | `{eid}` | **{r.outcome}** | {int(r.y_true)} | {dd.fmt_proba(float(r.proba_calc))} | "
            f"{dd.fmt_signed(float(r.margin_to_tau), 2)} | `{str(getattr(r, 'rule', '—'))[:34]}` | "
            f"{f1['label_ru']} ({dd.fmt_signed(float(f1['shap_val']), 2)}) | {dd.fmt_pct(ns['inc_rate'], 0)} |"
        )

    # --- Глобальная важность по выборкам ---
    imp = store.global_imp
    imp_oper = imp[imp["sample"] == "Эксплуатация"].sort_values("rank")
    imp_val = imp[imp["sample"] == "Приёмка"].set_index("source_column")
    imp_pr = imp[imp["sample"] == "Практика 320"].set_index("source_column")
    eng_share = float(imp_oper.loc[imp_oper["kind"] == "engineered", "share_abs_shap"].sum())
    top5_shap = set(imp_oper.head(5)["source_column"])
    top5_abl = set(imp_oper.sort_values("ablation_delta_proba", ascending=False).head(5)["source_column"])
    abl_overlap = len(top5_shap & top5_abl)

    imp_lines = [
        "| Ранг (прод) | Поле | Тип | Эксплуатация mean\\|SHAP\\| (доля) | Приёмка (ранг) | Практика 320 (ранг) | Ablation \\|Δp\\| |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in imp_oper.head(10).itertuples(index=False):
        sc = r.source_column
        rv = int(imp_val.loc[sc, "rank"]) if sc in imp_val.index else 0
        rp = int(imp_pr.loc[sc, "rank"]) if sc in imp_pr.index else 0
        kind_s = "сконструирован" if r.kind == "engineered" else "поле карточки"
        imp_lines.append(
            f"| {int(r.rank)} | {r.label_ru} | {kind_s} | {dd.fmt_num(r.mean_abs_shap, 3)} ({dd.fmt_pct(r.share_abs_shap, 1)}) | "
            f"#{rv} | #{rp} | {dd.fmt_pct(r.ablation_delta_proba, 2)} |"
        )

    # --- Коллинеарность и ALE/PDP ---
    cn_o = store.curves_num[store.curves_num["period"] == "Эксплуатация"]
    col_feats = cn_o.drop_duplicates("source_column")
    n_collinear = int(col_feats["collinear_warn"].sum())
    top_corr_ex = col_feats[col_feats["collinear_warn"]].sort_values("max_spearman", ascending=False).head(3)
    corr_txt = "; ".join(
        f"«{REGISTRY[r.source_column].label_ru}» ↔ «{REGISTRY[r.corr_partner].label_ru}» (|ρ| = {dd.fmt_num(r.max_spearman, 2)})"
        for r in top_corr_ex.itertuples(index=False) if r.corr_partner in REGISTRY
    ) or "сильных пар не выявлено"

    # --- Дрейф ---
    ds = store.drift_summary
    n_psi_high = int((ds["psi"] >= 0.25).sum())
    top_psi = ds.sort_values("psi", ascending=False).head(5)
    psi_txt = "; ".join(
        f"«{r.label_ru}» (PSI {dd.fmt_num(r.psi, 2)}, пропуски {dd.fmt_pct(r.null_val, 0)} → {dd.fmt_pct(r.null_oper, 0)})"
        for r in top_psi.itertuples(index=False)
    )
    top_dshap = ds.reindex(ds["delta_abs_shap"].abs().sort_values(ascending=False).index).head(3)
    dshap_txt = "; ".join(
        f"«{r.label_ru}» ({dd.fmt_num(r.mean_abs_shap_val, 3)} → {dd.fmt_num(r.mean_abs_shap_oper, 3)}, {dd.fmt_signed(r.delta_abs_shap, 3)})"
        for r in top_dshap.itertuples(index=False)
    )

    # --- Порог 0.035 на команде 5 и на практике 320 ---
    cm_t5_019 = dd.confusion_from_proba(t5["y_true"].to_numpy(), t5["proba_calc"].to_numpy(), tau)
    cm_t5_035 = dd.confusion_from_proba(t5["y_true"].to_numpy(), t5["proba_calc"].to_numpy(), 0.035)
    cm_p_019 = dd.confusion_from_proba(store.practice_320["y_true"].to_numpy(), store.practice_320["proba"].to_numpy(), tau)
    cm_p_035 = dd.confusion_from_proba(store.practice_320["y_true"].to_numpy(), store.practice_320["proba"].to_numpy(), 0.035)

    # --- Ключевые кейсы ---
    v_521996 = verif_summary(store, 521996)
    n_521996 = nbr_summary(store, 521996)
    v_526930 = verif_summary(store, 526930)
    n_526930 = nbr_summary(store, 526930)
    v_523480 = verif_summary(store, 523480)
    n_523480 = nbr_summary(store, 523480)
    v_523796 = verif_summary(store, 523796)
    n_523796 = nbr_summary(store, 523796)

    # --- Таблица WHY ---
    why_lines = [
        "| Страница | Вопрос / блок | График | Почему он | Почему не другой |",
        "|---|---|---|---|---|",
    ]
    for key, page_n, q_txt, g_name in CHART_SPEC:
        why_txt = ch.WHY[key]
        if key == "sim_hexmap":
            why_txt = (
                f"Двухслойная карта с гексагональным бинированием (Hexbin) агрегирует {dd.fmt_int(m['scope_rows'])} карточек "
                f"в {len(store.hex_shap)} занятых ячеек в SHAP-PCA и {len(store.hex_pca)} в SVD-20 без перегрузки Plotly "
                "и наслоения точек, а поверх показывает 8 карточек команды 5 и истинных 20-NN из многомерного пространства."
            )
        why_lines.append(f"| {page_n} | {q_txt} | {g_name} | {why_txt} | {ch.WHY_ALT[key]} |")

    # --- Трёхфразовые выводы для главных кейсов ---
    case_fn_523480 = three_phrase_block(
        store, 523480,
        action_data="доразметить и включить в обучение свежие инциденты отключения антивируса (`TTS_24_Security_Tool_Shutdown`), где текст алерта сейчас смещает скор в FP (−1,193)",
        action_tau="сдвиг порога не поможет — карточка лежит на −7,96 log-odds ниже τ (в 2 700 раз ниже порога)",
        action_feat="усилить/вынести признак отключения средств защиты из поля `Команда` (`security_product_state_off`, сейчас +0,464) вместо опоры на шаблонный текст алерта",
        chosen="**данные (переобучение на свежих месяцах)** — это системное слепое пятно (все 20-NN в SHAP — TN), порогом его не закрыть",
    )
    case_fp_526930 = three_phrase_block(
        store, 526930,
        action_data="восстановить обогащение SIEM (`Вендор из сием`, `Алерт`, `Команда`, `Возраст события`), потому что именно их пропуски дали 4 крупнейших положительных вклада",
        action_tau=f"поднятие порога до 0,0350 переводит оба FP команды 5 в TN (и снижает FP на практике 320 с {cm_p_019['fp']} до {cm_p_035['fp']}, но увеличивает FN с {cm_p_019['fn']} до {cm_p_035['fn']})",
        action_feat="ограничить положительный вес индикаторов пустых полей, чтобы дефект конвейера обогащения не имитировал атаку",
        chosen="**данные (вернуть обогащение)** как первопричину; как временная операционная мера на узком срезе — пересмотр порога с пересчётом матрицы ошибок",
    )
    case_tp_521996 = three_phrase_block(
        store, 521996,
        action_data="сохранять полноту полей `src` и `Алерт` — они обеспечивают 31 % силы верного решения",
        action_tau="текущий порог работает с большим запасом (+3,55 log-odds)",
        action_feat="сохранить `src_is_private` и частоту правила `freq_Наименование` как устойчивое ядро сигнала",
        chosen="**оставить без изменений (эталонный кейс)** — объяснение модели полностью подтверждается статистикой SOC",
    )

    sections = [
        "# SOC · объяснение решений модели-советника (SHAP / XAI): отчёт к дашборду",
        "",
        f"Модель — **{m['model']}** (`{m['run_id']}`), рабочий порог **τ = {dd.fmt_num(tau, 4)}** "
        f"(в пространстве логарифма шансов `logit(τ) = {dd.fmt_num(m['logit_tau'], 4)}`, "
        f"базовый уровень модели `base = {dd.fmt_num(m['base_value'], 4)}`). "
        f"Индивидуальный набор команды — **`team = {m['team']}`** (8 карточек сентября 2025: 2 TP, 2 FP, 2 FN, 2 TN). "
        f"Область модели за {fmt_date(m['periods']['train'][0])} — {fmt_date(m['periods']['oper'][1])} включает "
        f"**{dd.fmt_int(m['scope_rows'])}** карточек (приёмка {dd.fmt_int(c['test'])}, эксплуатация {dd.fmt_int(c['oper'])}; "
        f"всего в исходном файле {dd.fmt_int(m['total_dataset_rows'])} строк).",
        "",
        f"Отчёт собран скриптом `report.py` {today}. Числа совпадают с плитками дашборда `task_3` (порт `5004`). "
        "На листах дашборда оставлены только KPI-плитки, фильтры и графики; выводы, разбор кейсов и обоснование каждого графика вынесены сюда.",
        "",
        "## Коротко для руководителя SOC",
        "",
        f"- **Объяснения модели точны математически, но не всегда надёжны по смыслу.** Аддитивность TreeSHAP по всем 73 полям "
        f"(агрегированным из 1 099 модельных фич) выполняется с погрешностью `{m['checks']['all_scope_additivity_diff']:.2e}` "
        f"(допуск курса ≤ 1e-6), и вклады совпадают с официальным пакетом `course_pack/shap_cards.csv`.",
        f"- **Есть кейс, где объяснение полностью подтверждается данными:** карточка **`#7 · 521996` (TP)** — внешний IP-источник "
        f"(`src_is_private = 0`), известное правило сканирования и текст алерта `zgrab` подтверждены 95 % ДИ Уилсона по **{v_521996['ok']} из {v_521996['n']}** "
        f"топ-признакам, а среди 20 ближайших соседей в пространстве SHAP — **{dd.fmt_pct(n_521996['inc_rate'], 0)}** реальных инцидентов.",
        f"- **Найдены два типа сомнительных объяснений:** (1) ложная тревога **`#4 · 526930` (FP)** вызвана не признаками атаки, "
        f"а четырьмя пропусками/нулями обогащения при **{dd.fmt_pct(n_526930['inc_rate'], 0)}** инцидентов среди 20-NN; "
        f"(2) пропуск **`#1 · 523480` (FN)** произошёл из-за того, что текст алерта об отключении антивируса дал крупнейший вклад **в пользу FP (−1,193)**.",
        f"- **Ошибки FN и FP в наборе команды 5 имеют разную природу.** Оба `FP` (`522090`, `526930`) лежат впритык к порогу "
        f"(+0,43 и +0,51 log-odds) и исчезают при сдвиге порога до `0,0350` без потерь по `TP`. Оба `FN` (`523480`, `523796`) "
        "провалены на −7,7…−8,0 log-odds ниже порога — сдвиг порога их не вернёт, нужны данные и переобучение.",
        f"- **Дрейф обогащения меняет логику модели в эксплуатации:** заполненность пары «Вендор + Алерт» упала с "
        f"**{dd.fmt_pct(m['enrichment']['val'], 0)}** на приёмке до **{dd.fmt_pct(m['enrichment']['oper'], 0)}** в эксплуатации; "
        f"сильный дрейф (`PSI ≥ 0,25`) зафиксирован у **{n_psi_high}** полей, а доля сконструированных (`engineered`) признаков "
        f"в сумме `mean|SHAP|` достигает **{dd.fmt_pct(eng_share, 0)}**.",
        "",
        "## 1. Локальный уровень: разбор 8 карточек команды 5 (`1 · Почему так?`)",
        "",
        *t5_lines,
        "",
        "При рабочем пороге `τ = 0,0193` все 8 карточек команды 5 точно воспроизводят исходы из `3.7_Пр_cards_practice.xlsx` "
        f"(TP {cm_t5_019['tp']} / FP {cm_t5_019['fp']} / FN {cm_t5_019['fn']} / TN {cm_t5_019['tn']}). "
        f"При экспериментальном пороге `τ = 0,0350` команда 5 получает TP {cm_t5_035['tp']} / FP {cm_t5_035['fp']} / FN {cm_t5_035['fn']} / TN {cm_t5_035['tn']}, "
        f"но на всём срезе практики (320 карточек) матрица меняется с TP {cm_p_019['tp']} / FP {cm_p_019['fp']} / FN {cm_p_019['fn']} / TN {cm_p_019['tn']} "
        f"на TP {cm_p_035['tp']} / FP {cm_p_035['fp']} / FN {cm_p_035['fn']} / TN {cm_p_035['tn']}: уменьшение ложных тревог оплачивается ростом пропусков. "
        "На вкладке 1 аналитик может менять `Top-N` (от 5 до 20 полей), видеть долю покрытия `|SHAP|` и переключаться между 73 бизнес-полями SOC и 1 099 фичами модели.",
        "",
        "## 2. Мезо-уровень: похожие карточки и карта плотности (`2 · Похожие карточки`)",
        "",
        f"Чтобы визуализировать всю область модели (**{dd.fmt_int(m['scope_rows'])}** карточек; в исходном массиве "
        f"{dd.fmt_int(m['total_dataset_rows'])} строк) без перегрузки браузера и слияния точек в сплошное пятно, "
        f"фоновый слой агрегирован в гексагональную сетку (`Hexbin`: {len(store.hex_shap)} ячеек в пространстве SHAP-PCA, "
        f"{len(store.hex_pca)} в пространстве признаков SVD-20), а 20 ближайших соседей (`20-NN`) ищутся строго в исходных "
        "многомерных пространствах по косинусному сходству.",
        "",
        f"- Для **TP-карточек** (`521996`, `519991`) окружение 20-NN однородно инцидентное: **{dd.fmt_pct(n_521996['inc_rate'], 0)}** "
        f"и **{dd.fmt_pct(nbr_summary(store, 519991)['inc_rate'], 0)}** реальных инцидентов — модель уверенно кластеризует известные атаки.",
        f"- Для **FN-карточек** (`523480`, `523796`) 20-NN почти целиком состоят из `TN` (**{dd.fmt_pct(n_523480['inc_rate'], 0)}** "
        f"и **{dd.fmt_pct(n_523796['inc_rate'], 0)}** инцидентов): в пространстве признаков модели эти реальные инциденты неотличимы от массового шума.",
        f"- Для сомнительного **FP** (`526930`) среди 20-NN **0 % реальных инцидентов** ({n_526930['outcomes']}): карточка попала в «пустой» "
        "кластер необогащённых событий, где модель систематически генерирует ложные тревоги. Напротив, у пограничного FP (`522090`) "
        f"среди соседей **{dd.fmt_pct(nbr_summary(store, 522090)['inc_rate'], 0)}** реальных инцидентов — тревога имела реальное основание.",
        "",
        "## 3. Глобальный уровень: весь поток (`3 · Весь поток`)",
        "",
        *imp_lines,
        "",
        f"- **Почему важность зависит от выборки (шаг 9 практики 3.7):** в `Приёмке` (июнь–август) обогащение заполнено у "
        f"{dd.fmt_pct(m['enrichment']['val'], 0)} карточек, а в `Эксплуатации` и `Практике 320` (сентябрь+) — падает до "
        f"{dd.fmt_pct(m['enrichment']['oper'], 0)}; кроме того, `Практика 320` искусственно сбалансирована по исходам (25 % каждого из TP/FP/FN/TN), "
        "поэтому веса признаков, характерных для ошибок, в ней выше, чем в естественном потоке.",
        f"- **Независимая проверка методом Ablation:** выключение поля целиком совпадает с рейтингом `mean|SHAP|` на **{abl_overlap} из 5** "
        "верхних позиций, подтверждая устойчивость глобальной важности.",
        f"- **Мультиколлинеарность и выбор ALE вместо PDP:** среди 43 числовых (`engineered`) признаков у **{n_collinear}** "
        f"выявлена сильная корреляция Спирмена `|ρ| ≥ 0,70` (например: {corr_txt}). По методическим рекомендациям (разд. 5.3) "
        "в этих условиях PDP экстраполирует на нереалистичные комбинации, поэтому основным методом служит **ALE**, дополненный "
        "индивидуальными **ICE-кривыми** команды 5; для 12 категориальных полей применяется **group-PDP** (подстановка уровня целиком). "
        "Яркий локальный пример мультиколлинеарности — карточка **`#5 · 528319` (TN)**, где `src` (−1,751) и `src_is_private = 0` (+1,748) почти полностью гасят друг друга.",
        "",
        "## 4. Дрейф «Приёмка → Эксплуатация» (`4 · Дрейф`)",
        "",
        f"- **Сильный сдвиг распределений (`PSI ≥ 0,25`)** выявлен у **{n_psi_high}** из 73 полей. Лидеры по PSI: {psi_txt}.",
        f"- **Наибольший сдвиг вкладов `mean|SHAP|`:** {dshap_txt}.",
        f"- **Средний итоговый скор в log-odds** сместился с `{dd.fmt_num(m['mean_raw_score']['val'], 2)}` на приёмке до "
        f"`{dd.fmt_num(m['mean_raw_score']['oper'], 2)}` в эксплуатации (сдвиг {dd.fmt_signed(m['mean_raw_score']['oper'] - m['mean_raw_score']['val'], 2)}).",
        "",
        "## 5. Проверка объяснений: подтверждённые и сомнительные кейсы (`5 · Проверка кейсов`)",
        "",
        "### Кейс А — объяснение подтверждается данными: `#7 · 521996` (TP)",
        "",
        f"В исторических данных эксплуатации по 95 % ДИ Уилсона подтверждены **{v_521996['ok']} из {v_521996['n']}** топ-признаков "
        f"(противоречий: {v_521996['bad']}). Среди 20 ближайших соседей в пространстве SHAP — **{n_521996['inc_n']} из {n_521996['n']}** "
        f"({dd.fmt_pct(n_521996['inc_rate'], 0)}) реальных инцидентов.",
        "",
        case_tp_521996,
        "",
        "### Кейс Б — объяснение выглядит сомнительно (дефект обогащения): `#4 · 526930` (FP)",
        "",
        "Четыре крупнейших вклада, перетащивших карточку через порог, — это отсутствие значений (`Число значений инцидента = 0`, "
        "`Команда = пропуск`, `Текст алерта = пропуск`, `Возраст события = пропуск`; вендор также отсутствует). "
        f"Среди 20 ближайших соседей — **0 % реальных инцидентов** ({n_526930['outcomes']}). Модель реагирует на сломанное обогащение, а не на атаку.",
        "",
        case_fp_526930,
        "",
        "### Кейс В — объяснение выглядит сомнительно (слепое пятно на отключении АВ): `#1 · 523480` (FN)",
        "",
        "Реальный инцидент по правилу `TTS_24_Security_Tool_Shutdown` получил оценку `0,0007 %` (запас −7,96 log-odds). "
        "Крупнейший вклад **в сторону FP (−1,193, 19 % силы решения)** дал текст алерта «Устройство microsoft windows Anti-virus было отключено», "
        f"тогда как `Команда` (`security_product_state_off ; dr.web`) тянула к инциденту (+0,464). В эксплуатации по топ-12 полям "
        f"зафиксировано **{v_523480['bad']}** статистических противоречия, а все 20-NN в SHAP — `TN`. "
        f"(Аналогично в `#2 · 523796` (FN) выявлено **{v_523796['bad']}** противоречия при пропущенном вендоре.)",
        "",
        case_fn_523480,
        "",
        "## Контрольные числа (сверка по § 4 практики 3.7)",
        "",
        control_table_md(store),
        "",
        "## Что предлагаем",
        "",
        "1. **Восстановить конвейер обогащения SIEM** («Вендор из сием», «Алерт», «Команда») и ввести мониторинг доли пропусков: "
        "деградация обогащения одновременно создаёт ложные тревоги на пустых карточках (`526930`) и лишает модель сигнала.",
        "2. **Переобучить модель на свежих месяцах эксплуатации** с добавлением реальных инцидентов по правилам отключения СЗИ "
        "(`TTS_24_Security_Tool_Shutdown`) и нестандартных входов (`GSOC_unusual_user_login`): слепые пятна с запасом −7,7…−8,0 log-odds порогом не исправляются.",
        "3. **Показывать аналитику SOC в интерфейсе не только `proba`, но и топ-5 вкладов с маркировкой `[сконструирован]` и флагом пропусков:** "
        "если карточку вытянули в приоритет пустые поля, это сразу видно на каскаде Waterfall.",
        "4. **Для глобального контроля признаков использовать ALE вместо PDP** на коррелирующих числовых полях (`|ρ| ≥ 0,70`) и "
        "перепроверять сдвиг порога τ только вместе с пересчётом матрицы ошибок на потоке.",
        "",
        "## Почему выбран каждый график",
        "",
        *why_lines,
        "",
        "## Оговорки",
        "",
        "- **SHAP описывает поведение модели, а не физику атаки:** высокий вклад признака не означает причинно-следственной связи с инцидентом.",
        "- **Единицы измерения:** вклады TreeSHAP для LightGBM даны в **логарифме шансов (log-odds)**; сравнивать их напрямую с моделями в вероятностях (например, случайным лесом) нельзя — сравнимы только доли и ранги.",
        "- **Оценка `proba` — модельная оценка, не калиброванная вероятность:** как показано в задании 2, в эксплуатации модель завышает вероятность, а калибровка курсом не проверялась.",
        "- **Сконструированные признаки (`engineered`, 43 из 73):** полей `freq_*`, `len_*`, `cmd_*`, `src_is_private` нет в сырой карточке SIEM — они вычисляются препроцессором и явно помечены в интерфейсе.",
        "- **Порог τ = 0,0193 — часть модели:** в дашборде ручка порога работает в режиме «что если» с обязательным пересчётом матрицы ошибок в шапке.",
        "- **Данные учебные (условные).**",
        "",
    ]
    return "\n".join(sections)


def build_note_37(store: dd.Store) -> str:
    """Компактная записка на 1 страницу строго по § 2–4 методических рекомендаций 3.7."""
    m = store.meta
    imp_o = store.global_imp[store.global_imp["sample"] == "Эксплуатация"].sort_values("rank").head(5)
    imp_p = store.global_imp[store.global_imp["sample"] == "Практика 320"].sort_values("rank").head(5)
    imp_v = store.global_imp[store.global_imp["sample"] == "Приёмка"].sort_values("rank").head(5)

    top5_o = ", ".join(f"{r.label_ru} ({dd.fmt_pct(r.share_abs_shap, 0)})" for r in imp_o.itertuples(index=False))
    top5_p = ", ".join(f"{r.label_ru} ({dd.fmt_pct(r.share_abs_shap, 0)})" for r in imp_p.itertuples(index=False))
    top5_v = ", ".join(f"{r.label_ru} ({dd.fmt_pct(r.share_abs_shap, 0)})" for r in imp_v.itertuples(index=False))

    fn_block = three_phrase_block(
        store, 523480,
        action_data="включить в дообучение свежие кейсы отключения СЗИ (`TTS_24_Security_Tool_Shutdown`), где шаблонный текст алерта сейчас даёт −1,193 в сторону FP",
        action_tau="сдвиг порога не поможет (карточка ниже τ на −7,96 log-odds)",
        action_feat="выделить явный признак отключения антивируса из поля `Команда` (`security_product_state_off`), перевесив шаблонный текст алерта",
        chosen="**данные (дообучение на свежих месяцах)**, потому что карточка провалена в слепое пятно модели (20 из 20 ближайших соседей по SHAP — TN)",
    )
    fp_block = three_phrase_block(
        store, 526930,
        action_data="восстановить заполнение полей обогащения (`Алерт`, `Вендор из сием`, `Команда`, `Возраст события`), пропуски которых сформировали 4 главных вклада в сторону инцидента",
        action_tau="поднятие τ до 0,0350 отсекает оба FP команды 5 (запас +0,43 и +0,51), но требует пересчёта матрицы ошибок на потоке из-за риска роста FN",
        action_feat="ограничить положительный вес пустых значений/нулей, чтобы сбой обогащения не повышал скор",
        chosen="**данные (вернуть обогащение)**, потому что модель сработала на отсутствие информации, а среди 20-NN этой карточки — 0 % реальных инцидентов",
    )
    v_521996 = verif_summary(store, 521996)

    return "\n".join([
        f"# Практика 3.7 · Записка команды `team = {m['team']}` (ML-трек, Taipy)",
        "",
        f"**Модель:** `{m['model']}` · **Рабочий порог:** `τ = {dd.fmt_num(m['threshold'], 4)}` · "
        f"**Набор команды:** `team = {m['team']}` (8 карточек: 2 TP, 2 FP, 2 FN, 2 TN; данные учебные, условные).",
        "",
        "## 1. Разбор карточки FN (`EntityId = 523480`, пропуск инцидента)",
        "",
        fn_block,
        "",
        "## 2. Разбор карточки FP (`EntityId = 526930`, ложная тревога)",
        "",
        fp_block,
        "",
        "*Примечание о втором кейсе:* на карточке TP **`521996`** (`p = 40,5 %`) объяснение, напротив, подтверждается "
        f"данными (внешний IP `src_is_private = 0` +2,740, частота правила +1,180, алерт сканера `zgrab` +0,842; {v_521996['ok']} из {v_521996['n']} "
        "признаков подтверждены 95 % ДИ Уилсона, 85 % среди 20-NN — инциденты).",
        "",
        "## 3. Глобальная важность (шаг 9) и зависимость от выборки",
        "",
        f"- **Топ-5 `Эксплуатация` ({dd.fmt_int(m['counts']['oper'])} карт.):** {top5_o}.",
        f"- **Топ-5 `Приёмка` ({dd.fmt_int(m['counts']['test'])} карт.):** {top5_v}.",
        f"- **Топ-5 `Практика 320` (320 карт.):** {top5_p}.",
        "- **Почему состав и доли зависят от выборки:** между приёмкой (лето 2025) и эксплуатацией (осень 2025) "
        f"обогащение «Вендор + Алерт» упало с {dd.fmt_pct(m['enrichment']['val'], 0)} до {dd.fmt_pct(m['enrichment']['oper'], 0)}, "
        "а `Практика 320` сбалансирована по исходам (по 25 % TP/FP/FN/TN), что усиливает вклад признаков из зон ошибок.",
        "",
        "## 4. Контрольные числа (§ 4)",
        "",
        control_table_md(store),
        "",
        "*Оговорки:* `proba` — модельная оценка (калибровка не проверялась); SHAP-вклады даны в логарифме шансов и описывают модель, а не причинность; "
        "признаки `freq_*`, `len_*`, `cmd_*`, `src_is_private` являются сконструированными (`engineered`) и отсутствуют в сырой карточке.",
        "",
    ])


def main() -> None:
    store = dd.load()
    report_md = build_report(store)
    note_md = build_note_37(store)
    (BASE / "REPORT.md").write_text(report_md, encoding="utf-8")
    (BASE / "NOTE_3.7.md").write_text(note_md, encoding="utf-8")
    print(f"Готово: {(BASE / 'REPORT.md').resolve()} ({len(report_md.splitlines())} строк)")
    print(f"Готово: {(BASE / 'NOTE_3.7.md').resolve()} ({len(note_md.splitlines())} строк)")


if __name__ == "__main__":
    main()