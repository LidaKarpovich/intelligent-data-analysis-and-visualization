"""
prepare.py — оценки модели курса `lgbm_gate1__v001` для дашборда «Можно ли доверять модели».

Запуск из папки task_2:
    python prepare.py                                   # ищет датасет и пакет модели сам
    python prepare.py --path путь/к/iavd_ml_dataset.parquet --course-dir путь/к/IAVD_3.7_ML_team

Что считаем (всё — моделью курса, порог τ = 0,0193 не меняем):
    приёмка       тестовый сплит окна обучения 01.06–31.08.2025 — карточки, на которых курс
                  опубликовал метрики (TP 390 / FP 107 / FN 24 / TN 2652). Сплит воспроизводится
                  кодом курса (course_ml.train.make_split, seed 42), метрики сверяются с опубликованными;
    эксплуатация  все карточки с 01.09.2025 до конца выгрузки — их модель не видела.

Область модели та же, что при обучении: есть правило корреляции SIEM, вендор не «Ручной ввод».
Модель оценивает и карточки без вердикта, но метрики считаются только по «Решён» / «False Positive».

Результат в data/agg/:
    scored.parquet  — карточки приёмки и эксплуатации: вердикт, оценка модели, срезы для анализа ошибок
    meta.json       — периоды, порог и варианты порога, метрики приёмки, сверка с эталоном
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

from course_model import CourseAdvisor, find_course_dir

BASE = Path(__file__).resolve().parent
AGG_DIR = BASE / "data" / "agg"
RAW_NAME = "iavd_ml_dataset.parquet"
SEARCH_DIRS = [BASE / "data", BASE / "data_1", BASE.parent / "data_1", BASE.parent / "data"]

WINDOW = (pd.Timestamp("2025-06-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC"))   # окно обучения
OPER_START = WINDOW[1]
LOCAL_TZ = "Europe/Moscow"                       # только для часа суток в срезах
PUBLISHED_TEST = dict(tp=390, fp=107, fn=24, tn=2652)                             # model_card.md
LABELS = {"Решён": 1, "False Positive": 0}

# поля для срезов на дашборде (сами в модель не идут, кроме тех, что модель и так видит)
EXTRA_COLS = ["EntityId", "Создан", "Вердикт", "Наименование", "Тип инцидента", "Пиктограмма IRP",
              "Критичность", "Организация", "Вендор из сием", "Источник инцидента", "Группа устранения",
              "Алерт", "Значение Инцидента (группировка)", "Правило корреляции SIEM"]


def find_raw(path: str | None) -> Path:
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"Нет файла {p}")
        return p
    for d in SEARCH_DIRS:
        if (d / RAW_NAME).exists():
            return d / RAW_NAME
    raise SystemExit(f"Не нашёл {RAW_NAME} в: " + ", ".join(str(d) for d in SEARCH_DIRS))


def name_of(x) -> str | None:
    """JSON-объект с ключом name ('{"name": "L1"}') -> 'L1'."""
    if not isinstance(x, str) or not x:
        return None
    if x.startswith("{"):
        try:
            return json.loads(x).get("name")
        except ValueError:
            return None
    return x


def first_item(x) -> str | None:
    """Поле-список в формате JSON ('["Общий Инцидент ИБ"]') -> первый элемент; строка -> как есть."""
    if not isinstance(x, str) or not x:
        return None
    if x.startswith("["):
        try:
            items = json.loads(x, strict=False)
            return str(items[0]) if items else None
        except (ValueError, IndexError):
            return x
    return x


def rule_of(cards: pd.DataFrame) -> pd.Series:
    """Правило SIEM карточки — из «Правила корреляции SIEM», того же поля, что задаёт область модели."""
    return cards["Правило корреляции SIEM"].map(first_item)


def needed_columns(advisor: CourseAdvisor, available: list[str]) -> list[str]:
    cfg = advisor.cfg
    cols = set(EXTRA_COLS) | set(advisor.data_mod.LIST_LEN_SOURCE)
    for group in cfg["features"]["fields"].values():
        cols |= set(group)
    for key in ("column", "first_event", "last_event"):
        cols.add(cfg["dataset"]["time"][key])
    cols |= {"suser", "duser", "ProcessPath", "Список вредоносных команд", cfg["dataset"]["target"]["column"]}
    return [c for c in available if c in cols]                       # порядок как в файле


def load_raw(path: Path, advisor: CourseAdvisor) -> pd.DataFrame:
    """Карточки с 01.06.2025 в исходном порядке строк (от порядка зависит сплит курса)."""
    pf = pq.ParquetFile(path)
    cols = needed_columns(advisor, pf.schema_arrow.names if pf.schema_arrow is not None else pf.column_names)
    print(f"Читаю {path.name}: {pf.metadata.num_rows:,} строк, {len(cols)} колонок для модели и срезов")
    parts = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        created = pd.to_datetime(df["Создан"], format="ISO8601", utc=True)
        parts.append(df[created >= WINDOW[0]])
    return pd.concat(parts, ignore_index=True)


def in_scope(df: pd.DataFrame, advisor: CourseAdvisor) -> pd.Series:
    flt = advisor.cfg["dataset"]["filters"]
    mask = pd.Series(True, index=df.index)
    if flt.get("rule_not_null"):
        mask &= df["Правило корреляции SIEM"].notna()
    for vendor in flt.get("vendor_not_in", []) or []:
        mask &= ~df["Вендор из сием"].eq(vendor)
    return mask


def window_split(raw: pd.DataFrame, advisor: CourseAdvisor) -> dict:
    """Повторяет course_ml.data.build + course_ml.train.make_split: id карточек train / val / test."""
    cfg = advisor.cfg
    created = pd.to_datetime(raw["Создан"], format="ISO8601", utc=True)
    mask = (in_scope(raw, advisor) & raw[cfg["dataset"]["target"]["column"]].isin(cfg["dataset"]["filters"]["verdict_in"])
            & (created >= WINDOW[0]) & (created < WINDOW[1]))
    df = raw[mask].reset_index(drop=True)
    prepared, y, created_w = advisor.data_mod.prepare(df, cfg)
    parts = advisor.train_mod.make_split(prepared, y, created_w, cfg)
    ids = pd.to_numeric(df["EntityId"]).astype("int64")
    return {k: set(ids.loc[v]) for k, v in parts.items()}


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Наибольший порог, при котором доля найденных реальных инцидентов (recall) ≥ target."""
    pos = np.sort(p[y == 1])
    if not len(pos):
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[k])


def summary(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    return dict(n=int(len(y)), pos=int(tp + fn), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / max(tp + fn, 1), miss=fn / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                low_share=float((~pred).mean()), roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)))


def build_frame(cards: pd.DataFrame, proba: np.ndarray, part: str, rule_counts: pd.Series) -> pd.DataFrame:
    created = pd.to_datetime(cards["Создан"], format="ISO8601", utc=True)
    local = created.dt.tz_convert(LOCAL_TZ)
    out = pd.DataFrame({
        "id": pd.to_numeric(cards["EntityId"]).astype("int64").to_numpy(),
        "created": created.dt.tz_localize(None).to_numpy(),        # UTC, как окна курса
        "part": part,
        "verdict": cards["Вердикт"].to_numpy(),
        "y": cards["Вердикт"].map(LABELS).fillna(-1).astype("int8").to_numpy(),
        "proba": proba,
        "rule": rule_of(cards).to_numpy(),
        "inc_type": cards["Тип инцидента"].map(first_item).to_numpy(),
        "irp": cards["Пиктограмма IRP"].map(first_item).to_numpy(),
        "crit": cards["Критичность"].map(first_item).to_numpy(),
        "org": cards["Организация"].map(name_of).to_numpy(),
        "vendor": cards["Вендор из сием"].map(first_item).to_numpy(),
        "siem": cards["Источник инцидента"].map(first_item).to_numpy(),
        "group": cards["Группа устранения"].map(name_of).to_numpy(),
        "hour": local.dt.hour.astype("int8").to_numpy(),
        "dow": local.dt.dayofweek.astype("int8").to_numpy(),
        "has_vendor": cards["Вендор из сием"].notna().to_numpy(),
        "has_alert": cards["Алерт"].notna().to_numpy(),
    })
    out["rule_train_n"] = out["rule"].map(rule_counts).fillna(0).astype("int32")
    return out


def main(path: Path, course_dir: Path) -> None:
    advisor = CourseAdvisor(course_dir)
    print(f"Модель курса: {course_dir / 'models' / 'lgbm_gate1__v001'} · вычисление: {advisor.engine} "
          f"· порог τ = {advisor.threshold:.4f}")
    raw = load_raw(path, advisor)
    delta = advisor.self_check(raw)
    print(f"Самопроверка: 5 контрольных карточек совпали с эталоном, расхождение ≤ {delta:.1e}")

    split = window_split(raw, advisor)
    ids = pd.to_numeric(raw["EntityId"]).astype("int64")
    created = pd.to_datetime(raw["Создан"], format="ISO8601", utc=True)
    scope = in_scope(raw, advisor)
    test = raw[ids.isin(split["test"])]
    oper = raw[(created >= OPER_START) & scope]
    print(f"Сплит окна: train {len(split['train']):,} · val {len(split['val']):,} · test {len(split['test']):,}")
    print(f"Эксплуатация: {len(oper):,} карточек в области модели "
          f"(вне области: {int(((created >= OPER_START) & ~scope).sum())})")

    # частота правила в обучении — по «Правилу корреляции SIEM»; модель видит «Наименование»
    # (freq_Наименование), поэтому проверяем, что это одно и то же правило
    rules = rule_of(raw)
    same_rule = float((rules == raw["Наименование"])[rules.notna()].mean())
    print(f"«Правило корреляции SIEM» совпадает с «Наименованием» у {same_rule:.2%} карточек")
    rule_counts = rules[ids.isin(split["train"])].value_counts()
    p_test = advisor.predict_proba(test)
    p_oper = advisor.predict_proba(oper)
    scored = pd.concat([build_frame(test, p_test, "Приёмка", rule_counts),
                        build_frame(oper, p_oper, "Эксплуатация", rule_counts)], ignore_index=True)

    tau = advisor.threshold
    lab_t = scored[(scored["part"] == "Приёмка") & (scored["y"] >= 0)]
    lab_o = scored[(scored["part"] == "Эксплуатация") & (scored["y"] >= 0)]
    m_test = summary(lab_t["y"].to_numpy(), lab_t["proba"].to_numpy(), tau)
    m_oper = summary(lab_o["y"].to_numpy(), lab_o["proba"].to_numpy(), tau)
    same = all(m_test[k] == v for k, v in PUBLISHED_TEST.items())
    print(f"Приёмка (test-сплит): TP {m_test['tp']} / FP {m_test['fp']} / FN {m_test['fn']} / TN {m_test['tn']} — "
          + ("совпадает с карточкой модели" if same else f"НЕ совпадает с карточкой модели {PUBLISHED_TEST}"))
    for name, m in [("приёмка", m_test), ("эксплуатация", m_oper)]:
        print(f"{name:13s}: пропущено {m['fn']} из {m['pos']} ({m['miss']:.1%}), в низком приоритете "
              f"{m['low_share']:.1%}, точность {m['precision']:.1%}, ROC AUC {m['roc_auc']:.3f}")

    t99 = pick_threshold(lab_t["y"].to_numpy(), lab_t["proba"].to_numpy(), 0.99)
    tau_label = f"Рабочий порог модели τ = {tau:.4f}".replace(".", ",")
    policies = {tau_label: tau, "Находить 99 % инцидентов приёмки": t99, "Порог 0,5": 0.5}

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(AGG_DIR / "scored.parquet", index=False)
    meta = {
        "dataset": path.name,
        "model": "lgbm_gate1__v001", "engine": advisor.engine, "self_check_delta": delta,
        "run_id": advisor.run["run_id"],
        "periods": {
            "train": [str(WINDOW[0].date()), str((WINDOW[1] - pd.Timedelta(days=1)).date())],
            "val": [str(WINDOW[0].date()), str((WINDOW[1] - pd.Timedelta(days=1)).date())],
            "oper": [str(OPER_START.date()), str(created.max().date())],
        },
        "n_train": len(split["train"]),
        "pos_rate_train": float(raw.loc[ids.isin(split["train"]), "Вердикт"].eq("Решён").mean()),
        "threshold": tau, "target_miss": m_test["miss"], "target_recall": m_test["recall"],
        "policies": policies, "default_policy": tau_label, "rule_field_match": same_rule,
        "val_metrics": m_test, "oper_metrics": m_oper, "published_test_match": same,
    }
    (AGG_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово:", AGG_DIR.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="путь к iavd_ml_dataset.parquet")
    ap.add_argument("--course-dir", help="папка пакета модели курса IAVD_3.7_ML_team")
    args = ap.parse_args()
    main(find_raw(args.path), find_course_dir(args.course_dir))
