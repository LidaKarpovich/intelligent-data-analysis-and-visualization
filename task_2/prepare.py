"""
prepare.py — сырой iavd_ml_dataset.parquet -> признаки -> модель -> оценки для дашборда.

Запуск из папки task_2:
    python prepare.py                       # ищет датасет в ../data_1, data_1 и data
    python prepare.py --path путь/к/файлу.parquet

Схема эксперимента («модель в эксплуатации»):
    обучение      карточки до 01.03.2025          — модель учится на вердиктах аналитиков
    приёмка       01.03.2025 — 30.04.2025          — выбираем порог: модель должна находить
                                                     не меньше TARGET_RECALL реальных инцидентов
    эксплуатация  с 01.05.2025 до конца выгрузки   — модель с зафиксированным порогом сортирует
                                                     очередь; её оценки сверяем с вердиктами SOC

Модель видит только то, что известно в момент создания карточки (правило, критичность, категория,
поля события, командная строка). Вердикт, решение, комментарии, меры и времена обработки
в признаки не идут — это утечка ответа.

Что получается в data/agg/:
    scored.parquet  — карточки приёмки и эксплуатации: вердикт, оценка модели, срезы для анализа ошибок
    meta.json       — периоды, порог, метрики приёмки, список признаков
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
AGG_DIR = BASE / "data" / "agg"
RAW_NAME = "iavd_ml_dataset.parquet"
SEARCH_DIRS = [BASE / "data", BASE / "data_1", BASE.parent / "data_1", BASE.parent / "data"]

TZ = "Europe/Moscow"
TRAIN_END = pd.Timestamp("2025-03-01")      # начало приёмки
OPER_START = pd.Timestamp("2025-05-01")     # начало эксплуатации
TARGET_RECALL = 0.95                        # на приёмке модель должна находить ≥ 95 % реальных инцидентов
POLICIES = [0.90, 0.95, 0.99]               # варианты порога для «что если» на дашборде
SEED = 42

# 1 = «Решён» (реальный инцидент), 0 = «False Positive», остальные вердикты в метрики не входят
LABEL_MAP = {"False Positive": 0, "Решен": 1}
CRIT = {"Низкая": 1, "Средняя": 2, "Высокая": 3, "Критическая": 4}
CRIT_SRC = {"Low": 1, "Medium": 2, "High": 3}

RAW_COLS = [
    "EntityId", "Дата создания", "Вердикт", "Критичность", "Критичность инцидента в источнике",
    "Тип инцидента", "Источник инцидента", "Наименование", "Пиктограмма IRP", "Организация",
    "Конвейер SIEM", "Вендор из сием", "Теги", "categoryHigh", "categoryLow", "categoryMain", "object",
    "processName", "Имя родителя процесса", "commandName", "Привилегии аккаунта", "CMDLINE",
    "Описание инцидента", "Значение Инцидента (группировка)", "Правило корреляции SIEM", "Алерт",
    "src", "dst", "FilePath", "Время первого события", "Время последнего события", "Группа устранения",
]

# новое имя -> (исходная колонка, сколько самых частых значений оставить; остальное -> other)
CATEGORICAL = {
    "rule": ("Наименование", 80),
    "inc_type": ("Тип инцидента", 40),
    "siem": ("Источник инцидента", 4),
    "irp": ("Пиктограмма IRP", 15),
    "org": ("Организация", 8),
    "pipeline": ("Конвейер SIEM", 10),
    "vendor": ("Вендор из сием", 12),
    "cat_main": ("categoryMain", 25),
    "cat_high": ("categoryHigh", 25),
    "cat_low": ("categoryLow", 25),
    "obj": ("object", 20),
    "proc": ("processName", 30),
    "proc_parent": ("Имя родителя процесса", 30),
    "cmd_name": ("commandName", 30),
    "priv": ("Привилегии аккаунта", 6),
    "tag1": ("Теги", 15),
}
NUM = ["crit_ord", "crit_src_ord", "hour", "dow", "attack_dur_min", "n_rules", "n_tags", "n_src", "n_dst",
       "n_values", "n_alerts", "n_files", "desc_len", "cmd_len", "cmd_entropy", "cmd_n", "cmd_enc"]
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
              l2_regularization=1.0, early_stopping=False, random_state=SEED)


# ---------------------------------------------------------------- разбор полей
def items(x) -> list[str]:
    """Мультизначное поле: JSON-список строк ('["a", "b"]') -> list; одиночная строка -> [строка]."""
    if not isinstance(x, str) or not x:
        return []
    s = x.strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
            return [str(i) for i in v if i not in (None, "")] if isinstance(v, list) else [str(v)]
        except ValueError:
            pass
    return [s]


def name_of(x) -> str | None:
    """JSON-объект с ключом name ('{"name": "L1"}') -> 'L1'; прочее -> как есть."""
    if not isinstance(x, str) or not x:
        return None
    if x.startswith("{"):
        try:
            return json.loads(x).get("name")
        except ValueError:
            return None
    return x


def shannon(text) -> float:
    """Энтропия Шеннона строки (бит/символ); NaN для пустых."""
    if not isinstance(text, str) or not text:
        return np.nan
    text = text[:5000]
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def top_k(s: pd.Series, keep: list[str]) -> pd.Series:
    out = s.where(s.isin(keep), "other")
    return out.where(s.notna(), "missing")


# ---------------------------------------------------------------- признаки одного куска
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = pd.to_numeric(df["EntityId"], errors="coerce").astype("int64")
    created = pd.to_datetime(df["Дата создания"], utc=True).dt.tz_convert(TZ)
    out["created"] = created.dt.tz_localize(None)
    out["hour"] = created.dt.hour.astype("int8")
    out["dow"] = created.dt.dayofweek.astype("int8")

    out["verdict"] = df["Вердикт"]
    v = df["Вердикт"].str.replace("ё", "е", regex=False)
    out["y"] = v.map(LABEL_MAP).fillna(-1).astype("int8")

    out["crit"] = df["Критичность"]
    out["crit_ord"] = df["Критичность"].map(CRIT)
    out["crit_src_ord"] = df["Критичность инцидента в источнике"].map(CRIT_SRC)

    lists = {c: df[c].map(items) for c in ["Правило корреляции SIEM", "Теги", "src", "dst",
                                           "Значение Инцидента (группировка)", "Алерт", "FilePath", "CMDLINE"]}
    out["n_rules"] = lists["Правило корреляции SIEM"].map(len).astype("int16")
    out["n_tags"] = lists["Теги"].map(len).astype("int16")
    out["n_src"] = lists["src"].map(len).astype("int16")
    out["n_dst"] = lists["dst"].map(len).astype("int16")
    out["n_values"] = lists["Значение Инцидента (группировка)"].map(len).astype("int16")
    out["n_alerts"] = lists["Алерт"].map(len).astype("int16")
    out["n_files"] = lists["FilePath"].map(len).astype("int16")
    out["desc_len"] = df["Описание инцидента"].fillna("").str.len().astype("int32")

    first, last = to_utc(df["Время первого события"]), to_utc(df["Время последнего события"])
    dur = (last - first).dt.total_seconds() / 60
    out["attack_dur_min"] = dur.where(dur >= 0)

    cmd = lists["CMDLINE"].map(lambda xs: "\n".join(xs) if xs else None)
    out["cmd_len"] = cmd.map(lambda x: float(len(x)) if isinstance(x, str) and x else np.nan)
    out["cmd_entropy"] = cmd.map(shannon)
    out["cmd_n"] = lists["CMDLINE"].map(len).astype("int16")
    out["cmd_enc"] = cmd.fillna("").str.contains("-enc", case=False, regex=False).astype("int8")

    for new, (col, _) in CATEGORICAL.items():
        if new == "org":
            out[new] = df[col].map(name_of)
        else:
            out[new] = df[col].map(lambda x: (items(x) or [None])[0])
    # только для анализа ошибок — в модель не идёт (известно после назначения)
    out["group"] = df["Группа устранения"].map(name_of)
    return out


def find_raw(path: str | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Нет файла {p}")
        return p
    for d in SEARCH_DIRS:
        if (d / RAW_NAME).exists():
            return d / RAW_NAME
    raise SystemExit(f"Не нашёл {RAW_NAME} в: " + ", ".join(str(d) for d in SEARCH_DIRS))


def load_features(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    print(f"Читаю {path.name}: {pf.metadata.num_rows:,} строк, row groups: {pf.num_row_groups}")
    parts = []
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=RAW_COLS).to_pandas()
        parts.append(build_features(df))
        print(f"  row group {rg + 1}/{pf.num_row_groups}: {len(df):,} строк")
    return pd.concat(parts, ignore_index=True).sort_values("created", kind="stable").reset_index(drop=True)


def assign_part(created: pd.Series) -> pd.Series:
    return pd.Series(np.select([created < TRAIN_END, created < OPER_START], ["train", "val"], "oper"),
                     index=created.index)


def encode(feats: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Категории: топ-K значений считаем ТОЛЬКО по обучению; новые значения потом -> other."""
    cat_cols = []
    for new, (_, k) in CATEGORICAL.items():
        keep = feats.loc[train_mask, new].value_counts().nlargest(k).index.tolist()
        lab = top_k(feats[new], keep)
        cats = sorted(set(keep)) + ["other", "missing"]
        feats[new + "_c"] = pd.Categorical(lab, categories=cats).codes.astype("int16")
        cat_cols.append(new + "_c")
    return feats, cat_cols


def pick_threshold(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Наибольший порог, при котором доля найденных реальных инцидентов (recall) ≥ target."""
    pos = np.sort(p[y == 1])
    if not len(pos):
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))      # сколько инцидентов можно оставить ниже порога
    return float(pos[k])


def main(path: Path) -> None:
    feats = load_features(path)
    feats["part"] = assign_part(feats["created"])
    train_mask = feats["part"].eq("train")
    feats, cat_cols = encode(feats, train_mask)
    features = NUM + cat_cols

    lab = feats["y"] >= 0
    tr = feats[train_mask & lab]
    va = feats[feats["part"].eq("val") & lab]
    op = feats[feats["part"].eq("oper") & lab]
    for name, d in [("обучение", tr), ("приёмка", va), ("эксплуатация", op)]:
        print(f"{name:13s}: {len(d):7,} размеченных карточек, доля реальных инцидентов {d['y'].mean():.3f}, "
              f"{d['created'].min():%d.%m.%Y} — {d['created'].max():%d.%m.%Y}")

    cat_mask = [f in cat_cols for f in features]
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, **PARAMS)
    model.fit(tr[features], tr["y"])

    scored = feats[~train_mask].copy()
    scored["proba"] = model.predict_proba(scored[features])[:, 1]
    p_va = scored.loc[va.index, "proba"].to_numpy()
    thr = pick_threshold(va["y"].to_numpy(), p_va, TARGET_RECALL)

    def summary(y, p) -> dict:
        pred = p >= thr
        tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
        return dict(n=int(len(y)), pos=int((y == 1).sum()), tp=tp, fn=fn, fp=fp, tn=tn,
                    recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                    low_share=float((~pred).mean()), roc_auc=float(roc_auc_score(y, p)),
                    pr_auc=float(average_precision_score(y, p)))

    m_va = summary(va["y"].to_numpy(), p_va)
    m_op = summary(op["y"].to_numpy(), scored.loc[op.index, "proba"].to_numpy())
    print(f"\nПорог (recall на приёмке ≥ {TARGET_RECALL:.0%}): {thr:.4f}")
    for name, m in [("приёмка", m_va), ("эксплуатация", m_op)]:
        print(f"{name:13s}: найдено {m['recall']:.1%} инцидентов (пропущено {m['fn']} из {m['pos']}), "
              f"точность {m['precision']:.1%}, в низком приоритете {m['low_share']:.1%} карточек, "
              f"ROC AUC {m['roc_auc']:.3f}, PR AUC {m['pr_auc']:.3f}")

    # сколько карточек каждого правила модель видела при обучении (новые правила — зона риска)
    seen = feats.loc[train_mask, "rule"].value_counts()
    scored["rule_train_n"] = scored["rule"].map(seen).fillna(0).astype("int32")
    scored["part"] = scored["part"].map({"val": "Приёмка", "oper": "Эксплуатация"})

    keep = ["id", "created", "part", "verdict", "y", "proba", "rule", "inc_type", "irp", "crit", "org",
            "vendor", "siem", "group", "hour", "dow", "rule_train_n", "cmd_len", "attack_dur_min"]
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    scored[keep].to_parquet(AGG_DIR / "scored.parquet", index=False)

    meta = {
        "dataset": path.name,
        "periods": {
            "train": [str(feats["created"].min().date()), str((TRAIN_END - pd.Timedelta(days=1)).date())],
            "val": [str(TRAIN_END.date()), str((OPER_START - pd.Timedelta(days=1)).date())],
            "oper": [str(OPER_START.date()), str(feats["created"].max().date())],
        },
        "n_train": int(len(tr)), "pos_rate_train": float(tr["y"].mean()),
        "target_recall": TARGET_RECALL, "threshold": thr,
        "policies": {f"{t:.2f}": pick_threshold(va["y"].to_numpy(), p_va, t) for t in POLICIES},
        "val_metrics": m_va, "oper_metrics": m_op,
        "features": features, "params": PARAMS,
    }
    (AGG_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово:", AGG_DIR.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="путь к iavd_ml_dataset.parquet")
    main(find_raw(ap.parse_args().path))
