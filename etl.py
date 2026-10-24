"""
etl.py — сырой parquet -> data/agg/*.parquet (признаки и агрегаты для дашборда).

Запуск из папки проекта:
    python etl.py            # малый файл (1000 строк) — для отладки
    python etl.py --full     # большой файл (240 тыс. строк)

Большой файл читается по row group и только нужными колонками,
поэтому в память целиком он не попадает.
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent     # папка, где лежит etl.py (запускать можно откуда угодно)
DATA_DIR = BASE / "data"
AGG_DIR = DATA_DIR / "agg"
SMALL_NAME = "iavd_1000_incidents_raw_target.parquet"
FULL_NAME = "iavd_incidents_raw_target.parquet"


def find_raw(name: str) -> Path:
    """Ищем parquet внутри data/ (в том числе во вложенных папках)."""
    hits = sorted(DATA_DIR.rglob(name)) if DATA_DIR.exists() else []
    if not hits:
        found = [str(p.relative_to(BASE)) for p in DATA_DIR.rglob("*.parquet")] if DATA_DIR.exists() else []
        raise SystemExit(f"Не нашёл {name} в {DATA_DIR}\nНайденные parquet-файлы: {found or 'нет'}")
    return hits[0]

SEP = "~~||~~"          # разделитель элементов в мультизначных полях
TZ = "Europe/Moscow"    # часовой пояс SOC (час суток, день недели) — при необходимости поменять
MAX_MS = 5e10           # ~580 суток: длительность больше всего периода наблюдения = артефакт

# Fact_label: 1 = «Решён» (реальный инцидент), 0 = «False Positive», -1 = остальное (вне метрик)
LABEL_MAP = {"False Positive": 0, "Решен": 1}   # «ё» перед сопоставлением заменяется на «е»

CRIT = {"Отсутствует": 0, "Низкая": 1, "Средняя": 2, "Высокая": 3, "Критическая": 4}
CRIT_SRC = {"Low": 1, "Medium": 2, "High": 3, "Высокий": 3, "Высокая": 3}

RAW_COLS = [
    "EntityId", "Создан", "Вердикт", "Критичность", "Критичность инцидента в источнике",
    "Тип инцидента", "Источник инцидента", "Наименование", "Пиктограмма IRP", "Группа устранения",
    "Правило корреляции SIEM", "Теги", "categoryHigh", "categoryLow", "categoryMain",
    "object", "processName", "processParentName", "commandName", "Привилегии аккаунта",
    "Вендор из сием", "CMDLINE", "Описание инцидента", "Значение Инцидента (группировка)",
    "src", "dst", "Количество инцидентов",
    "Время первого события", "Время последнего события", "Дата создания в источнике",
    "Период взятия в работу", "Время решения",
]

# новое имя -> (исходная колонка, сколько самых частых значений оставить)
CATEGORICAL = {
    "rule": ("Наименование", 60),
    "inc_type": ("Тип инцидента", 40),
    "source": ("Источник инцидента", 6),
    "irp": ("Пиктограмма IRP", 20),
    "cat_main": ("categoryMain", 25),
    "cat_high": ("categoryHigh", 25),
    "cat_low": ("categoryLow", 25),
    "obj": ("object", 10),
    "proc": ("processName", 30),
    "proc_parent": ("processParentName", 30),
    "cmd_name": ("commandName", 30),
    "priv": ("Привилегии аккаунта", 10),
    "vendor": ("Вендор из сием", 10),
    "tag1": ("Теги", 20),
}


# ---------- вспомогательные функции ----------
def n_items(s: pd.Series) -> pd.Series:
    """Число элементов в мультизначном поле (пусто -> 0)."""
    return s.map(lambda x: x.count(SEP) + 1 if isinstance(x, str) and x else 0).astype("int16")


def first_item(s: pd.Series) -> pd.Series:
    """Первый элемент мультизначного поля (split по точному совпадению, не regex)."""
    return s.map(lambda x: x.split(SEP, 1)[0] if isinstance(x, str) and x else None)


def shannon(text) -> float:
    """Энтропия Шеннона строки (бит/символ); NaN для пустых."""
    if not isinstance(text, str) or not text:
        return np.nan
    text = text[:5000]
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def minutes_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a - b) в минутах; отрицательные значения считаем ошибкой данных -> NaN."""
    d = (a - b).dt.total_seconds() / 60
    return d.where(d >= 0)


def ms_to_min(s: pd.Series) -> pd.Series:
    """Миллисекунды -> минуты; отрицательные и «длиннее периода наблюдения» -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= 0) & (s <= MAX_MS)) / 60000


def top_k(s: pd.Series, k: int) -> pd.Series:
    """Оставить k самых частых значений, остальное -> 'other', пропуски -> 'missing'."""
    keep = s.value_counts().nlargest(k).index
    out = s.where(s.isin(keep), "other")
    return out.where(s.notna(), "missing")


# ---------- признаки одного куска файла ----------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = df["EntityId"]

    # время (в локальном часовом поясе SOC)
    created = df["Создан"].dt.tz_convert(TZ)
    out["created"] = created.dt.tz_localize(None)
    out["hour"] = created.dt.hour.astype("int8")
    out["dow"] = created.dt.dayofweek.astype("int8")

    # метка
    verdict = df["Вердикт"].str.replace("ё", "е", regex=False)
    out["verdict_txt"] = df["Вердикт"]
    out["y"] = verdict.map(LABEL_MAP).fillna(-1).astype("int8")

    # критичность
    out["crit"] = df["Критичность"]
    out["crit_ord"] = df["Критичность"].map(CRIT)
    out["crit_src_ord"] = df["Критичность инцидента в источнике"].map(CRIT_SRC)

    # масштаб карточки
    out["n_incidents"] = pd.to_numeric(df["Количество инцидентов"], errors="coerce")
    out["n_rules"] = n_items(df["Правило корреляции SIEM"])
    out["n_tags"] = n_items(df["Теги"])
    out["n_src"] = n_items(df["src"])
    out["n_dst"] = n_items(df["dst"])
    out["n_values"] = n_items(df["Значение Инцидента (группировка)"])
    out["desc_len"] = df["Описание инцидента"].fillna("").str.len().astype("int32")

    # времена события (известны на момент создания карточки)
    out["lag_min"] = minutes_between(df["Создан"], df["Дата создания в источнике"])
    out["attack_dur_min"] = minutes_between(df["Время последнего события"], df["Время первого события"])

    # командная строка
    cmd = df["CMDLINE"]
    out["cmd_len"] = cmd.map(lambda x: float(len(x)) if isinstance(x, str) and x else np.nan)
    out["cmd_entropy"] = cmd.map(shannon)
    out["cmd_n"] = n_items(cmd)
    out["cmd_enc"] = cmd.fillna("").str.contains("-enc", case=False, regex=False).astype("int8")

    # категории (топ-K считаем позже, по всему файлу)
    for new, (col, _) in CATEGORICAL.items():
        out[new] = first_item(df[col])

    # ТОЛЬКО для EDA — в модель не идут (известны после обработки карточки)
    out["group"] = df["Группа устранения"]
    out["react_min"] = ms_to_min(df["Период взятия в работу"])
    out["solve_min"] = ms_to_min(df["Время решения"])
    return out


def main(path: Path) -> None:
    pf = pq.ParquetFile(path)
    print(f"Читаю {path.name}: {pf.metadata.num_rows:,} строк, row groups: {pf.num_row_groups}")

    parts, src_counter = [], Counter()
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=RAW_COLS).to_pandas()
        parts.append(build_features(df))
        for items in df["src"].dropna():           # сколько карточек затронуло каждый src
            src_counter.update(set(items.split(SEP)))
        print(f"  row group {rg + 1}/{pf.num_row_groups}: {len(df):,} строк")

    feats = pd.concat(parts, ignore_index=True)

    # топ-K категорий считаем глобально
    for new, (_, k) in CATEGORICAL.items():
        feats[new + "_c"] = top_k(feats[new], k).astype("category")

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(AGG_DIR / "features.parquet", index=False)

    # дневная динамика по классам Fact_label
    feats["date"] = feats["created"].dt.normalize()
    daily = (feats.groupby("date")["y"].value_counts().unstack(fill_value=0)
             .reindex(columns=[-1, 0, 1], fill_value=0)
             .rename(columns={-1: "n_other", 0: "n_fp", 1: "n_real"}).reset_index())
    daily["n_total"] = daily[["n_other", "n_fp", "n_real"]].sum(axis=1)
    daily.to_parquet(AGG_DIR / "agg_daily.parquet", index=False)

    # число карточек по часам (для QQ-графика) — включая часы без карточек
    hourly = (feats.set_index("created").resample("h").size()
              .rename("n_cards").reset_index().rename(columns={"created": "hour_start"}))
    hourly.to_parquet(AGG_DIR / "agg_hourly.parquet", index=False)

    # топ-20 src по числу карточек (для Dot Plot)
    top_src = pd.DataFrame(src_counter.most_common(20), columns=["src", "n_cards"])
    top_src.to_parquet(AGG_DIR / "agg_top_src.parquet", index=False)

    print("\n=== Готово ===")
    print("Период:", feats["created"].min(), "—", feats["created"].max())
    print("Fact_label (-1 другое, 0 FP, 1 реальный):")
    print(feats["y"].value_counts().sort_index().to_string())
    lab = feats[feats["y"] >= 0]
    print(f"Доля положительного класса среди размеченных: {lab['y'].mean():.3f}")
    print("\nЗаполненность числовых признаков, %:")
    num_cols = ["lag_min", "attack_dur_min", "cmd_len", "n_incidents", "react_min", "solve_min"]
    print((feats[num_cols].notna().mean() * 100).round(1).to_string())
    print("\nreact_min / solve_min (минуты), квантили:")
    print(feats[["react_min", "solve_min"]].quantile([.5, .9, .99]).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="использовать большой файл")
    args = ap.parse_args()
    main(find_raw(FULL_NAME if args.full else SMALL_NAME))
