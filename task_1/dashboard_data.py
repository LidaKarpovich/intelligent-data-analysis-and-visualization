"""
dashboard_data.py — загрузка данных, фильтры и агрегаты для дашборда (без ML).

Источники (все уже лежат в проекте):
    data/agg/features.parquet     — карточки после etl.py (240 295 строк)
    data/agg/agg_top_src.parquet  — топ-20 src по числу карточек (считается в etl.py)
    data/dataframe.parquet        — таблица периодов: Дата -> Период (7/30/90/180 дней)
    data/calendar.parquet         — календарь: Дата -> Тип дня (рабочий / пятница / праздник / выходной)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
AGG = DATA / "agg"

# Колонки features.parquet, которые нужны дашборду (читаем только их — быстрее и меньше памяти)
FEATURE_COLS = [
    "id", "created", "hour", "dow", "verdict_txt", "y", "crit", "crit_ord", "n_rules",
    "lag_min", "attack_dur_min", "cmd_len", "cmd_entropy", "rule", "inc_type", "source", "irp",
    "group", "react_min", "solve_min",
]

# ---------------------------------------------------------------- справочники
ALL = "Все"
PERIODS = ["Весь период", "180 дней", "90 дней", "30 дней", "7 дней"]
DAY_TYPES = ["Все дни", "Рабочие", "Выходные и праздники"]
# calendar.parquet, поле «Тип дня»: 1 — рабочий (пн–чт), 2 — пятница / предпраздничный,
# 3 — праздник, 4 — выходной (проверено по дням недели)
DAY_TYPE_CODES = {"Рабочие": {1, 2}, "Выходные и праздники": {3, 4}}

CRIT_ORDER = ["Отсутствует", "Низкая", "Средняя", "Высокая", "Критическая"]
SOURCE_MAIN = ["MaxPatrol SIEM", "Ankey SIEM NG"]
SOURCES = [ALL] + SOURCE_MAIN + ["Ручной ввод и почта"]
GROUP_ORDER = ["L1", "L2", "ГИБ"]
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

# Классы исхода карточки (операционный вердикт аналитика, НЕ предсказание модели)
CLS_REAL, CLS_FP, CLS_NONE, CLS_OTHER = "Решён (реальный инцидент)", "False Positive", "Не указан", "Дубликат и прочие"
CLASS_ORDER = [CLS_REAL, CLS_FP, CLS_NONE, CLS_OTHER]

INSTANT_MIN = 0.01  # < 0,01 мин (0,6 с) — «мгновенно»: закрыто автоматикой; на лог-шкале не показать


# ---------------------------------------------------------------- форматирование (рус. локаль)
def fmt_int(n) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def fmt_num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_minutes(m: float) -> str:
    """Длительность в минутах -> человекочитаемо: «45 с», «2,2 мин», «3,4 ч», «5,1 сут»."""
    if m is None or not np.isfinite(m):
        return "—"
    if m < 1:
        return f"{m * 60:.0f} с"
    if m < 60:
        return f"{fmt_num(m, 1)} мин"
    if m < 1440:
        return f"{fmt_num(m / 60, 1)} ч"
    return f"{fmt_num(m / 1440, 1)} сут"


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d.%m.%Y")


def _as_naive_date(s: pd.Series) -> pd.Series:
    """Календарная дата без часового пояса. Наивный столбец не трогаем: tz_localize(None) на нём падает."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


# ---------------------------------------------------------------- загрузка
def _verdict_class(v: pd.Series) -> pd.Series:
    v = v.fillna("Не указан").str.replace("ё", "е", regex=False)
    out = np.select([v == "Решен", v == "False Positive", v == "Не указан"], [CLS_REAL, CLS_FP, CLS_NONE], CLS_OTHER)
    return pd.Series(pd.Categorical(out, categories=CLASS_ORDER), index=v.index)


def align_periods(period_dates: dict, data_end) -> dict:
    """Сдвигает окна «7/30/90/180 дней» так, чтобы они заканчивались в последний день данных.

    Таблица периодов построена для полного датасета и заканчивается 03.09.2025. Выборка
    из etl.py без --full заканчивается раньше, и без сдвига все окна оказались бы пустыми.
    На полных данных конец таблицы совпадает с концом данных, и окна не меняются.
    """
    ends = [max(v) for v in period_dates.values() if v]
    if not ends or pd.isna(data_end):
        return period_dates
    shift = pd.Timestamp(data_end).normalize() - max(ends)
    if shift == pd.Timedelta(0):
        return period_dates
    return {p: {d + shift for d in v} for p, v in period_dates.items()}


def load() -> "Store":
    feats = pd.read_parquet(AGG / "features.parquet", columns=FEATURE_COLS)
    feats["created"] = pd.to_datetime(feats["created"])
    feats["date"] = feats["created"].dt.normalize()
    feats["cls"] = _verdict_class(feats["verdict_txt"].astype("string"))
    feats["verdict_txt"] = feats["verdict_txt"].fillna("Не указан")
    feats["crit"] = pd.Categorical(feats["crit"], categories=CRIT_ORDER, ordered=True)
    feats["source_grp"] = np.where(feats["source"].isin(SOURCE_MAIN), feats["source"], "Ручной ввод и почта")
    for col in ["rule", "inc_type", "irp", "group"]:
        feats[col] = feats[col].fillna("не указано")

    # Таблица периодов: Дата -> Период (7 / 30 / 90 / 180 дней)
    per = pd.read_parquet(DATA / "dataframe.parquet")
    per["Дата"] = _as_naive_date(per["Дата"])
    period_dates = {p: set(g["Дата"]) for p, g in per.groupby("Период")}
    period_dates = align_periods(period_dates, feats["date"].max())

    # Календарь: Дата -> Тип дня. Даты вне календаря не считаем рабочими: тип остаётся пустым.
    cal = pd.read_parquet(DATA / "calendar.parquet", columns=["Дата", "Тип дня"])
    cal["Дата"] = _as_naive_date(cal["Дата"])
    feats["day_type"] = feats["date"].map(cal.set_index("Дата")["Тип дня"]).astype("Int8")

    feats = feats.sort_values("created", kind="stable")
    top_src = pd.read_parquet(AGG / "agg_top_src.parquet")
    # Календарь целиком, а не только даты, где уже есть карточки: пустой день нужного типа — ноль, не пропуск
    return Store(feats=feats, period_dates=period_dates, top_src=top_src,
                 day_type_by_date=cal.set_index("Дата")["Тип дня"])


@dataclass
class Store:
    feats: pd.DataFrame
    period_dates: dict
    top_src: pd.DataFrame
    day_type_by_date: pd.Series = field(default_factory=lambda: pd.Series(dtype="float"))


@dataclass
class Filters:
    period: str = PERIODS[0]
    day_type: str = DAY_TYPES[0]
    source: str = ALL
    crit: list = field(default_factory=lambda: list(CRIT_ORDER))


def apply_filters(store: Store, f: Filters) -> pd.DataFrame:
    d = store.feats
    m = np.ones(len(d), dtype=bool)
    if f.period in store.period_dates:
        m &= d["date"].isin(store.period_dates[f.period]).to_numpy()
    if f.day_type in DAY_TYPE_CODES:
        m &= d["day_type"].isin(DAY_TYPE_CODES[f.day_type]).to_numpy()
    if f.source != ALL:
        m &= (d["source_grp"] == f.source).to_numpy()
    crit = [c for c in (f.crit or []) if c in CRIT_ORDER]
    if not crit:
        return d.iloc[0:0]
    if len(crit) < len(CRIT_ORDER):
        m &= d["crit"].isin(crit).to_numpy()
    return d[m]


def scope_dates(store: Store, f: Filters) -> pd.DatetimeIndex:
    """Даты, которые проходят фильтр периода и типа дня.

    Источник и критичность дату не выкидывают: час без таких карточек — настоящий ноль.
    Даты без записи в календаре при выбранном типе дня не входят (их тип неизвестен).
    """
    if f.period in store.period_dates:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(store.period_dates[f.period])))).normalize()
    else:
        raw = pd.DatetimeIndex(pd.unique(store.feats["date"])).sort_values()
        if len(raw) == 0:
            return raw
        dates = pd.date_range(raw.min(), raw.max(), freq="D")
    if f.day_type not in DAY_TYPE_CODES:
        return pd.DatetimeIndex(dates)
    codes = DAY_TYPE_CODES[f.day_type]
    by_date = store.day_type_by_date
    keep = []
    for dt in dates:
        code = by_date.get(pd.Timestamp(dt), pd.NA)
        if pd.isna(code) or int(code) not in codes:
            continue
        keep.append(pd.Timestamp(dt))
    return pd.DatetimeIndex(keep)


def hourly_counts(store: Store, f: Filters) -> pd.Series:
    """Карточки по часам только для дат из scope_dates.

    Часы другого типа дня не заполняются нулём. Час нужного типа без карточек — ноль.
    """
    dates = scope_dates(store, f)
    if len(dates) == 0:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(day), periods=24, freq="h").to_numpy() for day in dates
    ]))
    d = apply_filters(store, f)
    if d.empty:
        return pd.Series(0, index=index, dtype=float)
    counts = d["created"].dt.floor("h").value_counts()
    return counts.reindex(index, fill_value=0).astype(float).sort_index()


def previous_window(store: Store, f: Filters) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Текущее окно и такое же по длине окно перед ним — для Waterfall.

    Длина окна — длина выбранного периода; для «Весь период» — 30 дней. Окно заканчивается
    в последний день с карточками (но не позже конца выбранного периода). Таблица периодов
    привязана к концу полного датасета, поэтому на выборке из etl.py без --full окна
    «последние 30 дней» по календарю оказались бы пустыми.
    """
    days = store.period_dates.get(f.period) or store.period_dates.get("30 дней")
    n_days = (max(days) - min(days)).days + 1 if days else 30
    base = apply_filters(store, Filters(PERIODS[0], f.day_type, f.source, f.crit))
    empty = base.iloc[0:0]
    if base.empty:
        return empty, empty, n_days
    end = base["date"].max()
    if f.period in store.period_dates:
        p_start, p_end = min(store.period_dates[f.period]), max(store.period_dates[f.period])
        if end < p_start:                          # в выбранном периоде карточек нет вовсе
            return empty, empty, n_days
        end = min(end, p_end)
    start = end - pd.Timedelta(days=n_days - 1)
    prev_start = start - pd.Timedelta(days=n_days)
    cur = base[(base["date"] >= start) & (base["date"] <= end)]
    prev = base[(base["date"] >= prev_start) & (base["date"] < start)]
    return prev, cur, n_days


def plotted_minutes(s: pd.Series) -> pd.Series:
    """Те же длительности, что попадают на логарифмическую шкалу: без пропусков и «мгновенных»."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    return v[v >= INSTANT_MIN]


MIN_GROUP = 30   # меньше 30 карточек — медиана группы слишком неустойчива для сравнения


@dataclass
class CritVsRest:
    """Медианы взятия в работу: критические карточки против карточек другой критичности."""
    n_crit: int
    n_rest: int
    crit: float = float("nan")
    rest: float = float("nan")

    @property
    def comparable(self) -> bool:
        return self.n_crit >= MIN_GROUP and self.n_rest >= MIN_GROUP

    @property
    def ratio(self) -> float:
        return self.crit / self.rest if self.comparable else float("nan")

    @property
    def slower(self) -> bool:
        """Критические заметно медленнее: множитель после округления до 0,1 больше 1,0."""
        return self.comparable and round(self.ratio, 1) > 1.0


def crit_vs_rest(d: pd.DataFrame) -> CritVsRest:
    """Сравнивает критические карточки с остальными, а не со всеми: иначе критические сравниваются
    сами с собой, и при узком фильтре критичности множитель стремится к 1."""
    is_crit = (d["crit"] == "Критическая").fillna(False).to_numpy(bool)
    crit = plotted_minutes(d.loc[is_crit, "react_min"])
    rest = plotted_minutes(d.loc[~is_crit, "react_min"])
    out = CritVsRest(len(crit), len(rest))
    if len(crit):
        out.crit = float(crit.median())
    if len(rest):
        out.rest = float(rest.median())
    return out


def log_minutes(s: pd.Series) -> np.ndarray:
    """Минуты -> log10; пропуски и «мгновенные» (< 0,01 мин) отбрасываем (лог-шкала их не покажет)."""
    v = plotted_minutes(s).to_numpy(float)
    return np.log10(v)


def instant_share(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float((v < INSTANT_MIN).mean()) if len(v) else 0.0
