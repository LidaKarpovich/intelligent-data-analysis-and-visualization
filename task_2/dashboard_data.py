"""
dashboard_data.py — загрузка оценок модели, фильтры и расчёт метрик для дашборда «Можно ли доверять модели».

Термины (одинаковые на дашборде, в отчёте и в тестах):
    реальный инцидент   — вердикт аналитика «Решён» (y = 1)
    False Positive      — вердикт «False Positive» (y = 0)
    в приоритет         — оценка модели ≥ порога: карточка идёт в первую очередь
    низкий приоритет    — оценка ниже порога: модель считает карточку ложной
    пропуск             — реальный инцидент в низком приоритете (FN)
    доля пропусков      — FN / (TP + FN) = 1 − recall; главная метрика для SOC
Карточки с другими вердиктами (дубликат, спам, без вердикта) модель тоже оценивает,
но сверить их не с чем — в метрики они не входят.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
AGG = BASE / "data" / "agg"

ALL = "Все"
VAL, OPER = "Приёмка", "Эксплуатация"
PERIOD_ALL = "Весь период эксплуатации"
CRIT_ORDER = ["Низкая", "Средняя", "Высокая", "Критическая"]
MONTHS_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
             "октябрь", "ноябрь", "декабрь"]
MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре", "октябре",
               "ноябре", "декабре"]          # «в мае»
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
              "ноября", "декабря"]           # «с июля»
MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

MIN_POS = 20      # меньше 20 реальных инцидентов в срезе — доля пропусков слишком неустойчива
MIN_NEG = 100     # то же для доли FP в приоритете
Z = 1.96          # 95 %-ный доверительный интервал


# ---------------------------------------------------------------- форматирование по-русски
def fmt_int(n) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f} %".replace(".", ",")


def fmt_num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d.%m.%Y")


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское согласование с числом: plural(91, ("пропуск", "пропуска", "пропусков")) -> «пропуск»."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def month_in(p: pd.Period) -> str:
    return f"в {MONTHS_PREP[p.month - 1]} {p.year}"


def month_since(p: pd.Period) -> str:
    return f"с {MONTHS_GEN[p.month - 1]} {p.year}"


def month_label(p: pd.Period, short: bool = False) -> str:
    names = MONTHS_SHORT if short else MONTHS_RU
    return f"{names[p.month - 1]} {p.year}" if not short else f"{names[p.month - 1]} {str(p.year)[2:]}"


# ---------------------------------------------------------------- данные
@dataclass
class Store:
    cards: pd.DataFrame          # все оценённые карточки приёмки и эксплуатации
    meta: dict
    policies: dict = field(default_factory=dict)   # подпись политики -> порог

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])


def policy_label(target: float) -> str:
    return f"Находить {target * 100:.0f} % инцидентов"


def load(agg: Path = AGG) -> Store:
    path = agg / "scored.parquet"
    if not path.exists():
        raise SystemExit(f"Нет {path}. Сначала запустите: python prepare.py")
    cards = pd.read_parquet(path)
    meta = json.loads((agg / "meta.json").read_text(encoding="utf-8"))
    return Store(cards=enrich(cards), meta=meta, policies=_policies(meta))


def _policies(meta: dict) -> dict:
    pol = {policy_label(float(k)): float(v) for k, v in sorted(meta.get("policies", {}).items())}
    return pol or {policy_label(meta["target_recall"]): float(meta["threshold"])}


def enrich(cards: pd.DataFrame) -> pd.DataFrame:
    d = cards.copy()
    d["created"] = pd.to_datetime(d["created"])
    d["month"] = d["created"].dt.to_period("M")
    d["org"] = d["org"].fillna("не указана")
    d["crit"] = pd.Categorical(d["crit"], categories=CRIT_ORDER, ordered=True)
    d["labeled"] = d["y"] >= 0
    return d


def period_options(store: Store) -> list[str]:
    months = sorted(store.cards.loc[store.cards["part"] == OPER, "month"].unique())
    return [PERIOD_ALL] + [month_label(m).capitalize() for m in months]


def org_options(store: Store) -> list[str]:
    o = store.cards.loc[store.cards["part"] == OPER, "org"].value_counts()
    return [ALL] + o.index.tolist()


def default_policy(store: Store) -> str:
    target = policy_label(store.meta["target_recall"])
    return target if target in store.policies else next(iter(store.policies))


@dataclass
class Filters:
    period: str = PERIOD_ALL
    org: str = ALL
    crit: list = field(default_factory=lambda: list(CRIT_ORDER))
    policy: str = ""


def threshold_of(store: Store, f: Filters) -> float:
    return store.policies.get(f.policy, store.threshold)


def selected_month(store: Store, f: Filters) -> pd.Period | None:
    if f.period == PERIOD_ALL:
        return None
    for m in store.cards["month"].unique():
        if month_label(m).capitalize() == f.period:
            return m
    return None


def apply_filters(store: Store, f: Filters, part: str | None = OPER, by_period: bool = True) -> pd.DataFrame:
    """Срез карточек. part=None — приёмка и эксплуатация вместе. Период касается только эксплуатации."""
    d = store.cards
    mask = pd.Series(True, index=d.index)
    if part:
        mask &= d["part"] == part
    if f.org != ALL:
        mask &= d["org"] == f.org
    crit = list(f.crit or [])
    if crit and len(crit) < len(CRIT_ORDER):
        mask &= d["crit"].isin(crit)
    m = selected_month(store, f)
    if by_period and m is not None:
        mask &= (d["month"] == m) | (d["part"] != OPER)
    out = d[mask].copy()
    out["pred"] = out["proba"] >= threshold_of(store, f)
    return out


def with_pred(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    out = d.copy()
    out["pred"] = out["proba"] >= thr
    return out


# ---------------------------------------------------------------- метрики
@dataclass
class Confusion:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0

    @property
    def pos(self) -> int:
        return self.tp + self.fn

    @property
    def neg(self) -> int:
        return self.fp + self.tn

    @property
    def n(self) -> int:
        return self.pos + self.neg

    @property
    def recall(self) -> float:
        return self.tp / self.pos if self.pos else np.nan

    @property
    def miss(self) -> float:
        return self.fn / self.pos if self.pos else np.nan

    @property
    def specificity(self) -> float:
        return self.tn / self.neg if self.neg else np.nan

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else np.nan

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else np.nan

    @property
    def low_share(self) -> float:
        return (self.fn + self.tn) / self.n if self.n else np.nan


def confusion(d: pd.DataFrame) -> Confusion:
    lab = d[d["y"] >= 0]
    y, p = lab["y"].to_numpy() == 1, lab["pred"].to_numpy(dtype=bool)
    return Confusion(tp=int((y & p).sum()), fn=int((y & ~p).sum()), fp=int((~y & p).sum()), tn=int((~y & ~p).sum()))


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """95 %-ный интервал Уилсона для доли k/n (устойчив при малых n и долях около 0)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC через ранги (Манна — Уитни): вероятность, что случайный инцидент оценён выше случайного FP."""
    y = np.asarray(y) == 1
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """Метрики по месяцам (только размеченные карточки)."""
    lab = d[d["y"] >= 0]
    rows = []
    for m, g in lab.groupby("month", sort=True):
        c = confusion(g)
        lo, hi = wilson(c.fn, c.pos)
        rows.append(dict(month=m, part=g["part"].iloc[0], n=c.n, pos=c.pos, tp=c.tp, fn=c.fn, fp=c.fp, tn=c.tn,
                         recall=c.recall, miss=c.miss, miss_lo=lo, miss_hi=hi, spec=c.specificity,
                         precision=c.precision, low_share=c.low_share, inc_rate=c.pos / c.n if c.n else np.nan,
                         auc=auc(g["y"], g["proba"]), new_rule=float((g["rule_train_n"] < 100).mean())))
    return pd.DataFrame(rows)


def gains(d: pd.DataFrame, points: int = 201) -> pd.DataFrame:
    """Кривая поиска: разбираем очередь сверху вниз по оценке модели.
    x — доля разобранных карточек, y — доля найденных реальных инцидентов."""
    lab = d[d["y"] >= 0].sort_values("proba", ascending=False, kind="stable")
    if lab.empty or lab["y"].sum() == 0:
        return pd.DataFrame(columns=["x", "y"])
    hits = np.r_[0, np.cumsum(lab["y"].to_numpy())] / lab["y"].sum()
    xs = np.arange(len(lab) + 1) / len(lab)
    idx = np.unique(np.linspace(0, len(lab), points).round().astype(int))
    return pd.DataFrame({"x": xs[idx], "y": hits[idx]})


CAL_EDGES = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001]


def calibration(d: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Корзины по оценке модели: средняя оценка против фактической доли реальных инцидентов."""
    lab = d[d["y"] >= 0]
    if lab.empty:
        return pd.DataFrame(columns=["bin", "pred", "actual", "n", "lo", "hi"])
    b = pd.cut(lab["proba"], CAL_EDGES, right=False)
    g = lab.groupby(b, observed=True).agg(pred=("proba", "mean"), actual=("y", "mean"), n=("y", "size"),
                                          k=("y", "sum")).reset_index().rename(columns={"proba": "bin"})
    g = g[g["n"] >= min_n].copy()
    ci = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


# ---------------------------------------------------------------- срезы для поиска систематических ошибок
def _rule_bucket(n: pd.Series) -> pd.Series:
    return pd.cut(n, [-1, 0, 99, 999, np.inf],
                  labels=["не было в обучении", "редкое: 1–99", "обычное: 100–999", "частое: 1000+"]).astype(str)


def _hour_bucket(h: pd.Series) -> pd.Series:
    return pd.cut(h, [-1, 6, 9, 18, 23], labels=["ночь 0–6 ч", "утро 7–9 ч", "день 10–18 ч", "вечер 19–23 ч"]).astype(str)


SEGMENTS = {
    "Правило в обучении": lambda d: _rule_bucket(d["rule_train_n"]),
    "Организация": lambda d: d["org"].astype(str),
    "Критичность": lambda d: d["crit"].astype(str),
    "Категория IRP": lambda d: d["irp"].fillna("не указана").astype(str),
    "Тип инцидента": lambda d: d["inc_type"].fillna("не указан").astype(str),
    "Правило SIEM": lambda d: d["rule"].fillna("не указано").astype(str),
    "Время суток": lambda d: _hour_bucket(d["hour"]),
}
SEGMENT_ORDERED = {"Правило в обучении", "Критичность", "Время суток"}   # у уровней есть естественный порядок
METRICS = {
    "Пропуски реальных инцидентов": "miss",
    "Лишняя работа: FP в приоритете": "fp_rate",
}
OTHER = "прочие"


def segment_table(d: pd.DataFrame, by: str, metric: str = "miss", top: int = 12) -> pd.DataFrame:
    """Доля ошибок по уровням среза с 95 %-ным интервалом.
    miss    — пропущенные / все реальные инциденты уровня;
    fp_rate — FP в приоритете / все FP уровня.
    Уровни с малым числом наблюдений сливаются в «прочие»; systematic = интервал целиком выше общего уровня."""
    lab = d[d["y"] >= 0].copy()
    if lab.empty:
        return pd.DataFrame()
    lab["seg"] = SEGMENTS[by](lab).to_numpy()
    if metric == "miss":
        base, k_col, minimum = lab[lab["y"] == 1], "fn", MIN_POS
        base = base.assign(err=~base["pred"])
    else:
        base, k_col, minimum = lab[lab["y"] == 0], "fp", MIN_NEG
        base = base.assign(err=base["pred"])
    if base.empty:
        return pd.DataFrame()
    counts = base["seg"].value_counts()
    keep = counts[counts >= minimum].index[:top]
    base["seg"] = base["seg"].where(base["seg"].isin(keep), OTHER)
    g = base.groupby("seg").agg(n=("err", "size"), k=("err", "sum")).reset_index()
    g = g[g["n"] >= minimum]
    if g.empty:
        return g
    overall = base["err"].mean()
    g["rate"] = g["k"] / g["n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    g["overall"] = overall
    g["systematic"] = g["lo"] > overall
    g["better"] = g["hi"] < overall
    g = g.rename(columns={"k": k_col})
    if by in SEGMENT_ORDERED:
        order = {v: i for i, v in enumerate(_order_for(by))}
        g["o"] = g["seg"].map(lambda s: order.get(s, 99))
        g = g.sort_values("o").drop(columns="o")
    else:
        g = g.sort_values("rate", ascending=False)
    return g.reset_index(drop=True)


def _order_for(by: str) -> list[str]:
    if by == "Правило в обучении":
        return ["не было в обучении", "редкое: 1–99", "обычное: 100–999", "частое: 1000+"]
    if by == "Критичность":
        return CRIT_ORDER
    return ["ночь 0–6 ч", "утро 7–9 ч", "день 10–18 ч", "вечер 19–23 ч"]


def near_threshold_share(d: pd.DataFrame, thr: float, factor: float = 2.0) -> tuple[int, int]:
    """Сколько пропусков «почти пойманы»: оценка не ниже thr / factor. -> (почти пойманные, все пропуски)."""
    miss = d[(d["y"] == 1) & ~d["pred"]]
    return int((miss["proba"] >= thr / factor).sum()), len(miss)
