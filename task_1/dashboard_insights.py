"""
dashboard_insights.py — аналитические выводы к графикам (Markdown-строки, пересчитываются по фильтрам).

Каждая функция получает отфильтрованные карточки и возвращает 1–3 предложения:
что видно на графике, насколько это велико и что из этого следует для SOC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import dashboard_data as dd

B = "**"


def _b(x) -> str:
    return f"{B}{x}{B}"


def _ratio(a: float, b: float) -> str:
    """«в 3,2 раза» с защитой от деления на ноль."""
    if not b or not np.isfinite(a / b):
        return "—"
    r = a / b
    return f"в {dd.fmt_num(r, 1)} раза" if r < 10 else f"в {dd.fmt_int(r)} раз"


def plural(n: float, forms: tuple[str, str, str]) -> str:
    """Согласование с числом: plural(1, ("карточка", "карточки", "карточек")) -> «карточка»."""
    n = abs(int(round(n)))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


CARDS = ("карточка", "карточки", "карточек")
RULES = ("правило", "правила", "правил")
HOURS = ("часовой интервал", "часовых интервала", "часовых интервалов")


def _n(n: float, forms: tuple[str, str, str], bold: bool = False) -> str:
    num = dd.fmt_int(n)
    return f"{_b(num) if bold else num} {plural(n, forms)}"


def _t(m: float) -> str:
    """Длительность для текста: «мгновенно» вместо «0 с»."""
    return "мгновенно" if m < dd.INSTANT_MIN else dd.fmt_minutes(m)


def _share(d: pd.DataFrame, cls: str) -> float:
    return float((d["cls"] == cls).mean()) if len(d) else 0.0


def _name(cls: str) -> str:
    return cls.split(" (")[0]


def _short(s, n: int = 48) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _trend(d: pd.DataFrame) -> tuple[float, float] | None:
    """Средний поток в день: первая и последняя четверть периода (нужно ≥ 8 дней)."""
    s = d.groupby("date").size()
    if len(s) < 8:
        return None
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0)
    q = max(len(s) // 4, 2)
    return float(s.iloc[:q].mean()), float(s.iloc[-q:].mean())


# ================================================================ Обзор
def summary(d: pd.DataFrame) -> str:
    """Главное за период — 5 тезисов для руководителя SOC."""
    if d.empty:
        return "Нет карточек для выбранных фильтров."
    out = []
    days = d["date"].nunique()
    tr = _trend(d)
    line = f"Поток: {_n(len(d), CARDS, True)} за {dd.fmt_int(days)} дн., в среднем {dd.fmt_int(len(d) / days)} в день"
    if tr and tr[0] > 0:
        a, b = tr
        chg = (b - a) / a
        if abs(chg) >= 0.1:
            line += (f"; к концу периода поток {'вырос' if chg > 0 else 'снизился'} с {dd.fmt_int(a)} до "
                     f"{dd.fmt_int(b)} в день ({'+' if chg > 0 else '−'}{dd.fmt_pct(abs(chg), 0)})")
        else:
            line += "; поток стабилен"
    out.append(line + ".")

    fp, real, none = _share(d, dd.CLS_FP), _share(d, dd.CLS_REAL), _share(d, dd.CLS_NONE)
    if real > 0:
        out.append(f"Шум: {_b(dd.fmt_pct(fp))} карточек — False Positive, реальных инцидентов {dd.fmt_pct(real)}: "
                   f"на один подтверждённый инцидент приходится {_b(dd.fmt_num(fp / real, 1))} ложных срабатываний.")
    else:
        out.append(f"Шум: {_b(dd.fmt_pct(fp))} карточек — False Positive, подтверждённых инцидентов нет.")

    vc = d["rule"].value_counts()
    lead = vc.index[0]
    lead_d = d[d["rule"] == lead]
    lead_cls = lead_d["cls"].value_counts(normalize=True)
    out.append(f"Концентрация: одно правило {_b(_short(lead, 40))} даёт {dd.fmt_pct(vc.iloc[0] / len(d))} потока, "
               f"топ-10 правил — {dd.fmt_pct(vc.head(10).sum() / len(d))}; у лидера "
               f"{dd.fmt_pct(lead_cls.iloc[0])} карточек — «{_name(lead_cls.index[0])}».")

    if none >= 0.05:
        nv = d.loc[d["cls"] == dd.CLS_NONE, "rule"].value_counts()
        out.append(f"Разметка: {_b(dd.fmt_pct(none))} карточек без вердикта, "
                   f"{dd.fmt_pct(nv.iloc[0] / nv.sum())} из них — правило {_short(nv.index[0], 40)}. "
                   "Пока они не размечены, доли FP и инцидентов оценены только по закрытым карточкам.")

    react = d["react_min"].dropna()
    crit = d.loc[d["crit"] == "Критическая", "react_min"].dropna()
    if len(react) and len(crit) < 30:
        out.append(f"Реакция: медиана взятия в работу {_b(_t(react.median()))}, 90 % карточек берутся быстрее "
                   f"{_t(react.quantile(0.9))}, дольше суток ждут {dd.fmt_pct((react > 1440).mean())}.")
    if len(react) and len(crit) >= 30:
        mc, ma = crit.median(), react.median()
        word = "медленнее" if mc > ma else "быстрее"
        line = f"Реакция: медиана взятия в работу {_b(_t(ma))}; критические карточки берут {word} — {_b(_t(mc))}"
        if mc > 0 and ma > 0:
            line += f" ({_ratio(max(mc, ma), min(mc, ma))})"
        line += ". Критичность пока не ускоряет реакцию — её стоит учитывать в очереди." if mc > ma else "."
        out.append(line)
    return "\n".join(f"- {s}" for s in out)


def daily(d: pd.DataFrame) -> str:
    if d.empty:
        return ""
    s = d.groupby("date").size()
    wk = d[d["dow"] < 5].groupby("date").size().mean()
    we = d[d["dow"] >= 5].groupby("date").size().mean()
    txt = (f"В среднем {_b(dd.fmt_int(s.mean()))} {plural(s.mean(), CARDS)} в день, пик — {_b(dd.fmt_int(s.max()))} "
           f"({dd.fmt_date(s.idxmax())}, {_ratio(s.max(), s.mean())} выше среднего).")
    if np.isfinite(wk) and np.isfinite(we) and we > 0:
        txt += f" По будням поток {_ratio(wk, we)} больше, чем в выходные ({dd.fmt_int(wk)} против {dd.fmt_int(we)})."
    spikes = (s > s.mean() + 3 * s.std()).sum()
    if spikes:
        txt += (f" Дней-всплесков выше «среднее + 3σ» — {spikes}: обычно это массовые сработки одного правила, "
                "их стоит разобрать отдельно.")
    return txt


def mix(d: pd.DataFrame, freq_label: str) -> str:
    if d.empty:
        return ""
    freq = {"дням": "D", "неделям": "W", "месяцам": "M"}[freq_label]
    key = d["date"].dt.to_period(freq).dt.start_time
    t = pd.crosstab(key, d["cls"])
    t = t[t.sum(axis=1) >= 30]
    if len(t) < 2:
        return "Для сравнения нужно хотя бы два интервала с ≥ 30 карточками."
    sh = t.div(t.sum(axis=1), axis=0)
    fmt = "%d.%m.%Y" if freq != "M" else "%m.%Y"
    real = sh.get(dd.CLS_REAL, pd.Series(0, index=sh.index))
    hi, lo = real.idxmax(), real.idxmin()
    spread = real[hi] - real[lo]
    txt = (f"Доли по {freq_label}. Доля реальных инцидентов — от {dd.fmt_pct(real[lo])} ({lo.strftime(fmt)}) "
           f"до {_b(dd.fmt_pct(real[hi]))} ({hi.strftime(fmt)})")
    txt += (": разброс больше 15 п.п., состав потока заметно меняется — стоит смотреть, какие правила включались "
            "и выключались." if spread > 0.15 else ": состав потока стабилен.")
    none = sh.get(dd.CLS_NONE, pd.Series(0, index=sh.index))
    bad = none[none > 0.2]
    if len(bad):
        txt += (f" В {len(bad)} {plural(len(bad), ('интервале', 'интервалах', 'интервалах'))} больше 20 % карточек без вердикта (максимум "
                f"{dd.fmt_pct(bad.max())}, {bad.idxmax().strftime(fmt)}): там доли FP и инцидентов занижены.")
    wk, we = d[d["dow"] < 5], d[d["dow"] >= 5]
    if len(wk) >= 100 and len(we) >= 100:
        rk, re_ = _share(wk, dd.CLS_REAL), _share(we, dd.CLS_REAL)
        if abs(re_ - rk) >= 0.05:
            txt += (f" В выходные доля реальных инцидентов {'выше' if re_ > rk else 'ниже'} "
                    f"({dd.fmt_pct(re_)} против {dd.fmt_pct(rk)} в будни)"
                    + (": шумовые правила срабатывают в основном в рабочее время." if re_ > rk else "."))
    return txt + " Интервалы с < 30 карточками скрыты."


def verdicts(d: pd.DataFrame) -> str:
    if d.empty:
        return ""
    fp, real, none = _share(d, dd.CLS_FP), _share(d, dd.CLS_REAL), _share(d, dd.CLS_NONE)
    txt = f"{_b(dd.fmt_pct(fp))} карточек закрыты как False Positive"
    txt += f", подтверждены как инцидент — {dd.fmt_pct(real)}" if real else ""
    txt += f", без вердикта — {dd.fmt_pct(none)}." if none else "."
    txt += " Это решение аналитика по карточке, а не доля трудозатрат: короткие FP-карточки дешевле расследований."
    return txt


# ================================================================ Сравнение
def top_rules(d: pd.DataFrame) -> str:
    if d.empty:
        return ""
    vc = d["rule"].value_counts()
    top = vc.head(10).index
    g = d[d["rule"].isin(top)].groupby("rule")["cls"].value_counts(normalize=True).unstack(fill_value=0)
    g = g.reindex(columns=dd.CLASS_ORDER, fill_value=0)
    txt = f"10 правил из {dd.fmt_int(len(vc))} дают {_b(dd.fmt_pct(vc.head(10).sum() / len(d)))} всех карточек."
    noisy = g[(g[dd.CLS_REAL] < 0.01)].index
    if len(noisy):
        share = vc[noisy].sum() / len(d)
        txt += (f" Из них {_n(len(noisy), RULES)} почти не {'даёт' if plural(len(noisy), RULES) == 'правило' else 'дают'} подтверждённых инцидентов (< 1 %) и создают "
                f"{_b(dd.fmt_pct(share))} потока — первые кандидаты на тюнинг или автозакрытие.")
    best = g[dd.CLS_REAL].idxmax()
    if g.loc[best, dd.CLS_REAL] >= 0.05:
        txt += (f" Самое «полезное» правило топа — {_short(best, 40)}: "
                f"{dd.fmt_pct(g.loc[best, dd.CLS_REAL])} его карточек — реальные инциденты.")
    return txt


def fp_deviation(d: pd.DataFrame) -> str:
    g = d.groupby("irp")["cls"].agg(n="size", fp=lambda s: (s == dd.CLS_FP).mean(),
                                    real=lambda s: (s == dd.CLS_REAL).mean())
    g = g[g["n"] >= 100]
    if len(g) < 2:
        return "Недостаточно категорий с ≥ 100 карточками."
    avg = _share(d, dd.CLS_FP)
    hi, lo = g["fp"].idxmax(), g["fp"].idxmin()
    txt = (f"Больше всего «шума» — {_b(hi)}: {dd.fmt_pct(g.loc[hi, 'fp'])} FP при среднем {dd.fmt_pct(avg)}. "
           f"Меньше всего — {_b(lo)} ({dd.fmt_pct(g.loc[lo, 'fp'])}).")
    if g.loc[lo, "real"] < 0.5 * (1 - g.loc[lo, "fp"]):
        txt += (f" Но низкая доля FP здесь не значит «больше атак»: реальных инцидентов лишь "
                f"{dd.fmt_pct(g.loc[lo, 'real'])}, остальное — без вердикта или дубликаты.")
    above = g[g["fp"] > avg]
    txt += f" Выше среднего — {len(above)} из {len(g)} категорий ({dd.fmt_pct(above['n'].sum() / g['n'].sum())} потока)."
    return txt


def heatmap(d: pd.DataFrame) -> str:
    if d.empty:
        return ""
    t = pd.crosstab(d["dow"], d["hour"])
    i, j = np.unravel_index(np.argmax(t.values), t.shape)
    work = (d["dow"] < 5) & d["hour"].between(9, 17)
    night = d["hour"].between(0, 5).mean()
    txt = (f"Пик нагрузки — {_b(f'{dd.WEEKDAYS[t.index[i]]}, {t.columns[j]:02d}:00')}. "
           f"В рабочие часы будней (09–18) поступает {_b(dd.fmt_pct(work.mean()))} карточек, "
           f"ночью (00–06) — {dd.fmt_pct(night)}.")
    if (d["dow"] < 5).mean() < 0.05:              # выбраны только выходные
        return (f"Пик нагрузки — {_b(f'{dd.WEEKDAYS[t.index[i]]}, {t.columns[j]:02d}:00')}; "
                f"ночью (00–06) поступает {dd.fmt_pct(night)} карточек.")
    txt += (" Поток привязан к рабочему дню пользователей — дневную смену L1 стоит усиливать, "
            "но ночное дежурство всё равно нужно." if work.mean() > 0.5 else
            " Нагрузка распределена по суткам равномерно — нужен полноценный сменный график.")
    return txt


# ================================================================ Распределения
def histogram(d_all: pd.DataFrame, sub: pd.DataFrame, crit: str) -> str:
    react = sub["react_min"].dropna()
    if len(react) < 10:
        who = "карточек" if crit == dd.ALL else f"карточек с критичностью «{crit}»"
        return (f"За выбранный период меньше 10 {who} с заполненным временем взятия в работу — выберите другой "
                "уровень критичности или период.")
    med, p90 = react.median(), react.quantile(0.9)
    tail = (react > 1440).mean()
    txt = f"{_n(len(react), CARDS)}. Медиана — {_b(_t(med))}, 90 % берутся быстрее {_t(p90)}"
    txt += (f", но {_b(dd.fmt_pct(tail))} ждут дольше суток — длинный правый хвост: это карточки, о которых "
            "забыли в очереди." if tail >= 0.01 else "; дольше суток почти никто не ждёт.")
    if crit != dd.ALL:
        base = d_all["react_min"].dropna()
        if len(base) and base.median() > 0 and med > 0:
            word = "дольше" if med > base.median() else "быстрее"
            txt += (f" Это {_ratio(max(med, base.median()), min(med, base.median()))} {word}, чем по всем карточкам "
                    f"({_t(base.median())}).")
    txt += (f" Ещё {dd.fmt_pct(dd.instant_share(react))} взяты «мгновенно» (< 0,6 с, автоматика) и на логарифмической "
            "шкале не показаны.")
    return txt


def qq(x: np.ndarray, r2: float, log: bool) -> str:
    """x — число карточек в час (без преобразования)."""
    v = np.log1p(x) if log else x
    mu, sd = v.mean(), v.std()
    above = (v > mu + 3 * sd).mean()
    scale = "ln(1 + x)" if log else "исходной шкале"
    if r2 >= 0.97:
        txt = (f"{_n(len(x), HOURS)}, R² = {_b(dd.fmt_num(r2, 3))} на {scale}: точки почти лежат на прямой, "
               "нормальный закон описывает поток приемлемо.")
    elif r2 >= 0.9:
        txt = (f"{_n(len(x), HOURS)}, R² = {_b(dd.fmt_num(r2, 3))} на {scale}: в центре точки близки к прямой, "
               "расхождения — только на краях распределения.")
    else:
        txt = (f"{_n(len(x), HOURS)}, R² = {_b(dd.fmt_num(r2, 3))} на {scale}: точки уходят от прямой — "
               "тяжёлый правый хвост, всплески сильнее, чем допускает нормальный закон.")
    if above > 0.002:
        txt += (f" Часов выше «среднее + 3σ» — {_b(dd.fmt_pct(above, 2))} вместо ожидаемых 0,13 %: статический "
                "порог по «трём сигмам» будет срабатывать чаще расчётного.")
    elif above == 0:
        txt += " Ни один час не превысил «среднее + 3σ» — на коротком окне такая оценка ненадёжна."
    else:
        txt += f" Часов выше «среднее + 3σ» — {dd.fmt_pct(above, 2)}, близко к ожидаемым 0,13 %."
    if not log and len(x) > 20:
        r2_log = stats.probplot(np.log1p(x), dist="norm")[1][2] ** 2
        if r2_log > r2 + 0.05:
            txt += (f" После ln(1 + x) R² растёт до {dd.fmt_num(r2_log, 3)} — пороги лучше считать "
                    "на логарифмической шкале или по процентилям (переключатель справа).")
    return txt


def kde(d: pd.DataFrame, col: str, is_log: bool, ns: dict) -> str:
    vals, meds, zeros, zmax = {}, [], [], 0.0
    for cls in [dd.CLS_REAL, dd.CLS_FP]:
        v = pd.to_numeric(d.loc[d["cls"] == cls, col], errors="coerce").dropna()
        if is_log and len(v):
            zeros.append(f"{_name(cls)} — {dd.fmt_pct((v <= 0).mean())}")
            zmax = max(zmax, float((v <= 0).mean()))
            v = v[v > 0]
        if len(v):
            vals[cls] = v.to_numpy(float)
            m = v.median()
            val = (dd.fmt_minutes(m) if col.endswith("_min") else _n(m, ("символ", "символа", "символов")) if col == "cmd_len"
                   else f"{dd.fmt_num(m, 2)} бит/символ")
            meds.append(f"{_name(cls)} — {_b(val)}")
    if not meds:
        return ""
    txt = "Медианы: " + "; ".join(meds) + "."
    if len(vals) == 2 and min(len(v) for v in vals.values()) >= 50:
        D = stats.ks_2samp(*vals.values()).statistic
        level = "слабо" if D < 0.1 else "умеренно" if D < 0.3 else "сильно"
        txt += (f" Распределения различаются {_b(level)} (статистика Колмогорова–Смирнова D = {dd.fmt_num(D, 2)}): "
                + ("сам по себе признак почти не отделяет инциденты от шума."
                   if D < 0.1 else "признак полезен для приоритизации карточек."))
    if is_log and zmax >= 0.001:
        txt += " Нулевые значения на лог-шкале не показаны, их доля: " + "; ".join(zeros) + "."
    small = [c for c, n in ns.items() if n < 50]
    if small:
        txt += (" В классе «" + "», «".join(_name(c) for c in small) + "» меньше 50 значений — кривая не строится, "
                "сравнение медиан на такой выборке ненадёжно.")
    return txt


def violin(d: pd.DataFrame, col: str, groups: list) -> str:
    med, slow = {}, {}
    for g in groups:
        v = d.loc[d[col] == g, "react_min"].dropna()
        if len(dd.log_minutes(v)) >= 30:               # те же группы, что показаны на графике
            med[g] = v.median()
            slow[g] = (v > 1440).mean()
    if len(med) < 2:
        return "Контур — плотность распределения, белый ящик — медиана и квартили."
    s = pd.Series(med)
    hi, lo = s.idxmax(), s.idxmin()
    if s[lo] > 0 and s[hi] / s[lo] < 1.3:
        txt = (f"Группы почти не различаются: медианы от {_t(s[lo])} до {_b(_t(s[hi]))}. "
               "Скорость реакции от этой группировки не зависит.")
    else:
        txt = (f"Дольше всех ждут карточки «{_short(hi, 40)}» — медиана {_b(_t(s[hi]))}, "
               f"быстрее всех «{_short(lo, 40)}» — {_t(s[lo])}.")
    w = pd.Series(slow).idxmax()
    if slow[w] >= 0.005:
        txt += f" Больше всего «забытых» карточек (дольше суток) в группе «{_short(w, 40)}» — {_b(dd.fmt_pct(slow[w]))}."
    return txt + " Контур — плотность, белый ящик — медиана и квартили; группы с < 30 значениями скрыты."


def raincloud(d: pd.DataFrame) -> str:
    med, day = {}, {}
    for g in dd.GROUP_ORDER:
        v = d.loc[d["group"] == g, "solve_min"].dropna()
        if len(v) >= 5:
            med[g], day[g] = v.median(), (v > 1440).mean()
    if not med:
        return ""
    txt = "Медиана времени решения: " + "; ".join(f"{g} — {_b(_t(m))}" for g, m in med.items()) + "."
    if "L1" in med and "L2" in med and med["L1"] > 0:
        txt += f" Карточка, ушедшая на L2, решается {_ratio(med['L2'], med['L1'])} дольше"
        txt += (f"; {dd.fmt_pct(day['L2'])} таких карточек живут больше суток." if day["L2"] >= 0.005 else ".")
        txt += " Эскалация — дорогой путь, поэтому важно, чтобы на L2 уходили только реальные инциденты."
    if "ГИБ" in med:
        n = len(dd.log_minutes(d.loc[d["group"] == "ГИБ", "solve_min"]))
        txt += (f" У ГИБ на графике всего {_n(n, CARDS)} — по ним важнее смотреть отдельные точки, "
                "чем форму облака.")
    return txt


# ================================================================ Потоки и плотность
def sankey(d: pd.DataFrame, col: str) -> str:
    """Выводы только по категориям, которые видны на графике (без «Прочие»)."""
    if d.empty:
        return ""
    left = d[col].astype(str)
    top = left.value_counts().head(8 if col in ("rule", "irp") else 10).index
    d = d[left.isin(top)]
    t = pd.crosstab(d[col].astype(str), d["cls"]).reindex(columns=dd.CLASS_ORDER, fill_value=0)
    t = t[t.sum(axis=1) >= max(30, 0.01 * len(d))]
    if len(t) < 2:
        return ""
    sh = t.div(t.sum(axis=1), axis=0)
    noisy, useful, unk = sh[dd.CLS_FP].idxmax(), sh[dd.CLS_REAL].idxmax(), sh[dd.CLS_NONE].idxmax()
    txt = (f"Самая «шумная» категория — {_b(_short(noisy))}: {dd.fmt_pct(sh.loc[noisy, dd.CLS_FP])} её карточек — "
           "False Positive.")
    if useful != noisy:
        txt += (f" Чаще всего подтверждаются инциденты в «{_short(useful)}» "
                f"({_b(dd.fmt_pct(sh.loc[useful, dd.CLS_REAL]))}).")
    else:
        txt += (f" Она же лидирует и по подтверждённым инцидентам ({_b(dd.fmt_pct(sh.loc[useful, dd.CLS_REAL]))}) — "
                "в этой группировке категории различаются в основном долей карточек без вердикта.")
    if sh.loc[unk, dd.CLS_NONE] > 0.3:
        txt += f" В «{_short(unk)}» {dd.fmt_pct(sh.loc[unk, dd.CLS_NONE])} карточек остаются без вердикта."
    return txt


def waterfall(prev: pd.DataFrame, cur: pd.DataFrame, n_days: int, whole_period: bool) -> str:
    if prev.empty and cur.empty:
        return ""
    a, b = prev["irp"].value_counts(), cur["irp"].value_counts()
    idx = a.index.union(b.index)
    dlt = b.reindex(idx, fill_value=0) - a.reindex(idx, fill_value=0)
    delta = len(cur) - len(prev)
    rel = f" ({'+' if delta >= 0 else '−'}{dd.fmt_pct(abs(delta) / len(prev), 0)})" if len(prev) else ""
    txt = (f"За последние {n_days} дн. поток {'вырос' if delta >= 0 else 'снизился'} на "
           f"{_b(dd.fmt_int(abs(delta)))} карточек{rel} к предыдущим {n_days} дн.")
    up, down = dlt[dlt > 0].sort_values(ascending=False), dlt[dlt < 0].sort_values()
    main, other = (up, down) if delta >= 0 else (down, up)
    if len(main):
        k, v = main.index[0], main.iloc[0]
        share = f", {dd.fmt_pct(abs(v / delta), 0)} всего изменения" if delta else ""
        txt += f" Главный вклад — {_b(k)} ({'+' if v > 0 else '−'}{dd.fmt_int(abs(v))}{share})."
    if len(other) and abs(other.iloc[0]) >= 0.05 * max(abs(delta), 1):
        k, v = other.index[0], other.iloc[0]
        txt += f" В обратную сторону сильнее всего изменилась «{k}» ({'+' if v > 0 else '−'}{dd.fmt_int(abs(v))})."
    if whole_period:
        txt += " Для «Весь период» сравниваются последние 30 дней с предыдущими."
    return txt


def hexbin(d: pd.DataFrame, mode_first: bool, n: int) -> str:
    if mode_first:
        t = d[["react_min", "solve_min"]].dropna()
        t = t[(t["react_min"] >= dd.INSTANT_MIN) & (t["solve_min"] >= dd.INSTANT_MIN)]
        if len(t) < 100:
            return ""
        rho = stats.spearmanr(t["react_min"], t["solve_min"]).statistic
        level = "слабая" if abs(rho) < 0.3 else "умеренная" if abs(rho) < 0.6 else "сильная"
        h, xe, ye = np.histogram2d(np.log10(t["react_min"]), np.log10(t["solve_min"]), bins=30)
        i, j = np.unravel_index(np.argmax(h), h.shape)
        x, y = 10 ** ((xe[i] + xe[i + 1]) / 2), 10 ** ((ye[j] + ye[j + 1]) / 2)
        return (f"{_n(n, CARDS)} — на диаграмме рассеяния они слились бы в пятно. Связь между временем взятия "
                f"и временем решения {_b(level)} (ρ Спирмена = {dd.fmt_num(rho, 2)})"
                + (": долгое ожидание в очереди не означает долгого разбора." if abs(rho) < 0.3 else
                   ": карточки, которые долго ждут, обычно и разбираются дольше.")
                + f" Самая плотная сота — взятие ≈ {dd.fmt_minutes(x)}, решение ≈ {dd.fmt_minutes(y)}.")
    s = d.set_index("created").resample("h").size()
    if len(s) < 100:
        return ""
    by_h = s.groupby(s.index.hour)
    p95 = by_h.quantile(0.95)
    med = by_h.median()
    hh = int(p95.idxmax())
    return (f"{_n(n, HOURS)} (точка = один час). Самые сильные всплески — около "
            f"{_b(f'{hh:02d}:00')}: в 5 % таких часов приходит больше {_n(p95[hh], CARDS)} при медиане "
            f"{dd.fmt_int(med[hh])}. Планировать смену по среднему недостаточно — нужен запас на пики.")


def dotplot(top_src: pd.DataFrame) -> str:
    s = top_src.sort_values("n_cards", ascending=False)
    private = s["src"].str.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")
    g = s["src"].str.startswith("142.25")
    txt = (f"Лидер — {_b(s['src'].iloc[0])}: {_n(s['n_cards'].iloc[0], CARDS)}. "
           f"Внешних адресов в топе — {_b(int((~private).sum()))} из {len(s)}")
    if g.sum() >= 3:
        txt += (f", из них {int(g.sum())} из подсети 142.250.0.0/15 (вероятно, сервисы Google) — "
                "стоит проверить и при необходимости внести в белый список.")
    else:
        txt += "."
    return txt + " Агрегат за весь период: фильтры шапки к нему не применяются."


def bubble(d: pd.DataFrame) -> str:
    g = d.groupby("rule").agg(n=("cls", "size"), solve=("solve_min", "median"),
                              real=("cls", lambda s: (s == dd.CLS_REAL).mean()))
    g = g[g["n"] >= 30].sort_values("n", ascending=False).head(20).dropna()
    if len(g) < 3:
        return ""
    best = g["real"].idxmax()
    zero = g[g["real"] < 0.01]
    txt = (f"Пузырёк — правило из топ-20, размер — число карточек. Чаще всего подтверждается "
           f"{_b(_short(best, 40))} ({dd.fmt_pct(g.loc[best, 'real'])} карточек — инциденты).")
    if len(zero):
        txt += (f" {_n(len(zero), RULES).capitalize()} почти не дают инцидентов (< 1 %), но создают "
                f"{_n(zero['n'].sum(), CARDS, True)} — они лежат на нижней границе графика: это кандидаты на "
                "тюнинг или автозакрытие.")
    rho = stats.spearmanr(g["solve"], g["real"]).statistic
    if np.isfinite(rho) and abs(rho) >= 0.3:
        txt += (f" Чем выше доля реальных инцидентов, тем дольше разбор (ρ = {dd.fmt_num(rho, 2)}) — настоящие "
                "инциденты требуют расследования." if rho > 0 else
                f" Правила с высокой долей инцидентов разбираются быстрее (ρ = {dd.fmt_num(rho, 2)}).")
    return txt
