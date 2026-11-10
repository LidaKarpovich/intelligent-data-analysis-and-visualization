"""
report.py — текстовый отчёт к дашборду (REPORT.md).

На листах дашборда остаются графики и KPI-плитки, а выводы с цифрами собираются здесь —
из тех же функций (dashboard_insights.py), поэтому числа в отчёте и на дашборде совпадают.

Запуск из папки task_1 (после etl.py --full):
    python report.py              # отчёт за весь период -> REPORT.md
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_insights as ins


def _section(title: str, question: str, text: str) -> str:
    return f"### {question}\n\n*{title}.* {text}\n" if text else ""


def build(store: dd.Store, f: dd.Filters | None = None) -> str:
    f = f or dd.Filters()
    d = dd.apply_filters(store, f)
    if d.empty:
        return "# Отчёт\n\nНет карточек для выбранных фильтров.\n"
    dates = dd.scope_dates(store, f)
    hourly = dd.hourly_counts(store, f)
    period = f"{dd.fmt_date(d['date'].min())} — {dd.fmt_date(d['date'].max())}"
    parts = [
        "# SOC · триаж алертов SIEM: отчёт к дашборду",
        "",
        f"Период: **{period}**, {dd.fmt_int(len(d))} карточек инцидентов. "
        f"Отчёт собран скриптом `report.py` {date.today():%d.%m.%Y}; разделы повторяют страницы дашборда.",
        "",
        "## Главное за период",
        "",
        ins.summary(d, dates),
        "",
        "## Обзор",
        "",
        _section("Линейный график по дням", "Сколько карточек поступает каждый день",
                 ins.daily(d, dates, store.day_type_by_date)),
        _section("100-процентная столбчатая", "Как меняется исход карточек", ins.mix(d, ch.grain_label(d))),
        _section("Столбцы долей", "Какой вердикт получает карточка?", ins.verdicts(d)),
        "## Сравнение",
        "",
        _section("Линейчатая диаграмма", "Какие правила SIEM дают больше всего карточек", ins.top_rules(d)),
        _section("Дивергентная столбчатая", "Где «шума» больше среднего", ins.fp_deviation(d)),
        _section("Тепловая карта", "Когда поступают алерты: день недели × час", ins.heatmap(d)),
        "## Распределения",
        "",
    ]
    crit = d[d["crit"] == "Критическая"]
    parts.append(_section("Гистограмма", "Сколько ждут карточки до взятия в работу (критические)",
                          ins.histogram(d, crit, "Критическая")))
    x = hourly.to_numpy(float)
    if len(x) > 20 and x.std() > 0:
        r2 = stats.probplot(x, dist="norm")[1][2] ** 2
        parts.append(_section("QQ-график", "Подходит ли «правило трёх сигм» для порогов SIEM",
                              ins.qq(x, r2, False)))
    kde_lines = []
    for label, (col, is_log) in ch.KDE_FEATURES.items():
        ns = {c: int(pd.to_numeric(d.loc[d["cls"] == c, col], errors="coerce").notna().sum())
              for c in (dd.CLS_FP, dd.CLS_REAL)}
        txt = ins.kde(d, col, is_log, ns)
        if txt:
            kde_lines.append(f"- **{label}.** {txt}")
    if kde_lines:
        parts += ["### Чем реальные инциденты отличаются от False Positive", "",
                  "*KDE (сглаженная плотность), по каждому признаку.*", "", *kde_lines, ""]
    for by in ch.VIOLIN_GROUPS:
        col, groups = ch.violin_groups(d, by)
        parts.append(_section(f"Violin, группировка «{by}»", f"Какие группы ждут взятия в работу дольше: «{by}»",
                              ins.violin(d, col, groups)))
    parts.append(_section("Raincloud", "Насколько дольше живёт карточка после эскалации на L2?", ins.raincloud(d)))

    parts += ["## Потоки и плотность", ""]
    for label, col in ch.SANKEY_LEFT.items():
        parts.append(_section(f"Sankey, слева — «{label}»", f"Куда уходят карточки: «{label}» → вердикт",
                              ins.sankey(d, col)))
    prev, cur, n_days = dd.previous_window(store, f)
    parts.append(_section("Waterfall", "За счёт чего изменился поток карточек",
                          ins.waterfall(prev, cur, n_days, f.period not in store.period_dates)))
    for mode in ch.HEX_MODES:
        first = mode == ch.HEX_MODES[0]
        if first:
            t = d[["react_min", "solve_min"]].dropna()
            n = int(((t["react_min"] >= dd.INSTANT_MIN) & (t["solve_min"] >= dd.INSTANT_MIN)).sum())
        else:
            n = len(hourly)
        parts.append(_section(f"Hexbin, оси «{mode}»", ch.HEX_QUESTIONS[mode],
                              ins.hexbin(d, first, n, None if first else hourly)))
    parts.append(_section("Dot plot", "Какие адреса-источники порождают больше всего карточек?",
                          ins.dotplot(store.top_src)))
    parts.append(_section("Bubble chart", "Какие правила долго разбирают и при этом находят реальные инциденты?",
                          ins.bubble(d)))
    parts += [ABOUT, ""]
    text = "\n".join(p for p in parts if p is not None)
    return text.replace("\n\n\n", "\n\n").rstrip() + "\n"


ABOUT = """## Как читать дашборд и оговорки

Дашборд описывает операционную работу SOC по карточкам инцидентов `iavd_incidents_raw`. Все графики, кроме топа адресов-источников, пересчитываются по фильтрам в шапке: период, тип дня (рабочие / выходные и праздники по производственному календарю), источник и критичность. Ключевые числа каждой страницы — в KPI-плитках над графиками; плитки с красной полосой, розовой заливкой и значком ▲ — сигналы, на которые стоит обратить внимание.

**Исход карточки** — вердикт аналитика: «Решён» — реальный инцидент, «False Positive» — ложное срабатывание, «Не указан», «Дубликат» и редкие вердикты — прочее. Цвета исходов одинаковы на всех страницах: оранжевый — реальный инцидент, синий — False Positive, серый — прочее.

- **Доля False Positive — решение аналитика по карточке**, а не доля трудозатрат SOC: короткая FP-карточка обходится дешевле расследования.
- **Время** («Период взятия в работу», «Время решения») хранится в **миллисекундах** — проверено по разнице дат «Создан» и «Дата взятия в работу». Отрицательные значения и значения длиннее периода наблюдения считаются ошибками и исключены.
- **Около 31 % карточек берутся в работу «мгновенно»** (< 0,6 с) — их закрывает автоматика. На логарифмической шкале ноль не показать, поэтому медианы времени на дашборде считаются без них, а их доля вынесена в отдельную плитку.
- **Около 23 % карточек без вердикта.** Для периодов, где таких карточек много, доли FP и реальных инцидентов занижены.
- **Окна периодов** («7 / 30 / 90 / 180 дней») берутся из таблицы `dataframe.parquet` и заканчиваются в последний день данных; на выборке из `etl.py` без `--full` они сдвигаются к её концу. Waterfall сравнивает окно с таким же окном перед ним; для «Весь период» берётся окно «30 дней».
- Время суток и дни недели — по Москве (Europe/Moscow).
"""


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else dd.BASE / "REPORT.md"
    out.write_text(build(dd.load()), encoding="utf-8")
    print("Отчёт записан:", out)
