"""Синтетические проверки логики дашборда. Не требуют parquet и не запускают Taipy."""
import unittest
from datetime import datetime

import pandas as pd

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_insights as ins


def _cards(rows: list[tuple]) -> pd.DataFrame:
    """rows: (created, day_type, source, crit)."""
    created = [pd.Timestamp(r[0]) for r in rows]
    frame = pd.DataFrame({
        "created": created,
        "date": [c.normalize() for c in created],
        "day_type": [r[1] for r in rows],
        "source_grp": [r[2] for r in rows],
        "crit": pd.Categorical([r[3] for r in rows], categories=dd.CRIT_ORDER, ordered=True),
        "cls": dd.CLS_FP,
        "dow": [c.dayofweek for c in created],
        "hour": [c.hour for c in created],
        "rule": "rule-a",
        "irp": "irp-a",
        "react_min": 10.0,
        "solve_min": 20.0,
        "verdict_txt": "False Positive",
    })
    return frame


def _store(frame: pd.DataFrame, calendar: dict, periods: dict | None = None) -> dd.Store:
    by_date = pd.Series({pd.Timestamp(k): v for k, v in calendar.items()})
    return dd.Store(feats=frame.sort_values("created"), period_dates=periods or {},
                    top_src=pd.DataFrame(columns=["src", "n_cards"]), day_type_by_date=by_date)


class HourlyCountsTest(unittest.TestCase):
    def test_weekend_filter_does_not_invent_weekday_zeros(self):
        # Две субботы и воскресенье между ними. Раньше resample дописывал нули на будни.
        rows = [
            (datetime(2024, 1, 6, 12, 0), 4, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 6, 12, 30), 4, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 13, 12, 0), 4, "MaxPatrol SIEM", "Низкая"),
        ]
        calendar = {}
        for day in pd.date_range("2024-01-06", "2024-01-13", freq="D"):
            calendar[day] = 4 if day.dayofweek >= 5 else 1
        store = _store(_cards(rows), calendar)
        counts = dd.hourly_counts(store, dd.Filters(day_type="Выходные и праздники"))
        self.assertFalse((counts.index.dayofweek < 5).any())
        # 6–7 и 13 января: два выходных + воскресенье 7-го. 13-е суббота, 14-е вне диапазона карточек? 
        # scope для «весь период» — от первой до последней даты в данных (6–13), из них только тип 4.
        weekend_days = [d for d in pd.date_range("2024-01-06", "2024-01-13") if d.dayofweek >= 5]
        self.assertEqual(len(counts), len(weekend_days) * 24)
        self.assertEqual(counts.loc[pd.Timestamp("2024-01-06 12:00")], 2)
        self.assertEqual(counts.loc[pd.Timestamp("2024-01-07 00:00")], 0)
        self.assertNotIn(pd.Timestamp("2024-01-08 10:00"), counts.index)


class CalendarGapTest(unittest.TestCase):
    def test_naive_calendar_dates_do_not_need_a_timezone(self):
        out = dd._as_naive_date(pd.Series(["2024-01-01", "2024-01-02"]))
        self.assertIsNone(out.dt.tz)
        self.assertEqual(out.iloc[0], pd.Timestamp("2024-01-01"))
        aware = pd.Series(pd.to_datetime(["2024-01-01"], utc=True))
        self.assertEqual(dd._as_naive_date(aware).iloc[0], pd.Timestamp("2024-01-01"))

    def test_unknown_day_is_not_a_workday(self):
        frame = _cards([(datetime(2024, 1, 1, 10), 1, "MaxPatrol SIEM", "Высокая")])
        frame["day_type"] = pd.Series([pd.NA], dtype="Int8")
        store = _store(frame, {})
        self.assertEqual(len(dd.apply_filters(store, dd.Filters(day_type="Все дни"))), 1)
        self.assertEqual(len(dd.apply_filters(store, dd.Filters(day_type="Рабочие"))), 0)
        self.assertEqual(len(dd.scope_dates(store, dd.Filters(day_type="Рабочие"))), 0)
        self.assertEqual(len(dd.scope_dates(store, dd.Filters(day_type="Все дни"))), 1)


class CalendarDayTypeInsightTest(unittest.TestCase):
    def test_holiday_monday_is_not_called_a_workday(self):
        rows = [
            (datetime(2024, 1, 1, 11), 3, "MaxPatrol SIEM", "Высокая"),
            *[(datetime(2024, 1, 2, 11), 1, "MaxPatrol SIEM", "Высокая") for _ in range(9)],
        ]
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        types = pd.Series({pd.Timestamp("2024-01-01"): 3, pd.Timestamp("2024-01-02"): 1})
        text = ins.daily(_cards(rows), dates, types)
        self.assertIn("По будням", text)
        self.assertIn("9", text)

    def test_heatmap_does_not_call_a_holiday_office_hours(self):
        rows = [(datetime(2024, 1, 1, 11), 3, "MaxPatrol SIEM", "Высокая") for _ in range(5)]
        text = ins.heatmap(_cards(rows))
        self.assertNotIn("рабочие часы", text)


class PlottedMedianTest(unittest.TestCase):
    def test_violin_text_uses_the_median_on_the_chart(self):
        rows = []
        for g, slow in (("A", 30.0), ("B", 10.0)):
            rows += [{"bucket": g, "react_min": 0.0} for _ in range(40)]
            rows += [{"bucket": g, "react_min": slow} for _ in range(40)]
        frame = pd.DataFrame(rows)
        text = ins.violin(frame, "bucket", ["A", "B"])
        self.assertIn("30,0", text)
        self.assertIn("10,0", text)
        self.assertNotIn("15,0", text)

    def test_histogram_line_is_the_plotted_median(self):
        frame = pd.DataFrame({"react_min": [0.0] * 20 + [30.0] * 20})
        fig = ch.fig_histogram(frame)
        labels = [a.text for a in fig.layout.annotations]
        self.assertTrue(any("30,0" in t for t in labels))
        self.assertFalse(any("15,0" in t for t in labels))


class EmptyDayAverageTest(unittest.TestCase):
    def test_daily_mean_keeps_the_quiet_day(self):
        rows = [
            (datetime(2024, 1, 1, 10), 1, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 1, 11), 1, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 3, 10), 1, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 3, 11), 1, "MaxPatrol SIEM", "Высокая"),
        ]
        text = ins.daily(_cards(rows), pd.date_range("2024-01-01", "2024-01-03"))
        # 4 карточки на 3 дня, включая пустой 2 января → 1, а не 2
        self.assertIn("В среднем **1**", text)

    def test_heatmap_denominator_includes_a_monday_with_no_cards(self):
        rows = [
            (datetime(2024, 1, 1, 10), 1, "MaxPatrol SIEM", "Высокая"),
            (datetime(2024, 1, 15, 10), 1, "MaxPatrol SIEM", "Высокая"),
        ]
        dates = pd.date_range("2024-01-01", "2024-01-15")
        fig = ch.fig_heatmap_hour(_cards(rows), dates)
        # Понедельники 1, 8 и 15 января; карточки только в 10:00 двух из них → 2/3
        self.assertAlmostEqual(float(fig.data[0].z[0][10]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
