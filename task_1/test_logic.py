"""Синтетические проверки логики дашборда. Не требуют parquet и не запускают Taipy."""
import unittest
from datetime import datetime

import pandas as pd

import dashboard_data as dd


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


if __name__ == "__main__":
    unittest.main()
