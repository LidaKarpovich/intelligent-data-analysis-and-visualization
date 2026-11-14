"""Синтетические проверки логики дашборда и подготовки данных. Не требуют parquet и не запускают Taipy.

    python -m unittest test_logic.py
"""
import importlib.util
import re
import unittest

import numpy as np
import pandas as pd

import dashboard_charts as ch
import dashboard_data as dd
import dashboard_kpis as kp

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def _cards(n_pos=40, n_neg=400, seed=0, part=dd.OPER, month="2025-06") -> pd.DataFrame:
    """Синтетические карточки: инциденты получают оценки выше FP, но с перекрытием."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    proba = np.clip(np.where(y == 1, rng.beta(5, 2, len(y)), rng.beta(2, 5, len(y))), 1e-4, 1)
    created = pd.Timestamp(month + "-01") + pd.to_timedelta(rng.integers(0, 27 * 24, len(y)), unit="h")
    return pd.DataFrame({
        "id": np.arange(len(y)), "created": created, "part": part, "verdict": np.where(y == 1, "Решён", "False Positive"),
        "y": y, "proba": proba, "rule": rng.choice(["rule-a", "rule-b", "rule-c"], len(y)),
        "inc_type": "type-a", "irp": rng.choice(["irp-a", "irp-b"], len(y)),
        "crit": rng.choice(dd.CRIT_ORDER[:3], len(y)), "org": rng.choice(["Орг-1", "Орг-2", None], len(y)),
        "vendor": "v", "siem": "s", "group": "L1", "hour": created.hour, "dow": created.dayofweek,
        "rule_train_n": rng.choice([0, 50, 500, 5000], len(y)), "cmd_len": np.nan, "attack_dur_min": np.nan,
    })


def _store(frames: list[pd.DataFrame], thr: float = 0.5) -> dd.Store:
    cards = dd.enrich(pd.concat(frames, ignore_index=True))
    meta = {"threshold": thr, "target_recall": 0.95, "policies": {"0.95": thr, "0.99": thr / 2},
            "periods": {"train": ["2024-01-01", "2025-02-28"], "val": ["2025-03-01", "2025-04-30"],
                        "oper": ["2025-05-01", "2025-11-20"]}, "pos_rate_train": 0.2}
    return dd.Store(cards=cards, meta=meta, policies=dd._policies(meta))


class Metrics(unittest.TestCase):
    def test_confusion_and_rates(self):
        d = pd.DataFrame({"y": [1, 1, 1, 0, 0, 0, 0, -1], "pred": [True, True, False, True, False, False, False, True]})
        c = dd.confusion(d)
        self.assertEqual((c.tp, c.fn, c.fp, c.tn), (2, 1, 1, 3))      # карточка без вердикта не учитывается
        self.assertAlmostEqual(c.recall, 2 / 3)
        self.assertAlmostEqual(c.miss, 1 / 3)
        self.assertAlmostEqual(c.specificity, 3 / 4)
        self.assertAlmostEqual(c.low_share, 4 / 7)
        self.assertAlmostEqual(c.agreement, 5 / 7)

    def test_empty_confusion_is_nan_not_error(self):
        c = dd.confusion(pd.DataFrame({"y": [], "pred": []}))
        self.assertEqual(c.n, 0)
        self.assertTrue(np.isnan(c.recall) and np.isnan(c.precision))

    def test_wilson_contains_point_and_handles_zero(self):
        lo, hi = dd.wilson(5, 100)
        self.assertLess(lo, 0.05)
        self.assertGreater(hi, 0.05)
        lo0, hi0 = dd.wilson(0, 50)
        self.assertEqual(lo0, 0.0)
        self.assertGreater(hi0, 0.0)                                   # 0 из 50 — не «точно ноль»
        self.assertTrue(all(np.isnan(dd.wilson(0, 0))))

    def test_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        d = _cards(seed=3)
        self.assertAlmostEqual(dd.auc(d["y"], d["proba"]), roc_auc_score(d["y"], d["proba"]), places=10)
        self.assertTrue(np.isnan(dd.auc(np.ones(5), np.arange(5))))

    def test_gains_goes_from_zero_to_one_and_beats_diagonal(self):
        g = dd.gains(_cards(seed=1))
        self.assertEqual((g["x"].iloc[0], g["y"].iloc[0]), (0.0, 0.0))
        self.assertAlmostEqual(g["x"].iloc[-1], 1.0)
        self.assertAlmostEqual(g["y"].iloc[-1], 1.0)
        self.assertTrue((np.diff(g["y"]) >= 0).all())
        self.assertGreater(np.interp(0.2, g["x"], g["y"]), 0.2)

    def test_calibration_bins_have_ci(self):
        cal = dd.calibration(_cards(n_pos=200, n_neg=2000, seed=2), min_n=10)
        self.assertTrue(((cal["lo"] <= cal["actual"]) & (cal["actual"] <= cal["hi"])).all())
        self.assertTrue(((cal["pred"] >= 0) & (cal["pred"] <= 1)).all())


class Threshold(unittest.TestCase):
    def test_pick_threshold_reaches_target_recall(self):
        prep = _load_prepare()
        if prep is None:
            self.skipTest("prepare.py требует pyarrow")
        d = _cards(n_pos=500, n_neg=500, seed=4)
        for target in (0.9, 0.95, 0.99):
            thr = prep.pick_threshold(d["y"].to_numpy(), d["proba"].to_numpy(), target)
            recall = (d.loc[d["y"] == 1, "proba"] >= thr).mean()
            self.assertGreaterEqual(recall, target)
            self.assertLess(recall, target + 0.01)                    # и порог не занижен без нужды


def _load_prepare():
    if not HAS_PYARROW:
        return None
    import prepare
    return prepare


class Parsing(unittest.TestCase):
    def test_items_and_name_of(self):
        prep = _load_prepare()
        if prep is None:
            self.skipTest("prepare.py требует pyarrow")
        self.assertEqual(prep.items('["a", "b"]'), ["a", "b"])
        self.assertEqual(prep.items('[""]'), [])
        self.assertEqual(prep.items("plain"), ["plain"])
        self.assertEqual(prep.items(None), [])
        self.assertEqual(prep.items(float("nan")), [])
        self.assertEqual(prep.name_of('{"name": "L1"}'), "L1")
        self.assertEqual(prep.name_of('{"name": "ИАиВД", "address": null}'), "ИАиВД")

    def test_time_split(self):
        prep = _load_prepare()
        if prep is None:
            self.skipTest("prepare.py требует pyarrow")
        s = pd.Series(pd.to_datetime(["2025-02-28 23:59", "2025-03-01 00:00", "2025-04-30 23:00", "2025-05-01 00:00"]))
        self.assertEqual(prep.assign_part(s).tolist(), ["train", "val", "val", "oper"])


class Filters(unittest.TestCase):
    def setUp(self):
        self.store = _store([_cards(seed=5, part=dd.VAL, month="2025-03"), _cards(seed=6, month="2025-06"),
                             _cards(seed=7, month="2025-07")])

    def test_period_filter_touches_only_operation(self):
        f = dd.Filters(period="Июль 2025", policy=dd.default_policy(self.store))
        oper = dd.apply_filters(self.store, f)
        self.assertEqual(set(oper["month"].astype(str)), {"2025-07"})
        both = dd.apply_filters(self.store, f, part=None)
        self.assertIn("2025-03", set(both["month"].astype(str)))       # приёмка остаётся для сравнения

    def test_policy_changes_threshold(self):
        f95 = dd.Filters(policy=dd.policy_label(0.95))
        f99 = dd.Filters(policy=dd.policy_label(0.99))
        self.assertGreater(dd.threshold_of(self.store, f95), dd.threshold_of(self.store, f99))
        self.assertGreaterEqual(dd.confusion(dd.apply_filters(self.store, f99)).recall,
                                dd.confusion(dd.apply_filters(self.store, f95)).recall)

    def test_missing_org_is_named(self):
        self.assertIn("не указана", dd.org_options(self.store))

    def test_crit_filter(self):
        d = dd.apply_filters(self.store, dd.Filters(crit=["Низкая"]))
        self.assertEqual(set(d["crit"].astype(str)), {"Низкая"})


class Segments(unittest.TestCase):
    def test_systematic_flag(self):
        # срез «Орг-плохая»: 30 % пропусков против ~2 % у остальных -> систематически
        rows = []
        for org, n_pos, miss in [("Орг-плохая", 100, 30), ("Орг-1", 1000, 10), ("Орг-2", 1000, 10)]:
            rows += [(org, 1, i >= miss) for i in range(n_pos)]
        d = pd.DataFrame(rows, columns=["org", "y", "pred"])
        d["crit"] = pd.Categorical(["Средняя"] * len(d), categories=dd.CRIT_ORDER)
        t = dd.segment_table(d, "Организация", "miss")
        bad = t.set_index("seg").loc["Орг-плохая"]
        self.assertTrue(bad["systematic"])
        self.assertAlmostEqual(bad["rate"], 0.3)
        self.assertFalse(t.set_index("seg").loc["Орг-1", "systematic"])

    def test_small_levels_fold_into_other(self):
        d = pd.DataFrame({"org": ["A"] * 50 + ["B"] * 5 + ["C"] * 5, "y": 1, "pred": True})
        d["crit"] = pd.Categorical(["Средняя"] * len(d), categories=dd.CRIT_ORDER)
        segs = set(dd.segment_table(d, "Организация", "miss")["seg"])
        self.assertEqual(segs, {"A"})                                  # «прочие» = 10 < MIN_POS -> скрыто

    def test_near_threshold(self):
        d = pd.DataFrame({"y": [1, 1, 1, 0], "proba": [0.09, 0.03, 0.2, 0.01]})
        d["pred"] = d["proba"] >= 0.1
        self.assertEqual(dd.near_threshold_share(d, 0.1), (1, 2))


class Formatting(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(dd.fmt_int(46711), "46 711")
        self.assertEqual(dd.fmt_pct(0.0182), "1,8 %")
        self.assertEqual(dd.fmt_pct(float("nan")), "—")

    def test_plural_and_months(self):
        self.assertEqual([dd.plural(n, ("пропуск", "пропуска", "пропусков")) for n in (1, 2, 5, 11, 21, 91, 112)],
                         ["пропуск", "пропуска", "пропусков", "пропусков", "пропуск", "пропуск", "пропусков"])
        self.assertEqual(dd.month_in(pd.Period("2025-05", "M")), "в мае 2025")
        self.assertEqual(dd.month_since(pd.Period("2025-07", "M")), "с июля 2025")


class ChartsAndTiles(unittest.TestCase):
    def setUp(self):
        self.store = _store([_cards(seed=8, part=dd.VAL, month="2025-03"), _cards(seed=9, month="2025-06"),
                             _cards(seed=10, month="2025-07")])
        self.f = dd.Filters(policy=dd.default_policy(self.store))

    def test_every_chart_builds_on_data_and_on_empty(self):
        for d in (dd.apply_filters(self.store, self.f), dd.apply_filters(self.store, self.f).iloc[0:0]):
            val = dd.apply_filters(self.store, self.f, part=dd.VAL)
            c = dd.confusion(d)
            mon = dd.monthly(pd.concat([val, d]))
            figs = [ch.fig_queue_waterfall(c), ch.fig_gains(val, d, 0.5), ch.fig_sankey(c),
                    ch.fig_agreement_monthly(mon), ch.fig_miss_monthly(mon, 0.05), ch.fig_missed_rules(d),
                    ch.fig_segments(dd.segment_table(d, "Организация", "miss"), "miss"), ch.fig_score_hist(d, 0.5),
                    ch.fig_calibration(val, d), ch.fig_dumbbell([("x", 0.1, c.recall)]), ch.fig_drift(mon)]
            for fig in figs:
                self.assertIsNotNone(fig.to_dict())

    def test_waterfall_steps_add_up(self):
        c = dd.Confusion(tp=90, fn=10, fp=600, tn=300)
        bar = ch.fig_queue_waterfall(c).data[0]
        start, tn, fn, keep = bar.y
        self.assertEqual(start - tn - fn, keep)
        self.assertEqual(keep, c.tp + c.fp)

    def test_kpi_alerts(self):
        d = dd.apply_filters(self.store, self.f)
        c = dd.confusion(d)
        out = kp.results(d, c, dd.Confusion(tp=95, fn=5, fp=400, tn=500), target_miss=0.05)
        self.assertEqual(set(out), {f"{k}_{s}" for k, _ in kp.TILES["results"] for s in "vnc"})
        empty = kp.results(d.iloc[0:0], dd.Confusion(), dd.Confusion(), 0.05)
        self.assertTrue(all(v == kp.DASH for k, v in empty.items() if k.endswith("_v")))

    def test_every_chart_has_why(self):
        import report
        keys = {row[3] for row in report.CHARTS}
        self.assertEqual(keys, set(ch.WHY))

    def test_page_markup_uses_known_tiles(self):
        for page, tiles in kp.TILES.items():
            md = kp.markup(page)
            used = set(re.findall(r"\{(\w+)_v\}", md))
            self.assertEqual(used, {k for k, _ in tiles})


if __name__ == "__main__":
    unittest.main()
