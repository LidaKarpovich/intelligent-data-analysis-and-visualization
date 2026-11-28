"""
test_logic.py — синтетические проверки логики задания 3 без запуска Taipy.

Запуск из папки task_3:
    python -m unittest test_logic.py
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import dashboard_data as dd
from feature_registry import CAT_FEATURES_12, FEATURE_ORDER_73, NUMERIC_FEATURES_43, REGISTRY
from prepare import compute_psi_numeric, hexbin_aggregate


class TestRegistry(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(FEATURE_ORDER_73), 73)
        self.assertEqual(len(REGISTRY), 73)
        self.assertEqual(len(NUMERIC_FEATURES_43), 43)
        self.assertEqual(len(CAT_FEATURES_12), 12)
        n_eng = sum(1 for m in REGISTRY.values() if m.kind == "engineered")
        n_card = sum(1 for m in REGISTRY.values() if m.kind == "card")
        self.assertEqual(n_eng, 43)
        self.assertEqual(n_card, 30)


class TestThresholdAndOutcomes(unittest.TestCase):
    def test_outcomes_team5_at_default_and_shifted_tau(self):
        # Реальные скоры команды 5: FN (~1e-5), FP (~0.03), TN (~3e-6), TP (~0.42)
        cards = [(1, 0.000007), (1, 0.000009), (0, 0.029370), (0, 0.031811),
                 (0, 0.000003), (0, 0.000003), (1, 0.405199), (1, 0.449694)]
        got_default = [dd.outcome_of(y, p, 0.0193) for y, p in cards]
        self.assertEqual(got_default, ["FN", "FN", "FP", "FP", "TN", "TN", "TP", "TP"])

        # Сдвиг до 0.035 убирает оба FP, не затрагивая TP
        got_035 = [dd.outcome_of(y, p, 0.035) for y, p in cards]
        self.assertEqual(got_035, ["FN", "FN", "TN", "TN", "TN", "TN", "TP", "TP"])

        y = np.array([c[0] for c in cards])
        p = np.array([c[1] for c in cards])
        cm = dd.confusion_from_proba(y, p, 0.0193)
        self.assertEqual((cm["tp"], cm["fp"], cm["fn"], cm["tn"]), (2, 2, 2, 2))


class TestTopNAdditivity(unittest.TestCase):
    def test_topn_preserves_exact_raw_score(self):
        rng = np.random.default_rng(7)
        shap_73 = rng.normal(0.0, 0.4, size=73)
        base_v = -8.852417
        raw_v = base_v + float(shap_73.sum())
        proba = float(dd.sigmoid(raw_v))

        # Собираем минимальный mock Store
        cards_df = pd.DataFrame([{
            "EntityId": 999999, "rank_in_team": 1, "y_true": 1, "proba_calc": proba,
            "base": base_v, "raw_score": raw_v, "sum_shap": float(shap_73.sum()),
            "sum_pos": float(np.clip(shap_73, 0, None).sum()),
            "sum_neg": float(np.clip(shap_73, None, 0).sum()),
            "flag": "инцидент", "outcome": "TP",
        }])
        abs_v = np.abs(shap_73)
        fields_df = pd.DataFrame({
            "EntityId": 999999,
            "source_column": FEATURE_ORDER_73,
            "label_ru": [REGISTRY[c].label_ru for c in FEATURE_ORDER_73],
            "kind": [REGISTRY[c].kind for c in FEATURE_ORDER_73],
            "kind_label": [REGISTRY[c].kind_label for c in FEATURE_ORDER_73],
            "value_str": ["val"] * 73,
            "shap_val": shap_73,
            "abs_shap": abs_v,
            "share_abs_shap": abs_v / abs_v.sum(),
            "rank_abs": pd.Series(abs_v).rank(ascending=False, method="first").astype(int),
        })

        class _DummyStore:
            team5_cards = cards_df
            team5_fields = fields_df
            team5_1099 = pd.DataFrame()

        for n in (5, 10, 15, 20):
            slc = dd.slice_top_n(_DummyStore(), 999999, top_n=n, detail="73 поля SOC")
            reconstructed = slc["base"] + slc["sum_top_net"] + slc["other_val"]
            self.assertAlmostEqual(reconstructed, raw_v, places=10)
            self.assertLessEqual(slc["additivity_diff"], 1e-12)
            self.assertGreaterEqual(slc["coverage"], 0.0)
            self.assertLessEqual(slc["coverage"], 1.0)


class TestStatsAndHexbin(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = dd.wilson(9, 10)
        self.assertGreater(lo, 0.5)
        self.assertLess(hi, 1.0)
        lo0, hi0 = dd.wilson(0, 20)
        self.assertEqual(lo0, 0.0)
        self.assertGreater(hi0, 0.0)

    def test_psi_zero_for_identical_and_high_for_shifted(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, size=2000)
        psi_same, _ = compute_psi_numeric(a, a.copy())
        psi_shift, _ = compute_psi_numeric(a, a + 2.5)
        self.assertLess(psi_same, 1e-4)
        self.assertGreater(psi_shift, 0.5)

    def test_hexbin_preserves_total_count(self):
        rng = np.random.default_rng(2)
        n = 5000
        coords = rng.normal(size=(n, 2))
        y = rng.integers(0, 2, size=n, dtype=np.int8)
        proba = rng.uniform(0, 1, size=n)
        cl = rng.integers(0, 5, size=n)
        rules = np.array(["RuleA"] * n)
        parts = np.array(["Эксплуатация"] * n)
        hx = hexbin_aggregate(coords, y, proba, 0.0193, cl, rules, parts, gridsize=20)
        self.assertEqual(int(hx["n"].sum()), n)
        self.assertLess(len(hx), 700)


if __name__ == "__main__":
    unittest.main()