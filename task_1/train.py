"""
train.py — обучает модель-советник и готовит таблицы для страниц дашборда.

Запуск после etl.py:
    python train.py

Что получается в data/agg/:
    predictions.parquet   — y_proba для ВСЕХ карточек + split (train / test / unlabeled)
    shap_sample.parquet   — SHAP-значения для выборки из теста (Summary и Waterfall)
    pdp.parquet           — частичная зависимость по нескольким признакам
    bias_variance.parquet — ошибка на train и валидации по сложности модели
    meta.json             — метрики, список признаков, базовое значение SHAP
    model.txt             — сама модель LightGBM
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

AGG = Path(__file__).resolve().parent / "data" / "agg"
SEED = 42
TEST_SHARE = 0.25        # последние 25% размеченных карточек по времени — тест

NUM = ["crit_ord", "crit_src_ord", "n_incidents", "hour", "dow", "lag_min", "attack_dur_min",
       "n_rules", "n_tags", "n_src", "n_dst", "n_values", "desc_len",
       "cmd_len", "cmd_entropy", "cmd_n", "cmd_enc"]

PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=SEED, verbose=-1)

PDP_FEATURES = ["cmd_len", "cmd_entropy", "n_rules", "attack_dur_min"]


def pdp(model, test, features, feature, n_grid=25):
    """Частичная зависимость: среднее предсказание при подстановке значения признака всем строкам."""
    pool = test[test[feature].notna()]
    if len(pool) < 50:
        return None
    X = pool.sample(min(2000, len(pool)), random_state=SEED)[features].copy()
    grid = np.unique(np.quantile(pool[feature], np.linspace(0.02, 0.98, n_grid)))
    rows = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        p = model.predict_proba(Xg)[:, 1]
        rows.append((feature, float(g), p.mean(), np.quantile(p, 0.25), np.quantile(p, 0.75)))
    return pd.DataFrame(rows, columns=["feature", "value", "mean", "q25", "q75"])


def main() -> None:
    feats = pd.read_parquet(AGG / "features.parquet")
    cat = [c for c in feats.columns if c.endswith("_c")]
    features = NUM + cat

    # ---- временное разбиение размеченных карточек (прошлое -> будущее) ----
    lab = feats[feats["y"] >= 0].sort_values("created", kind="stable")
    cut = int(len(lab) * (1 - TEST_SHARE))
    train, test = lab.iloc[:cut], lab.iloc[cut:]
    print(f"train: {len(train):,} (доля 1 = {train['y'].mean():.3f}) | "
          f"test: {len(test):,} (доля 1 = {test['y'].mean():.3f}) | "
          f"граница по времени: {test['created'].min()}")

    # ---- модель ----
    model = lgb.LGBMClassifier(**PARAMS).fit(train[features], train["y"])
    p_test = model.predict_proba(test[features])[:, 1]
    roc = roc_auc_score(test["y"], p_test)
    ap = average_precision_score(test["y"], p_test)
    print(f"\nТЕСТ: ROC AUC = {roc:.3f} | PR AUC = {ap:.3f} (случайная модель: PR AUC = {test['y'].mean():.3f})")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=features)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    print("\nТоп-10 признаков по вкладу (gain, %):")
    print(imp.head(10).round(1).to_string())

    # ---- предсказания для всех карточек ----
    feats["y_proba"] = model.predict_proba(feats[features])[:, 1]
    feats["split"] = "unlabeled"
    feats.loc[train.index, "split"] = "train"
    feats.loc[test.index, "split"] = "test"
    keep = ["id", "created", "hour", "dow", "split", "y", "verdict_txt", "y_proba", "rule", "inc_type",
            "crit", "group", "n_rules", "attack_dur_min", "lag_min", "react_min", "solve_min"]
    feats[keep].to_parquet(AGG / "predictions.parquet", index=False)

    # ---- SHAP (TreeSHAP средствами LightGBM: pred_contrib=True) ----
    sample = test.sample(min(3000, len(test)), random_state=SEED)
    X = sample[features]
    contrib = model.booster_.predict(X, pred_contrib=True)      # (n, признаки + 1), в log-odds
    base = float(contrib[0, -1])
    shap_df = pd.DataFrame(contrib[:, :-1], columns=["shap__" + f for f in features], index=sample.index)
    vals = X.copy()
    for c in cat:
        vals[c] = vals[c].cat.codes.astype("float")             # для окраски точек
    shap_out = pd.concat([sample[["id", "y"]], vals, shap_df], axis=1)
    shap_out["y_proba"] = model.predict_proba(X)[:, 1]
    shap_out.to_parquet(AGG / "shap_sample.parquet", index=False)

    # ---- PDP ----
    pdps = [d for d in (pdp(model, test, features, f) for f in PDP_FEATURES) if d is not None]
    if pdps:
        pd.concat(pdps, ignore_index=True).to_parquet(AGG / "pdp.parquet", index=False)

    # ---- Bias-variance: ошибка на обучении и на валидации по глубине деревьев ----
    inner = int(len(train) * 0.8)
    tr_in, va = train.iloc[:inner], train.iloc[inner:]
    rows = []
    for depth in [1, 2, 3, 4, 6, 8, 10, 12]:
        params = {**PARAMS, "max_depth": depth, "num_leaves": min(2 ** depth, 255)}
        m = lgb.LGBMClassifier(**params).fit(tr_in[features], tr_in["y"])
        rows.append((depth,
                     log_loss(tr_in["y"], m.predict_proba(tr_in[features])[:, 1], labels=[0, 1]),
                     log_loss(va["y"], m.predict_proba(va[features])[:, 1], labels=[0, 1])))
    bv = pd.DataFrame(rows, columns=["max_depth", "train_loss", "val_loss"])
    bv.to_parquet(AGG / "bias_variance.parquet", index=False)
    print("\nBias-variance (log-loss):")
    print(bv.round(3).to_string(index=False))

    # ---- служебное ----
    model.booster_.save_model(str(AGG / "model.txt"))
    meta = {
        "features": features, "num": NUM, "cat": cat,
        "n_train": len(train), "n_test": len(test),
        "pos_rate_train": float(train["y"].mean()), "pos_rate_test": float(test["y"].mean()),
        "test_starts": str(test["created"].min()),
        "roc_auc": roc, "pr_auc": ap, "shap_base": base, "params": PARAMS,
    }
    (AGG / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово: файлы в", AGG.resolve())


if __name__ == "__main__":
    main()
