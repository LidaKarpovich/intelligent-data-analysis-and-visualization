"""
course_model.py — обёртка модели курса `lgbm_gate1__v001` для задания 3 (XAI / SHAP).

Наследует совместимость pandas 2 / sklearn 1.8+ из task_2/course_model.py и добавляет:
  * prepare_and_transform(cards) -> (prepared_df, X_matrix) для извлечения 43 engineered-значений
    и быстрого переиспользования матрицы X (1099 колонок) в ALE / PDP / ICE / Ablation;
  * predict_proba_matrix(X) — инференс напрямую по готовой матрице X;
  * explain_shap_matrix(X) / explain_shap(cards) — точный TreeSHAP (pred_contrib=True в LightGBM)
    на уровне 1099 фич и с агрегацией в 73 поля (с проверкой аддитивности);
  * verify_shap_pack() — сверка расчётных SHAP с официальным course_pack/shap_cards.csv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import FEATURE_ORDER_73, build_group_matrix, load_feature_mapping, source_to_indices

MODEL_SUBDIR = Path("models") / "lgbm_gate1__v001"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SEARCH = [
    _HERE / "course_model",
    _HERE / "course_model" / "IAVD_3.7_ML_team",
    _ROOT / "task_2" / "course_model",
    _ROOT / "task_2" / "course_model" / "IAVD_3.7_ML_team",
    _ROOT / "IAVD_3.7_ML_team",
    Path.home() / "Downloads" / "IAVD_3.7_ML_team",
]


def find_course_dir(path: str | None = None) -> Path:
    candidates = [Path(path).expanduser()] if path else SEARCH
    expanded: list[Path] = []
    for d in candidates:
        expanded.append(d)
        expanded.append(d / "IAVD_3.7_ML_team")
    for d in expanded:
        if (d / MODEL_SUBDIR / "model.joblib").exists() and (d / "course_ml").is_dir():
            return d.resolve()
    raise SystemExit(
        "Не нашёл пакет модели курса IAVD_3.7_ML_team (с course_ml/ и models/lgbm_gate1__v001/). "
        "Укажите путь: python prepare.py --course-dir путь/к/IAVD_3.7_ML_team"
    )


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _to_str_pandas2(x):
    """Повторяет course_ml.features._to_str под pandas 2: пропуск остаётся None."""
    arr = np.asarray(pd.DataFrame(x).astype(object), dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for idx, v in np.ndenumerate(arr):
        out[idx] = None if _is_missing(v) else str(v)
    return out


def _patch_preprocessor(est) -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer
    if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
        est._fill_dtype = getattr(est, "_fit_dtype", np.asarray(est.statistics_).dtype)
    if isinstance(est, FunctionTransformer) and getattr(est.func, "__name__", "") == "_to_str":
        est.func = _to_str_pandas2
    for attr in ("steps", "transformers_"):
        for item in getattr(est, attr, None) or []:
            _patch_preprocessor(item[1])


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class CourseAdvisorXAI:
    def __init__(self, course_dir: Path):
        import joblib

        if importlib.util.find_spec("lightgbm") is None:
            raise SystemExit(
                "Для точного расчёта TreeSHAP (pred_contrib=True) требуется библиотека lightgbm.\n"
                "Установите её в текущее окружение: pip install lightgbm"
            )

        self.course_dir = Path(course_dir)
        if str(self.course_dir) not in sys.path:
            sys.path.insert(0, str(self.course_dir))
        from course_ml import data as data_mod, train as train_mod

        self.data_mod, self.train_mod = data_mod, train_mod
        mdir = self.course_dir / MODEL_SUBDIR
        self.model_dir = mdir
        self.pack_dir = self.course_dir / "course_pack"
        self.run = json.loads((mdir / "run.json").read_text(encoding="utf-8"))
        self.cfg = self.run["config"]
        self.threshold = float(self.run["metrics"]["tau"])
        self.freq_maps = json.loads((mdir / "freq_maps.json").read_text(encoding="utf-8"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.preproc = joblib.load(mdir / "preproc.joblib")
            self.model = joblib.load(mdir / "model.joblib")
        _patch_preprocessor(self.preproc)

        self.booster = getattr(self.model, "booster_", None) or getattr(self.model, "_Booster", None)
        if self.booster is None:
            raise SystemExit("Не удалось получить LightGBM Booster из model.joblib")
        self.engine = "lightgbm"

        self.feature_mapping = load_feature_mapping(mdir / "feature_mapping.csv")
        self.group_matrix = build_group_matrix(self.feature_mapping)     # (1099, 73)
        self.col_indices = source_to_indices(self.feature_mapping)       # source_column -> idx[]

    # ------------------------------------------------------------ препроцессинг и инференс
    def prepare_frame(self, cards: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame с исходными полями + 43 engineered-колонками (включая freq_*)."""
        part = cards.reset_index(drop=True)
        prepared, _, _ = self.data_mod.prepare(part, self.cfg)
        for column, source in self.data_mod.FREQ_FIELDS.items():
            keys = prepared[source].map(lambda v: "None" if _is_missing(v) else str(v))
            prepared[column] = keys.map(self.freq_maps.get(source, {})).fillna(0.0).astype(float)
        return prepared

    def transform_prepared(self, prepared: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = self.preproc.transform(self.train_mod.feature_frame(prepared, self.cfg))
        return np.asarray(matrix, dtype=np.float32)

    def prepare_and_transform(self, cards: pd.DataFrame, chunk: int = 5000) -> tuple[pd.DataFrame, np.ndarray]:
        """Порционно готовит DataFrame (engineered-значения) и матрицу X (n × 1099, float32)."""
        if len(cards) == 0:
            return pd.DataFrame(), np.empty((0, len(self.feature_mapping)), dtype=np.float32)
        prep_parts: list[pd.DataFrame] = []
        x_parts: list[np.ndarray] = []
        for i in range(0, len(cards), chunk):
            p = self.prepare_frame(cards.iloc[i:i + chunk])
            prep_parts.append(p)
            x_parts.append(self.transform_prepared(p))
        return pd.concat(prep_parts, ignore_index=True), np.vstack(x_parts)

    def predict_proba_matrix(self, X: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Быстрое предсказание вероятности напрямую по готовой матрице X (n × 1099)."""
        if len(X) == 0:
            return np.array([], dtype=np.float64)
        out = []
        for i in range(0, len(X), chunk):
            out.append(np.asarray(self.booster.predict(X[i:i + chunk]), dtype=np.float64))
        return np.concatenate(out)

    def predict_proba(self, cards: pd.DataFrame, chunk: int = 5000) -> np.ndarray:
        _, X = self.prepare_and_transform(cards, chunk=chunk)
        return self.predict_proba_matrix(X)

    # ------------------------------------------------------------ TreeSHAP (1099 и 73)
    def explain_shap_matrix(self, X: np.ndarray, chunk: int = 5000,
                            return_full: bool = False) -> dict[str, np.ndarray]:
        """Считает точный TreeSHAP (log-odds) через booster.predict(..., pred_contrib=True)."""
        n = len(X)
        if n == 0:
            return {
                "shap_73": np.empty((0, len(FEATURE_ORDER_73)), dtype=np.float64),
                "base": np.array([], dtype=np.float64),
                "raw_score": np.array([], dtype=np.float64),
                "proba": np.array([], dtype=np.float64),
            }
        s73_parts, base_parts, s1099_parts = [], [], []
        for i in range(0, n, chunk):
            c = np.asarray(self.booster.predict(X[i:i + chunk], pred_contrib=True), dtype=np.float64)
            s1099 = c[:, :-1]
            base = c[:, -1]
            s73_parts.append(s1099 @ self.group_matrix)
            base_parts.append(base)
            if return_full:
                s1099_parts.append(s1099.astype(np.float32))
        shap_73 = np.vstack(s73_parts)
        base = np.concatenate(base_parts)
        raw_score = base + shap_73.sum(axis=1)
        proba = sigmoid(raw_score)
        res = {"shap_73": shap_73, "base": base, "raw_score": raw_score, "proba": proba}
        if return_full:
            res["shap_1099"] = np.vstack(s1099_parts)
        return res

    # ------------------------------------------------------------ самопроверки § 4
    def self_check(self, raw: pd.DataFrame, tol: float = 1e-5) -> float:
        """Сверка 5 контрольных карточек модели (control_cards.csv, допуск курса 4e-7)."""
        ref = pd.read_csv(self.model_dir / "control_cards.csv", sep=";", encoding="utf-8-sig")
        ids = pd.to_numeric(raw["EntityId"], errors="coerce")
        cards = raw[ids.isin(ref["EntityId"])]
        if len(cards) != len(ref):
            raise SystemExit(f"Контрольных карточек в датасете {len(cards)} из {len(ref)}")
        got = pd.DataFrame({
            "EntityId": pd.to_numeric(cards["EntityId"]).astype("int64").to_numpy(),
            "proba": self.predict_proba(cards),
        }).merge(ref, on="EntityId")
        delta = float((got["proba"] - got["expected_proba"]).abs().max())
        if delta > tol:
            raise SystemExit(f"Модель курса не воспроизводится: расхождение {delta:.2e} > {tol:.0e}")
        return delta

    def verify_shap_pack(self) -> dict[str, float]:
        """Сверка TreeSHAP с официальным course_pack/shap_cards.csv и проверка аддитивности."""
        shap_csv = self.pack_dir / "shap_cards.csv"
        cards_pq = self.pack_dir / "cards_raw.parquet"
        if not shap_csv.exists() or not cards_pq.exists():
            return {"checked_cards": 0, "max_shap_diff": float("nan"), "max_additivity_diff": float("nan")}

        ref = pd.read_csv(shap_csv, sep=";", encoding="utf-8-sig")
        raw_pack = pd.read_parquet(cards_pq)
        raw_pack["_eid"] = pd.to_numeric(raw_pack["EntityId"]).astype("int64")
        ref["_eid"] = pd.to_numeric(ref["EntityId"]).astype("int64")

        # Выравниваем порядок карточек строго по shap_cards.csv
        idx_map = {eid: i for i, eid in enumerate(raw_pack["_eid"].to_numpy())}
        keep = [i for i, eid in enumerate(ref["_eid"].to_numpy()) if eid in idx_map]
        ref_m = ref.iloc[keep].reset_index(drop=True)
        raw_m = raw_pack.iloc[[idx_map[eid] for eid in ref_m["_eid"].to_numpy()]].drop(columns=["_eid"]).reset_index(drop=True)

        _, X_m = self.prepare_and_transform(raw_m)
        expl = self.explain_shap_matrix(X_m)
        proba_direct = self.predict_proba_matrix(X_m)

        ref_shap = ref_m[FEATURE_ORDER_73].to_numpy(dtype=np.float64)
        max_shap_diff = float(np.abs(expl["shap_73"] - ref_shap).max())
        max_base_diff = float(np.abs(expl["base"] - ref_m["base"].to_numpy(dtype=np.float64)).max())
        max_add_diff = float(np.abs(expl["proba"] - proba_direct).max())
        max_csv_proba_diff = float(np.abs(expl["proba"] - ref_m["proba"].to_numpy(dtype=np.float64)).max())

        if max_add_diff > 1e-6:
            raise SystemExit(f"Нарушена аддитивность SHAP: max |σ(base+Σ) - proba| = {max_add_diff:.2e} > 1e-6")
        if max_shap_diff > 1e-4:
            raise SystemExit(f"Расхождение SHAP с course_pack/shap_cards.csv: {max_shap_diff:.2e}")

        return {
            "checked_cards": int(len(ref_m)),
            "max_shap_diff": max_shap_diff,
            "max_base_diff": max_base_diff,
            "max_additivity_diff": max_add_diff,
            "max_csv_proba_diff": max_csv_proba_diff,
        }