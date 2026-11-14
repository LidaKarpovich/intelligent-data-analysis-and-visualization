"""
course_model.py — модель курса `lgbm_gate1__v001` из пакета практик 3.6–3.7 (папка IAVD_3.7_ML_team).

Обёртка повторяет course_ml.inference.CourseModel и добавляет две вещи:

1. Совместимость. Пакет собран под pandas 2.3 и scikit-learn 1.7. Под pandas 3 пустые поля
   превращаются в NaN вместо None, и модель видит другие категории («missing» вместо «None»).
   Обёртка возвращает поведение pandas 2, поэтому оценки совпадают с эталоном при любой версии.
2. Работа без LightGBM. Если библиотека lightgbm не ставится (на macOS ей нужен libomp),
   деревья модели вычисляются на numpy по текстовому дампу из model.joblib — результат тот же.

Самопроверка: self_check() сверяет 5 контрольных карточек пакета (control_cards.csv),
допуск курса — 4e-7 (берём 1e-5: эталон округлён до 6 знаков).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SUBDIR = Path("models") / "lgbm_gate1__v001"
SEARCH = [
    Path(__file__).resolve().parent / "course_model",
    Path(__file__).resolve().parent.parent / "IAVD_3.7_ML_team",
    Path.home() / "Downloads" / "IAVD_3.7_ML_team",
]
K_ZERO = 1e-35


def find_course_dir(path: str | None = None) -> Path:
    candidates = [Path(path).expanduser()] if path else SEARCH
    for d in candidates:
        if (d / MODEL_SUBDIR / "model.joblib").exists() and (d / "course_ml").is_dir():
            return d.resolve()
    raise SystemExit("Не нашёл пакет модели курса (папка IAVD_3.7_ML_team с course_ml/ и models/). "
                     "Укажите путь: python prepare.py --course-dir путь/к/IAVD_3.7_ML_team")


# ---------------------------------------------------------------- деревья LightGBM на numpy
class _Tree:
    def __init__(self, kv: dict):
        self.num_leaves = int(kv["num_leaves"])
        self.leaf_value = np.array(kv["leaf_value"].split(), dtype=float)
        if self.num_leaves > 1:
            self.split_feature = np.array(kv["split_feature"].split(), dtype=int)
            self.threshold = np.array(kv["threshold"].split(), dtype=float)
            self.decision_type = np.array(kv["decision_type"].split(), dtype=int)
            self.left = np.array(kv["left_child"].split(), dtype=int)
            self.right = np.array(kv["right_child"].split(), dtype=int)
            if int(kv.get("num_cat", 0)):
                raise NotImplementedError("категориальные разбиения LightGBM не поддерживаются")

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.num_leaves == 1:
            return np.full(n, self.leaf_value[0])
        node = np.zeros(n, dtype=int)                 # ≥ 0 — внутренний узел, < 0 — лист (~индекс)
        active = np.ones(n, dtype=bool)
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            v = X[idx, self.split_feature[nd]].copy()
            dt = self.decision_type[nd]
            default_left = (dt & 2) == 2
            missing = (dt >> 2) & 3                    # 0 — нет, 1 — ноль, 2 — NaN
            v[np.isnan(v) & (missing != 2)] = 0.0
            is_missing = ((missing == 1) & (np.abs(v) <= K_ZERO)) | ((missing == 2) & np.isnan(v))
            go_left = np.where(is_missing, default_left, v <= self.threshold[nd])
            nxt = np.where(go_left, self.left[nd], self.right[nd])
            node[idx] = nxt
            active[idx] = nxt >= 0
        return self.leaf_value[~node]


class NumpyBooster:
    """Бинарный классификатор LightGBM по текстовому дампу модели (model_to_string)."""

    def __init__(self, model_str: str):
        self.trees = []
        for block in model_str.split("\nTree=")[1:]:
            kv = {}
            for line in block.split("\nend of trees")[0].split("\n")[1:]:
                if "=" in line:
                    k, val = line.split("=", 1)
                    kv[k.strip()] = val.strip()
            if "num_leaves" in kv:
                self.trees.append(_Tree(kv))
        head = model_str.split("\nTree=")[0]
        self.sigmoid = next((float(l.split("sigmoid:")[1].split()[0]) for l in head.split("\n")
                             if l.startswith("objective=") and "sigmoid:" in l), 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        raw = np.zeros(X.shape[0])
        for t in self.trees:
            raw += t.predict(X)
        p = 1.0 / (1.0 + np.exp(-self.sigmoid * raw))
        return np.c_[1 - p, p]


def _booster_string_without_lightgbm(model_path: Path) -> str:
    """Достаёт дамп деревьев из model.joblib, подставив заглушки вместо классов lightgbm."""
    import joblib

    class _Stub:
        def __setstate__(self, state):
            self.__dict__.update(state)

    saved = {k: sys.modules.get(k) for k in ("lightgbm", "lightgbm.sklearn", "lightgbm.basic")}
    try:
        for name in saved:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["lightgbm.sklearn"].LGBMClassifier = type("LGBMClassifier", (_Stub,), {})
        sys.modules["lightgbm.basic"].Booster = type("Booster", (_Stub,), {})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = joblib.load(model_path)
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    booster = clf.__dict__["_Booster"].__dict__
    return booster.get("_handle") or booster.get("handle")


# ---------------------------------------------------------------- совместимость с pandas 2
def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def _to_str_pandas2(x):
    """course_ml.features._to_str, как он работает под pandas 2: пропуск остаётся None."""
    arr = np.asarray(pd.DataFrame(x).astype(object), dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for idx, v in np.ndenumerate(arr):
        out[idx] = None if _is_missing(v) else str(v)
    return out


def _patch_preprocessor(est) -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer
    if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):      # sklearn ≥ 1.8
        est._fill_dtype = getattr(est, "_fit_dtype", np.asarray(est.statistics_).dtype)
    if isinstance(est, FunctionTransformer) and getattr(est.func, "__name__", "") == "_to_str":
        est.func = _to_str_pandas2
    for attr in ("steps", "transformers_"):
        for item in getattr(est, attr, None) or []:
            _patch_preprocessor(item[1])


# ---------------------------------------------------------------- модель
class CourseAdvisor:
    def __init__(self, course_dir: Path):
        import joblib
        self.course_dir = Path(course_dir)
        if str(self.course_dir) not in sys.path:
            sys.path.insert(0, str(self.course_dir))
        from course_ml import data as data_mod, train as train_mod     # код курса
        self.data_mod, self.train_mod = data_mod, train_mod
        mdir = self.course_dir / MODEL_SUBDIR
        self.model_dir = mdir
        self.run = json.loads((mdir / "run.json").read_text(encoding="utf-8"))
        self.cfg = self.run["config"]
        self.threshold = float(self.run["metrics"]["tau"])
        self.freq_maps = json.loads((mdir / "freq_maps.json").read_text(encoding="utf-8"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                            # версия sklearn может отличаться
            self.preproc = joblib.load(mdir / "preproc.joblib")
        _patch_preprocessor(self.preproc)
        if importlib.util.find_spec("lightgbm") is not None:
            self.model = joblib.load(mdir / "model.joblib")
            self.engine = "lightgbm"
        else:
            self.model = NumpyBooster(_booster_string_without_lightgbm(mdir / "model.joblib"))
            self.engine = "numpy (без lightgbm)"

    def predict_proba(self, cards: pd.DataFrame, chunk: int = 5000) -> np.ndarray:
        out = []
        for i in range(0, len(cards), chunk):
            part = cards.iloc[i:i + chunk].reset_index(drop=True)
            prepared, _, _ = self.data_mod.prepare(part, self.cfg)
            for column, source in self.data_mod.FREQ_FIELDS.items():
                keys = prepared[source].map(lambda v: "None" if _is_missing(v) else str(v))
                prepared[column] = keys.map(self.freq_maps.get(source, {})).fillna(0.0).astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matrix = self.preproc.transform(self.train_mod.feature_frame(prepared, self.cfg))
            out.append(self.model.predict_proba(np.asarray(matrix, dtype=np.float32))[:, 1])
        return np.concatenate(out) if out else np.array([])

    def self_check(self, raw: pd.DataFrame, tol: float = 1e-5) -> float:
        ref = pd.read_csv(self.model_dir / "control_cards.csv", sep=";", encoding="utf-8-sig")
        ids = pd.to_numeric(raw["EntityId"], errors="coerce")
        cards = raw[ids.isin(ref["EntityId"])]
        if len(cards) != len(ref):
            raise SystemExit(f"Контрольных карточек в датасете {len(cards)} из {len(ref)}")
        got = pd.DataFrame({"EntityId": pd.to_numeric(cards["EntityId"]).astype("int64").to_numpy(),
                            "proba": self.predict_proba(cards)}).merge(ref, on="EntityId")
        delta = float((got["proba"] - got["expected_proba"]).abs().max())
        if delta > tol:
            raise SystemExit(f"Модель курса не воспроизводится: расхождение {delta:.2e} > {tol:.0e}")
        return delta
