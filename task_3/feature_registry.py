"""
feature_registry.py — реестр 73 полей модели `lgbm_gate1__v001` (замена docs/feature_registry.md).

Каждое из 73 `source_column` из feature_mapping.csv описано:
  * label_ru      — понятное аналитику SOC название;
  * kind          — "card" (исходное поле карточки) или "engineered" (сконструированный признак);
  * last_step     — шаг препроцессора (QuantileTransformer / OneHotEncoder / TfidfVectorizer / PCA / DateTimeFeatures);
  * raw_hint      — исходное поле или формула (для подсказки в таблице карточки);
  * xai_methods   — какие методы объяснения математически применимы (см. xai_methods.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FieldMeta:
    source_column: str
    label_ru: str
    kind: str                       # "card" | "engineered"
    last_step: str
    raw_hint: str
    xai_methods: tuple[str, ...]

    @property
    def kind_label(self) -> str:
        return "поле карточки" if self.kind == "card" else "сконструирован"


_XAI_NUM = ("SHAP", "ALE", "PDP", "ICE", "Ablation")
_XAI_CAT = ("SHAP", "group-PDP", "Ablation")
_XAI_TXT = ("SHAP", "Ablation")
_XAI_PCA = ("SHAP", "Ablation")
_XAI_DT = ("SHAP", "Ablation")

# Порядок совпадает с заголовком course_pack/shap_cards.csv (ровно 73 поля)
_SPECS: list[tuple[str, str, str, str, str, tuple[str, ...]]] = [
    # --- DateTimeFeatures (1) ---
    ("Создан", "Время создания карточки", "card", "DateTimeFeatures", "Создан (час, день недели, выходной)", _XAI_DT),

    # --- QuantileTransformer / engineered (43) ---
    ("event_duration_s", "Длительность события, с", "engineered", "QuantileTransformer", "last_event − first_event", _XAI_NUM),
    ("create_lag_s", "Задержка создания карточки, с", "engineered", "QuantileTransformer", "Создан − last_event", _XAI_NUM),
    ("event_age_days", "Возраст события, дн.", "engineered", "QuantileTransformer", "Создан − first_event (в днях)", _XAI_NUM),
    ("len_src", "Число IP-источников (len_src)", "engineered", "QuantileTransformer", "размер списка src", _XAI_NUM),
    ("len_dst", "Число IP-целей (len_dst)", "engineered", "QuantileTransformer", "размер списка dst", _XAI_NUM),
    ("len_suser", "Число учётных записей источника", "engineered", "QuantileTransformer", "размер списка suser", _XAI_NUM),
    ("len_duser", "Число учётных записей цели", "engineered", "QuantileTransformer", "размер списка duser", _XAI_NUM),
    ("len_CMDLINE", "Число строк CMDLINE", "engineered", "QuantileTransformer", "размер списка CMDLINE", _XAI_NUM),
    ("len_Команда", "Число записей «Команда»", "engineered", "QuantileTransformer", "размер списка Команда", _XAI_NUM),
    ("len_commandName", "Число имён команд", "engineered", "QuantileTransformer", "размер списка commandName", _XAI_NUM),
    ("len_processName", "Число имён процессов", "engineered", "QuantileTransformer", "размер списка processName", _XAI_NUM),
    ("len_FilePath", "Число путей к файлам", "engineered", "QuantileTransformer", "размер списка FilePath", _XAI_NUM),
    ("len_FileName", "Число имён файлов", "engineered", "QuantileTransformer", "размер списка FileName", _XAI_NUM),
    ("len_objectLinks", "Число связей объектов", "engineered", "QuantileTransformer", "размер списка objectLinks", _XAI_NUM),
    ("len_Алерт", "Число алертов в карточке", "engineered", "QuantileTransformer", "размер списка Алерт", _XAI_NUM),
    ("len_Значение Инцидента (группировка)", "Число значений инцидента", "engineered", "QuantileTransformer", "размер списка «Значение Инцидента»", _XAI_NUM),
    ("len_Значение Атаки (группировка)", "Число значений атаки", "engineered", "QuantileTransformer", "размер списка «Значение Атаки»", _XAI_NUM),
    ("n_rules", "Число правил корреляции", "engineered", "QuantileTransformer", "число правил SIEM в карточке", _XAI_NUM),
    ("is_business_hours", "Рабочие часы (флаг)", "engineered", "QuantileTransformer", "из времени создания карточки", _XAI_NUM),
    ("is_night", "Ночное время (флаг)", "engineered", "QuantileTransformer", "из времени создания карточки", _XAI_NUM),
    ("alert_chars", "Длина текста алерта, симв.", "engineered", "QuantileTransformer", "число символов в поле «Алерт»", _XAI_NUM),
    ("alert_words", "Число слов в алерте", "engineered", "QuantileTransformer", "число слов в поле «Алерт»", _XAI_NUM),
    ("cmd_chars", "Длина командной строки, симв.", "engineered", "QuantileTransformer", "число символов в CMDLINE", _XAI_NUM),
    ("cmd_words", "Число токенов в CMDLINE", "engineered", "QuantileTransformer", "число слов в CMDLINE", _XAI_NUM),
    ("cmd_entropy", "Энтропия командной строки", "engineered", "QuantileTransformer", "энтропия Шеннона по CMDLINE", _XAI_NUM),
    ("cmd_flags", "Число флагов CLI в команде", "engineered", "QuantileTransformer", "подсчёт флагов (- / --) в CMDLINE", _XAI_NUM),
    ("cmd_has_url", "URL в командной строке (флаг)", "engineered", "QuantileTransformer", "наличие http/https в CMDLINE", _XAI_NUM),
    ("cmd_has_b64", "Base64 / -enc в команде (флаг)", "engineered", "QuantileTransformer", "признак обфускации в CMDLINE", _XAI_NUM),
    ("cmd_has_pipe", "Пайп | в команде (флаг)", "engineered", "QuantileTransformer", "наличие перенаправления | в CMDLINE", _XAI_NUM),
    ("path_depth", "Глубина пути процесса", "engineered", "QuantileTransformer", "число каталогов в ProcessPath", _XAI_NUM),
    ("path_in_temp", "Запуск из Temp / AppData (флаг)", "engineered", "QuantileTransformer", "подозрительный каталог в ProcessPath", _XAI_NUM),
    ("ip_count_src", "Число уникальных IP в src", "engineered", "QuantileTransformer", "парсинг IP из поля src", _XAI_NUM),
    ("ip_count_dst", "Число уникальных IP в dst", "engineered", "QuantileTransformer", "парсинг IP из поля dst", _XAI_NUM),
    ("src_is_private", "Источник — приватный IP (флаг)", "engineered", "QuantileTransformer", "RFC1918 для IP в src", _XAI_NUM),
    ("dst_is_private", "Цель — приватный IP (флаг)", "engineered", "QuantileTransformer", "RFC1918 для IP в dst", _XAI_NUM),
    ("dst_external_count", "Число внешних IP в dst", "engineered", "QuantileTransformer", "внешние адреса в поле dst", _XAI_NUM),
    ("host_depth", "Глубина доменного имени хоста", "engineered", "QuantileTransformer", "уровни FQDN хоста", _XAI_NUM),
    ("host_is_server", "Хост похож на сервер (флаг)", "engineered", "QuantileTransformer", "паттерн имени сервера в Hostname", _XAI_NUM),
    ("has_suser", "Заполнено suser (флаг)", "engineered", "QuantileTransformer", "наличие значения в suser", _XAI_NUM),
    ("has_duser", "Заполнено duser (флаг)", "engineered", "QuantileTransformer", "наличие значения в duser", _XAI_NUM),
    ("has_malicious", "Есть вредоносные команды (флаг)", "engineered", "QuantileTransformer", "наличие «Список вредоносных команд»", _XAI_NUM),
    ("freq_Наименование", "Частота правила в обучении", "engineered", "QuantileTransformer", "freq_maps[Наименование] (только train)", _XAI_NUM),
    ("freq_Вендор", "Частота вендора в обучении", "engineered", "QuantileTransformer", "freq_maps[Вендор из сием] (только train)", _XAI_NUM),

    # --- TfidfVectorizer / text (13) ---
    ("CMDLINE", "Командная строка (CMDLINE)", "card", "TfidfVectorizer", "CMDLINE (TF-IDF, 50 фич)", _XAI_TXT),
    ("Команда", "Команда", "card", "TfidfVectorizer", "Команда (TF-IDF, 50 фич)", _XAI_TXT),
    ("commandName", "Имя команды (commandName)", "card", "TfidfVectorizer", "commandName (TF-IDF, 50 фич)", _XAI_TXT),
    ("processName", "Имя процесса (processName)", "card", "TfidfVectorizer", "processName (TF-IDF, 50 фич)", _XAI_TXT),
    ("Имя родителя процесса", "Имя родителя процесса", "card", "TfidfVectorizer", "Имя родителя процесса (TF-IDF, 50 фич)", _XAI_TXT),
    ("Путь родителя процесса", "Путь родителя процесса", "card", "TfidfVectorizer", "Путь родителя процесса (TF-IDF, 50 фич)", _XAI_TXT),
    ("ProcessPath", "Путь процесса (ProcessPath)", "card", "TfidfVectorizer", "ProcessPath (TF-IDF, 50 фич)", _XAI_TXT),
    ("object.name", "Имя объекта (object.name)", "card", "TfidfVectorizer", "object.name (TF-IDF, 50 фич)", _XAI_TXT),
    ("objectValue", "Значение объекта (objectValue)", "card", "TfidfVectorizer", "objectValue (TF-IDF, 50 фич)", _XAI_TXT),
    ("objectLinks", "Связи объекта (objectLinks)", "card", "TfidfVectorizer", "objectLinks (TF-IDF, 50 фич)", _XAI_TXT),
    ("suser", "Пользователь-источник (suser)", "card", "TfidfVectorizer", "suser (TF-IDF, 50 фич)", _XAI_TXT),
    ("duser", "Пользователь-цель (duser)", "card", "TfidfVectorizer", "duser (TF-IDF, 50 фич)", _XAI_TXT),
    ("Алерт", "Текст алерта SIEM", "card", "TfidfVectorizer", "Алерт (обогащение, TF-IDF, 50 фич)", _XAI_TXT),

    # --- OneHotEncoder / categorical (12) ---
    ("Наименование", "Наименование (правило SIEM)", "card", "OneHotEncoder", "Наименование (OHE, 85 фич)", _XAI_CAT),
    ("categoryMain", "Основная категория (categoryMain)", "card", "OneHotEncoder", "categoryMain (OHE, 28 фич)", _XAI_CAT),
    ("categoryHigh", "Верхняя категория (categoryHigh)", "card", "OneHotEncoder", "categoryHigh (OHE, 43 фичи)", _XAI_CAT),
    ("categoryLow", "Детальная категория (categoryLow)", "card", "OneHotEncoder", "categoryLow (OHE, 67 фич)", _XAI_CAT),
    ("Пиктограмма IRP", "Пиктограмма IRP", "card", "OneHotEncoder", "Пиктограмма IRP (OHE, 9 фич)", _XAI_CAT),
    ("Критичность", "Критичность карточки", "card", "OneHotEncoder", "Критичность (OHE, 4 фичи)", _XAI_CAT),
    ("Критичность инцидента в источнике", "Критичность в источнике", "card", "OneHotEncoder", "Критичность в источнике (OHE, 3 фичи)", _XAI_CAT),
    ("Источник инцидента", "Источник инцидента (SIEM)", "card", "OneHotEncoder", "Источник инцидента (OHE, 1 фича)", _XAI_CAT),
    ("Вендор из сием", "Вендор из SIEM", "card", "OneHotEncoder", "Вендор из сием (обогащение, OHE, 8 фич)", _XAI_CAT),
    ("Конвейер SIEM", "Конвейер SIEM", "card", "OneHotEncoder", "Конвейер SIEM (OHE, 9 фич)", _XAI_CAT),
    ("Организация", "Организация", "card", "OneHotEncoder", "Организация (OHE, 5 фич)", _XAI_CAT),
    ("object", "Тип объекта (object)", "card", "OneHotEncoder", "object (OHE, 21 фича)", _XAI_CAT),

    # --- PCA / network & hosts (4) ---
    ("src", "IP-адрес источника (src)", "card", "PCA", "src (PCA, 30 фич)", _XAI_PCA),
    ("dst", "IP-адрес цели (dst)", "card", "PCA", "dst (PCA, 30 фич)", _XAI_PCA),
    ("Hostname Цели", "Имя хоста цели", "card", "PCA", "Hostname Цели (PCA, 30 фич)", _XAI_PCA),
    ("Hostname атакующего", "Имя хоста атакующего", "card", "PCA", "Hostname атакующего (PCA, 30 фич)", _XAI_PCA),
]

REGISTRY: dict[str, FieldMeta] = {s[0]: FieldMeta(*s) for s in _SPECS}
FEATURE_ORDER_73: list[str] = [s[0] for s in _SPECS]
NUMERIC_FEATURES_43: list[str] = [s[0] for s in _SPECS if s[3] == "QuantileTransformer"]
CAT_FEATURES_12: list[str] = [s[0] for s in _SPECS if s[3] == "OneHotEncoder"]
TEXT_FEATURES_13: list[str] = [s[0] for s in _SPECS if s[3] == "TfidfVectorizer"]
PCA_FEATURES_4: list[str] = [s[0] for s in _SPECS if s[3] == "PCA"]


def registry_frame() -> pd.DataFrame:
    """Справочник 73 полей в виде DataFrame (сохраняется в data/agg/feature_registry.parquet)."""
    return pd.DataFrame([
        {
            "source_column": m.source_column,
            "label_ru": m.label_ru,
            "kind": m.kind,
            "kind_label": m.kind_label,
            "last_step": m.last_step,
            "raw_hint": m.raw_hint,
            "xai_methods": ", ".join(m.xai_methods),
        }
        for m in REGISTRY.values()
    ])


def load_feature_mapping(path: Path) -> pd.DataFrame:
    """Читает feature_mapping.csv (1099 строк) и проверяет точное совпадение с 73 полями реестра."""
    fm = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    fm = fm.sort_values("feature_index").reset_index(drop=True)
    found = set(fm["source_column"].unique())
    expected = set(FEATURE_ORDER_73)
    if found != expected:
        raise ValueError(f"Несовпадение feature_mapping.csv с реестром: "
                         f"лишние={found - expected}, не хватает={expected - found}")
    return fm


def build_group_matrix(fm: pd.DataFrame) -> np.ndarray:
    """Матрица M (1099 × 73) для аддитивного суммирования SHAP: shap_73 = shap_1099 @ M."""
    col_to_pos = {col: idx for idx, col in enumerate(FEATURE_ORDER_73)}
    M = np.zeros((len(fm), len(FEATURE_ORDER_73)), dtype=np.float64)
    for row in fm.itertuples(index=False):
        M[int(row.feature_index), col_to_pos[row.source_column]] = 1.0
    return M


def source_to_indices(fm: pd.DataFrame) -> dict[str, np.ndarray]:
    """Словарь source_column -> массив индексов столбцов в матрице X (1099)."""
    out: dict[str, list[int]] = {c: [] for c in FEATURE_ORDER_73}
    for row in fm.itertuples(index=False):
        out[row.source_column].append(int(row.feature_index))
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}