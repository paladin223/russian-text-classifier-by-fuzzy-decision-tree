import os
import pickle
import math
from pathlib import Path
from typing import Any

from .preprocessing import clean_text, is_lemmatizer_enabled

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "fuzzy_text_model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

_pipeline: dict[str, Any] | None = None
_load_error: str | None = None


class FuzzyNode:
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature_name = None
        self.feature_idx = None
        self.branches = {}
        self.is_leaf = False
        self.leaf_value = None


class FuzzyDecisionTree:
    """Совместимый класс только для инференса из pickle."""

    def predict(self, vec):
        votes = {}
        self._traverse(self.root, vec, 1.0, votes)
        if not votes:
            return "Unknown"
        return max(votes, key=votes.get)

    def _traverse(self, node, vec, weight, votes):
        if weight < 0.01:
            return

        if node.is_leaf:
            votes[node.leaf_value] = votes.get(node.leaf_value, 0) + weight
            return

        val = vec[node.feature_idx]
        mu_high = 1.0 / (1.0 + math.exp(-150.0 * (val - 0.03)))
        mu_low = 1.0 - mu_high

        self._traverse(node.branches["Low"], vec, weight * mu_low, votes)
        self._traverse(node.branches["High"], vec, weight * mu_high, votes)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "FuzzyDecisionTree":
            return FuzzyDecisionTree
        if module == "__main__" and name == "FuzzyNode":
            return FuzzyNode
        return super().find_class(module, name)

def _load_pipeline() -> None:
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return

    try:
        if MODEL_PATH.is_dir():
            _load_error = (
                f"Путь модели указывает на папку, а не на .pkl файл: {MODEL_PATH}. "
                "Проверьте volume в docker-compose.yml."
            )
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        with open(MODEL_PATH, "rb") as f:
            _pipeline = CompatUnpickler(f).load()
    except FileNotFoundError:
        _load_error = (
            f"Не найден файл модели: {MODEL_PATH}. "
            "Сначала обучите модель: python3 solv_14.py --mode train"
        )
    except Exception as exc:
        _load_error = f"Ошибка загрузки модели: {exc}"


def get_model_status() -> str:
    _load_pipeline()
    if _pipeline is not None:
        topics = _pipeline.get("topics", [])
        topics_info = f" | Классы: {', '.join(topics)}" if topics else ""
        morph_info = " | Лемматизация: on" if is_lemmatizer_enabled() else " | Лемматизация: off"
        return f"Модель загружена: {MODEL_PATH}{topics_info}{morph_info}"
    return _load_error or "Модель еще не загружена."


def classify_text(text: str) -> tuple[str, str]:
    _load_pipeline()
    if _pipeline is None:
        return "", _load_error or "Модель недоступна."

    raw = (text or "").strip()
    if not raw:
        return "", "Введите текст для классификации."

    cleaned = clean_text(raw)
    if not cleaned:
        return "", "После очистки текст пустой. Введите более длинный русский текст."

    try:
        x_raw = _pipeline["tfidf"].transform([cleaned])
        x_sel = _pipeline["selector"].transform(x_raw).toarray()
        pred = _pipeline["model"].predict(x_sel[0])
        return str(pred), ""
    except Exception as exc:
        return "", f"Ошибка предсказания: {exc}"
