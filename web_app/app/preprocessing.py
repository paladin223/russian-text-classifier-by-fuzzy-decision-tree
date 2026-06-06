import re

try:
    import pymorphy3 as pymorphy2
except ImportError:
    try:
        import pymorphy2  # type: ignore[reportMissingImports]
    except ImportError:
        pymorphy2 = None

_morph = pymorphy2.MorphAnalyzer() if pymorphy2 else None


def is_lemmatizer_enabled() -> bool:
    return _morph is not None


def clean_text(text: str) -> str:
    # Единая предобработка для обучения и веб-инференса.
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^а-яА-ЯёЁ ]", " ", text).lower()
    words = [w for w in text.split() if len(w) > 2]
    if not words:
        return ""

    if _morph is None:
        return " ".join(words)

    lemmas = [_morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)
