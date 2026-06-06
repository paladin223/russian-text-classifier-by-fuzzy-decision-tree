# Web App (FastAPI + Jinja)

Одностраничное приложение для классификации текста в темной теме.

## Что нужно перед запуском

1. Обучить модель в корне проекта:
   - `python3 solv_14.py --mode train`
2. Скопировать/переименовать модель в `web_app/app/fuzzy_text_model.pkl`.
   - Например: `cp ../fuzzy_text_model_5cls_tuned.pkl app/fuzzy_text_model.pkl`

## Запуск через Docker Compose

Из папки `web_app`:

- `docker compose up --build`

После запуска откройте: `http://localhost:8000`

Приложение использует такую же предобработку текста, как в обучении (`solv_14.py`):
лемматизация `pymorphy` + очистка текста.

## Локальный запуск без Docker

Из папки `web_app`:

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
