from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .classifier import classify_text, get_model_status

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Text Classifier UI")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _resolve_theme(theme: str | None) -> str:
    return "light" if (theme or "").lower() == "light" else "dark"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, theme: str | None = None):
    context = {
        "request": request,
        "title": "Классификация текстов",
        "model_status": get_model_status(),
        "input_text": "",
        "prediction": "",
        "error": "",
        "theme": _resolve_theme(theme),
    }
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...), theme: str | None = None):
    prediction, error = classify_text(text)
    context = {
        "request": request,
        "title": "Классификация текстов",
        "model_status": get_model_status(),
        "input_text": text,
        "prediction": prediction,
        "error": error,
        "theme": _resolve_theme(theme),
    }
    return templates.TemplateResponse(request, "index.html", context)
