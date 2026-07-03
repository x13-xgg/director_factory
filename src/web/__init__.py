"""Web UI 管理面板 — Starlette + Jinja2 SSR"""

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from pathlib import Path

from src.web.routes import routes

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_web_app() -> Starlette:
    app = Starlette(routes=[
        Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
        Mount("/", routes),
    ])
    return app
