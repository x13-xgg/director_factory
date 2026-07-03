"""导演工厂 — REST API 服务 (FastAPI / Starlette)"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.config import config
from src.core.logging import get_logger

log = get_logger("API")


def create_app():
    """创建 ASGI 应用 (API + Web UI)"""
    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, PlainTextResponse
        from starlette.routing import Route, Mount
        from starlette.staticfiles import StaticFiles
    except ImportError:
        log.warn("starlette 未安装，使用简易 HTTP 服务。pip install starlette uvicorn")
        return _create_fallback_app()

    from src.web.routes import routes as web_routes

    STATIC_DIR = Path(__file__).parent / "web" / "static"
    OUTPUT_DIR = Path(config.output_dir).resolve()
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def health(request):
        return PlainTextResponse("ok")

    async def info(request):
        return JSONResponse(config.to_dict())

    async def run_pipeline(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        prompt = body.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        from src.pipeline.runner import PipelineRunner
        runner = PipelineRunner(output_dir=body.get("output_dir", config.output_dir))
        result = await runner.run(
            prompt,
            genre=body.get("genre", "drama"),
            duration_hint=body.get("duration_hint", 60),
            style_ref=body.get("style_ref", ""),
            quality_threshold=body.get("quality_threshold", 0.85),
            language=body.get("language", "zh"),
        )
        return JSONResponse(result)

    async def metrics_handler(request):
        if not config.logging.metrics_enabled:
            return PlainTextResponse("metrics disabled", status_code=404)
        try:
            from prometheus_client import generate_latest, REGISTRY
            return PlainTextResponse(
                generate_latest(REGISTRY).decode("utf-8"),
                media_type="text/plain",
            )
        except ImportError:
            return PlainTextResponse("prometheus_client not installed", status_code=500)

    async def favicon(request):
        return PlainTextResponse("", status_code=404)

    routes = [
        Route("/health", health),
        Route("/info", info),
        Route("/run", run_pipeline, methods=["POST"]),
        Route("/metrics", metrics_handler),
        Route("/favicon.ico", favicon),
        Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
        Mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR.resolve())), name="outputs"),
    ] + list(web_routes.routes)

    app = Starlette(routes=routes)
    log.info("API + Web UI 应用已创建 (Starlette)")
    return app


def _create_fallback_app():
    """无 Starlette 时的简易 WSGI/ASGI 回退"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import asyncio
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._respond(200, "ok")
            elif self.path == "/info":
                self._respond(200, json.dumps(config.to_dict(), indent=2), "application/json")
            elif self.path == "/metrics":
                self._respond(200, "# metrics disabled\n")
            else:
                self._respond(404, "not found")

        def do_POST(self):
            if self.path == "/run":
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
                prompt = body.get("prompt", "")
                if not prompt:
                    self._respond(400, json.dumps({"error": "prompt required"}))
                    return
                self._respond(200, json.dumps({"status": "queued", "prompt": prompt[:100]}))
            else:
                self._respond(404, "not found")

        def _respond(self, code: int, body: str, content_type: str = "text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format, *args):
            pass  # suppress logs

    class FallbackApp:
        def __init__(self):
            self._server = None

        async def __call__(self, scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        port = 8000
                        self._server = HTTPServer(("0.0.0.0", port), Handler)
                        t = threading.Thread(target=self._server.serve_forever, daemon=True)
                        t.start()
                        log.info(f"简易 HTTP 服务已启动: http://0.0.0.0:{port}")
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        if self._server:
                            self._server.shutdown()
                        await send({"type": "lifespan.shutdown.complete"})
                        return

    return FallbackApp()
