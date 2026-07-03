"""Web UI 路由 — 页面 + API 端点"""

import json
import time
from pathlib import Path

from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse
from starlette.routing import Route, Router
from starlette.templating import Jinja2Templates

from src.core.config import config
from src.core.logging import get_logger

log = get_logger("WebUI")

TEMPLATE_DIR = Path(__file__).parent / "templates"

def _inject_config(request):
    return {"config": config.to_dict()}

templates = Jinja2Templates(directory=str(TEMPLATE_DIR), context_processors=[_inject_config])

OUTPUT_DIR = Path(config.output_dir)


def _read_json(path: Path) -> dict | None:
    """读取 JSON 文件，自动处理 UTF-8 / GBK 编码"""
    if not path.exists():
        return None
    raw = path.read_bytes()
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


# ── Helpers ────────────────────────────────────────────


def _scan_projects() -> list[dict]:
    """扫描 outputs/ 目录，收集所有项目信息"""
    projects = []
    if not OUTPUT_DIR.exists():
        return projects

    for item in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        summary_path = item / "summary.json" if item.is_dir() else None
        if not summary_path or not summary_path.exists():
            continue

        try:
            data = _read_json(summary_path)
            if data is None:
                continue
            stats = data.get("stats", {})
            status = data.get("status", "unknown")
            title = data.get("title", item.name)

            # 计算进度 (非 ok 状态下最高 99%)
            planned = stats.get("shots_planned", 0)
            completed = stats.get("shots_completed", 0)
            if planned > 0:
                progress = int(completed / planned * 100)
                if progress >= 100 and status != "ok":
                    progress = 99
            else:
                progress = 100 if status == "ok" else 0

            # 计算总分
            frames = data.get("frames", [])
            scores = [f.get("composition_score", 0) for f in frames if f.get("composition_score")]
            avg_score = round(sum(scores) / len(scores), 2) if scores else 0

            output_video = data.get("output_video", "")
            has_video = bool(output_video and Path(output_video).exists())

            projects.append({
                "id": item.name,
                "title": title,
                "status": status,
                "progress": progress,
                "characters": stats.get("characters", 0),
                "scenes": stats.get("scenes", 0),
                "shots_planned": planned,
                "shots_completed": completed,
                "total_duration": stats.get("total_duration", 0),
                "avg_score": avg_score,
                "has_video": has_video,
                "output_video": output_video,
                "modified_at": summary_path.stat().st_mtime,
            })
        except Exception as e:
            log.warn(f"读取项目 {item.name} 失败: {e}")

    return projects


def _find_project(project_id: str) -> dict | None:
    """查找单个项目详情"""
    summary_path = OUTPUT_DIR / project_id / "summary.json"
    if not summary_path.exists():
        # 也尝试直接用 project_id 查找
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and d.name == project_id:
                summary_path = d / "summary.json"
                break
        else:
            return None

    try:
        data = _read_json(summary_path)
    except Exception:
        return None

    if data is None:
        return None

    stats = data.get("stats", {})
    status = data.get("status", "unknown")
    frames = data.get("frames", [])

    # 收集帧文件
    frame_files = []
    frames_dir = OUTPUT_DIR / project_id / "frames"
    if frames_dir.exists():
        for f in sorted(frames_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                frame_files.append({
                    "name": f.name,
                    "path": f"outputs/{project_id}/frames/{f.name}",
                    "size": f.stat().st_size,
                })

    # 收集输出视频
    video_files = []
    for pattern in ["*.mp4", "*.webm", "*.avi"]:
        for f in OUTPUT_DIR.glob(f"{project_id}/{pattern}"):
            video_files.append({
                "name": f.name,
                "path": f"outputs/{project_id}/{f.name}",
                "size": f.stat().st_size,
            })

    # 进度 (非 ok 状态下最高 99%)
    planned = stats.get("shots_planned", 0)
    completed = stats.get("shots_completed", 0)
    if planned > 0:
        progress = int(completed / planned * 100)
        if progress >= 100 and status != "ok":
            progress = 99
    else:
        progress = 100 if status == "ok" else 0

    # 质量分
    scores = [f.get("composition_score", 0) for f in frames if f.get("composition_score")]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0

    # 质量报告
    quality = {
        "avg_score": avg_score,
        "min_score": round(min(scores), 2) if scores else 0,
        "max_score": round(max(scores), 2) if scores else 0,
        "total_frames": len(frames),
        "passed": sum(1 for f in frames if f.get("composition_score", 0) >= 0.82),
        "shots": [],
    }
    for f in frames:
        quality["shots"].append({
            "shot_id": f.get("shot_id", ""),
            "score": f.get("composition_score", 0),
            "gen_method": f.get("metadata", {}).get("gen_method", "unknown"),
        })

    return {
        "id": project_id,
        "title": data.get("title", project_id),
        "status": status,
        "progress": progress,
        "characters": stats.get("characters", 0),
        "scenes": stats.get("scenes", 0),
        "shots_planned": planned,
        "shots_completed": completed,
        "total_duration": stats.get("total_duration", 0),
        "avg_score": avg_score,
        "output_video": data.get("output_video", ""),
        "frames": frame_files,
        "videos": video_files,
        "quality": quality,
        "trace_file": data.get("trace_file", ""),
        "modified_at": (OUTPUT_DIR / project_id / "summary.json").stat().st_mtime,
    }


def _browse_outputs(filter_type: str = "") -> list[dict]:
    """浏览输出文件"""
    files = []
    output_root = OUTPUT_DIR

    for item in sorted(output_root.rglob("*")):
        if item.is_file() and "__pycache__" not in str(item):
            ext = item.suffix.lower()
            ftype = "other"
            if ext in (".png", ".jpg", ".jpeg"):
                ftype = "image"
            elif ext in (".mp4", ".webm", ".avi"):
                ftype = "video"
            elif ext in (".wav", ".mp3", ".ogg"):
                ftype = "audio"
            elif ext in (".srt", ".ass"):
                ftype = "subtitle"
            elif ext in (".json",):
                ftype = "data"

            if filter_type and ftype != filter_type:
                continue

            rel_path = str(item.relative_to(OUTPUT_DIR.parent)).replace("\\", "/")
            files.append({
                "name": item.name,
                "path": rel_path,
                "type": ftype,
                "size": item.stat().st_size,
                "modified": item.stat().st_mtime,
            })

    files.sort(key=lambda f: f["modified"], reverse=True)
    return files[:200]  # 限制 200 个文件


# ── 页面路由 ───────────────────────────────────────────


async def dashboard(request):
    projects = _scan_projects()
    stats = {
        "total": len(projects),
        "completed": sum(1 for p in projects if p["status"] == "ok"),
        "avg_score": round(
            sum(p["avg_score"] for p in projects if p["avg_score"] > 0) / max(1, len([p for p in projects if p["avg_score"] > 0])),
            2,
        ),
    }
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"projects": projects, "stats": stats},
    )


async def new_project_page(request):
    return templates.TemplateResponse(
        request,
        "new_project.html",
    )


async def project_detail(request):
    project_id = request.path_params.get("project_id", "")
    project = _find_project(project_id)
    if not project:
        return HTMLResponse("<h1>项目不存在</h1><a href='/'>返回首页</a>", status_code=404)
    return templates.TemplateResponse(
        request,
        "project_detail.html",
        {"project": project},
    )


async def outputs_page(request):
    filter_type = request.query_params.get("type", "")
    files = _browse_outputs(filter_type)
    return templates.TemplateResponse(
        request,
        "outputs.html",
        {"files": files, "current_filter": filter_type},
    )


# ── API 端点 ───────────────────────────────────────────


async def api_projects(request):
    projects = _scan_projects()
    return JSONResponse(projects)


async def api_project_detail(request):
    project_id = request.path_params.get("project_id", "")
    project = _find_project(project_id)
    if not project:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(project)


async def api_project_frames(request):
    project_id = request.path_params.get("project_id", "")
    frames_dir = OUTPUT_DIR / project_id / "frames"
    if not frames_dir.exists():
        return JSONResponse([])
    frames = []
    for f in sorted(frames_dir.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            frames.append({
                "name": f.name,
                "path": f"outputs/{project_id}/frames/{f.name}",
            })
    return JSONResponse(frames)


async def api_project_trace(request):
    project_id = request.path_params.get("project_id", "")
    trace_path = OUTPUT_DIR / project_id / "trace.json"
    data = _read_json(trace_path)
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(data)


async def api_run(request):
    """接受运行请求，后台启动管线"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    import asyncio
    project_id = f"project_{int(time.time())}"
    project_dir = OUTPUT_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # 立即写入初始 summary，让项目页面立即可访问
    initial_summary = {
        "title": prompt[:80],
        "status": "running",
        "stats": {
            "characters": 0,
            "scenes": 0,
            "shots_planned": 0,
            "shots_completed": 0,
            "total_duration": 0,
        },
        "output_video": "",
        "frames": [],
        "trace_file": str(project_dir / "trace.json"),
    }
    (project_dir / "summary.json").write_text(
        json.dumps(initial_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    async def _run_in_background():
        try:
            from src.pipeline.runner import PipelineRunner
            runner = PipelineRunner(output_dir=str(project_dir))
            result = await runner.run(
                prompt,
                genre=body.get("genre", ""),
                duration_hint=body.get("duration_hint", 60),
                style_ref=body.get("style_ref", ""),
                quality_threshold=body.get("quality_threshold", 0.85),
                language=body.get("language", "zh"),
            )
            # 管线完成后再次写入 summary 确保最新
            (project_dir / "summary.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            log.info(f"项目 {project_id} 管线完成")
        except Exception as e:
            log.error(f"项目 {project_id} 管线失败: {e}")
            error_summary = {**initial_summary, "status": "failed", "error": str(e)}
            (project_dir / "summary.json").write_text(
                json.dumps(error_summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    asyncio.create_task(_run_in_background())

    return JSONResponse({
        "status": "started",
        "project_id": project_id,
    })


# ── 路由表 ─────────────────────────────────────────────


routes = Router([
    # 页面
    Route("/", dashboard, methods=["GET"]),
    Route("/new", new_project_page, methods=["GET"]),
    Route("/project/{project_id}", project_detail, methods=["GET"]),
    Route("/files", outputs_page, methods=["GET"]),
    # API
    Route("/api/projects", api_projects, methods=["GET"]),
    Route("/api/projects/{project_id}", api_project_detail, methods=["GET"]),
    Route("/api/projects/{project_id}/frames", api_project_frames, methods=["GET"]),
    Route("/api/projects/{project_id}/trace", api_project_trace, methods=["GET"]),
    Route("/api/run", api_run, methods=["POST"]),
])
