"""HTTP API over the same Pipeline the CLI uses (for the Laravel integration)."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.timeutil import parse_timestamp, to_iso


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="DeckWatch")

    @app.exception_handler(PipelineError)
    def _pipeline_error(request: Request, exc: PipelineError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.post("/frames")
    def post_frame(file: UploadFile = File(...), ts: str = Form(...), items: bool = Query(False)):
        try:
            dt = parse_timestamp(ts, pipeline.config.filename_tz)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid ts: {ts}", 422) from exc
        return pipeline.analyze(file.file.read(), dt, path=file.filename, include_items=items)

    @app.get("/status")
    def status():
        return pipeline.status()

    @app.get("/operations")
    def operations(since: str | None = None):
        try:
            return pipeline.operations(since)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid since: {since}", 422) from exc

    @app.get("/frames/{ts}")
    def frame(ts: str, items: bool = Query(False)):
        try:
            key = to_iso(parse_timestamp(ts, pipeline.config.filename_tz))
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid ts: {ts}", 422) from exc
        result = pipeline.frame(key, include_items=items)
        if result is None:
            return JSONResponse(status_code=404, content={"error": "not_found", "message": f"no frame at {key}"})
        return result

    return app


def serve(config_path: str, host: str, port: int) -> None:
    import uvicorn

    from deckwatch.cli import build_pipeline

    uvicorn.run(create_app(build_pipeline(config_path)), host=host, port=port)
