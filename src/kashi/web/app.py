"""Web app (ROADMAP §3.7): upload a music video (mp4/mp3/...), watch the
pipeline run, preview the subtitles on the video, download SRT/VTT/ASS/CSV.

Jobs run on a single background worker (the models are not re-entrant on one
GPU); state is in memory; artifacts under <runs_dir>/web/<job_id>/.
"""

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

FORMATS = ("srt", "vtt", "ass", "csv")
QUEUE_CAP = 3


@dataclass
class Job:
    id: str
    filename: str
    dir: Path
    state: str = "queued"          # queued | running | done | error
    stage: str = ""
    frac: float = 0.0
    timings: dict = field(default_factory=dict)
    error: str = ""
    n_syllables: int = 0

    def snapshot(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "state": self.state,
            "stage": self.stage, "frac": self.frac, "timings": self.timings,
            "error": self.error, "n_syllables": self.n_syllables,
        }


def create_app(cfg):
    from ..pipeline import transcribe

    app = FastAPI(title="kashi")
    jobs: dict[str, Job] = {}
    work: "queue.Queue[Job]" = queue.Queue()
    max_bytes = int(cfg["web.max_upload_mb"]) * 1024 * 1024

    def worker() -> None:
        while True:
            job = work.get()
            job.state = "running"
            try:
                def progress(stage: str, frac: float) -> None:
                    job.stage, job.frac = stage, frac

                result = transcribe(
                    cfg, job.dir / job.filename, out_dir=job.dir,
                    formats=list(FORMATS), progress=progress,
                )
                job.timings = result.timings
                job.n_syllables = sum(1 for s in result.segments if not s.is_silence)
                job.state = "done"
                job.frac = 1.0
            except BaseException as e:  # keep the worker alive
                job.state = "error"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                work.task_done()

    threading.Thread(target=worker, daemon=True, name="kashi-worker").start()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/jobs")
    async def create_job(request: Request, file: Annotated[UploadFile, File()]):
        length = int(request.headers.get("content-length", 0))
        if length > max_bytes:
            raise HTTPException(413, f"upload exceeds {cfg['web.max_upload_mb']} MB")
        if sum(1 for j in jobs.values() if j.state in ("queued", "running")) >= QUEUE_CAP:
            raise HTTPException(429, "queue full — try again when a job finishes")
        job_id = uuid.uuid4().hex[:12]
        job_dir = cfg.runs_dir / "web" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "input").name or "input"
        dest = job_dir / name
        written = 0
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > max_bytes:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "upload too large")
                f.write(chunk)
        job = Job(id=job_id, filename=name, dir=job_dir)
        jobs[job_id] = job
        work.put(job)
        return {"job_id": job_id}

    def _job(job_id: str) -> Job:
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id]

    @app.get("/jobs/{job_id}")
    def job_state(job_id: str) -> dict:
        return _job(job_id).snapshot()

    @app.get("/jobs/{job_id}/media")
    def job_media(job_id: str):
        job = _job(job_id)
        return FileResponse(job.dir / job.filename)

    @app.get("/jobs/{job_id}/files/{fmt}")
    def job_file(job_id: str, fmt: str):
        job = _job(job_id)
        if fmt not in FORMATS:
            raise HTTPException(404, f"format must be one of {FORMATS}")
        path = job.dir / f"{Path(job.filename).stem}.{fmt}"
        if not path.is_file():
            raise HTTPException(404, "not ready")
        return FileResponse(path, filename=path.name)

    return app


def serve(cfg, host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(
        create_app(cfg),
        host=host or cfg["web.host"],
        port=int(port or cfg["web.port"]),
        log_level="info",
    )
