"""Persistent, cancellable subprocess jobs for the existing workflows."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import uuid


class Jobs:
    def __init__(self, workspace):
        self.workspace = workspace
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.processes = {}
        self.records = {}
        folder = workspace.state_dir / "jobs"
        if folder.exists():
            for path in folder.glob("*/job.json"):
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] in {"queued", "running"}:
                    job["status"] = "interrupted"
                self.records[job["id"]] = job

    def save(self, job):
        path = self.workspace.state_dir / "jobs" / job["id"] / "job.json"
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temp.replace(path)

    def start(self, operation: str, payload: dict):
        with self.lock:
            job_id = uuid.uuid4().hex[:12]
            folder = self.workspace.state_dir / "jobs" / job_id
            folder.mkdir(parents=True, exist_ok=False)
            spec = {"operation": operation, "payload": payload}
            (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            job = {"id": job_id, "operation": operation, "status": "queued", "created": time.time()}
            self.records[job_id] = job
            self.save(job)
            self.pool.submit(self.run, job_id)
            return dict(job)

    def run(self, job_id):
        folder = self.workspace.state_dir / "jobs" / job_id
        with self.lock:
            job = self.records[job_id]
            if job["status"] == "cancelled":
                return
            command = [sys.executable, "-m", "tomogui.worker", "--workspace", str(self.workspace.root),
                       "--spec", str(folder / "spec.json"), "--result", str(folder / "result.json")]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
            env["MPLCONFIGDIR"] = str(self.workspace.state_dir / "matplotlib")
            env["PYTHONUNBUFFERED"] = "1"
            try:
                with (folder / "log.txt").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=self.workspace.root, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=os.name != "nt",
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                    self.processes[job_id] = process
                    job["status"] = "running"
                    self.save(job)
            except OSError as exc:
                job.update(status="failed", error=str(exc))
                self.save(job)
                return
        code = process.wait()
        with self.lock:
            self.processes.pop(job_id, None)
            if job["status"] == "cancelled":
                return
            job.update(status="completed" if code == 0 else "failed", exit_code=code, finished=time.time())
            result_path = folder / "result.json"
            if code == 0 and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                job["result"] = result
                if result.get("state_updates"):
                    self.workspace.update_state(**result["state_updates"])
            else:
                job["error"] = (folder / "log.txt").read_text(encoding="utf-8", errors="replace")[-5000:]
            self.save(job)

    def list(self):
        with self.lock:
            # Status polling does not need full dataset samples and model grids.
            return [{key: value for key, value in job.items() if key != 'result'}
                    for job in sorted(self.records.values(), key=lambda j: j["created"], reverse=True)][:60]

    def log(self, job_id):
        if job_id not in self.records:
            raise ValueError("Unknown job.")
        path = self.workspace.state_dir / "jobs" / job_id / "log.txt"
        if not path.exists():
            return "Queued."
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 40000))
            return handle.read().decode("utf-8", errors="replace")

    def cancel(self, job_id):
        with self.lock:
            job = self.records[job_id]
            if job["status"] not in {"queued", "running"}:
                return dict(job)
            job["status"] = "cancelled"
            process = self.processes.get(job_id)
            if process and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            self.save(job)
            return dict(job)
