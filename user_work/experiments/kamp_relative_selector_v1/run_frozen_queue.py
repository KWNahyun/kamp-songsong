"""Sequential queue for the frozen-base selector comparison."""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
PYTHON = "/home/viplab/contest/.detector-venv/bin/python"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> None:
    lock = (HERE / "frozen_queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = HERE / "frozen_queue_status.json"
    if path.exists():
        raise RuntimeError("Existing frozen queue state")
    jobs = [
        {"name": "dfine_frozen_UQ_seed20260929", "mode": "unary", "status": "pending"},
        {"name": "dfine_frozen_RQS_seed20260929", "mode": "relation", "status": "pending"},
    ]
    state = {"status": "running", "started": now(), "test_used": False, "jobs": jobs}

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)

    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    (HERE / "logs").mkdir(exist_ok=True)
    save()
    for job in jobs:
        job.update(status="running", started=now())
        save()
        command = [PYTHON, str(HERE / "train_frozen_selector.py"), "--mode", job["mode"]]
        with (HERE / "logs" / f"{job['name']}.log").open("x") as log:
            result = subprocess.run(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        job.update(returncode=result.returncode, finished=now())
        run = HERE / "runs_frozen" / job["name"]
        rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()] if run.exists() else []
        if result.returncode == 0 and len(rows) == 30 and (run / "last.pth").exists():
            job["status"] = "trained"
        else:
            job.update(status="failed", epochs=len(rows))
        save()
    state.update(
        status="finished" if all(job["status"] == "trained" for job in jobs) else "finished_with_failures",
        finished=now(),
    )
    save()


if __name__ == "__main__":
    main()

