"""Sequential single-seed boundary refinement pilot."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
PYTHON = "/home/viplab/contest/.detector-venv/bin/python"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> None:
    if not json.loads((HERE / "preflight.json").read_text())["passed"]:
        raise RuntimeError("Preflight did not pass")
    path = HERE / "pilot_status.json"
    if path.exists():
        raise RuntimeError("Existing pilot status")
    jobs = [
        {"name": "dfine_frozen_BR_seed20260929", "mode": "box", "status": "pending"},
        {"name": "dfine_frozen_EBR_seed20260929", "mode": "edge", "status": "pending"},
    ]
    state = {"status": "running", "started": now(), "test_used": False, "jobs": jobs}

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)

    (HERE / "logs").mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    save()
    for job in jobs:
        job.update(status="running", started=now())
        save()
        command = [
            PYTHON,
            str(HERE / "train_frozen_refiner.py"),
            "--mode",
            job["mode"],
            "--seed",
            "20260929",
        ]
        with (HERE / "logs" / f"{job['name']}.log").open("x") as log:
            result = subprocess.run(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        run = HERE / "runs" / job["name"]
        rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()] if run.exists() else []
        job.update(
            status="trained" if result.returncode == 0 and len(rows) == 30 else "failed",
            returncode=result.returncode,
            epochs=len(rows),
            finished=now(),
        )
        save()
    state.update(
        status="finished" if all(job["status"] == "trained" for job in jobs) else "finished_with_failures",
        finished=now(),
    )
    save()


if __name__ == "__main__":
    main()

