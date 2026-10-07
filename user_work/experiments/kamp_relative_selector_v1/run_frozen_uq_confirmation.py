"""Train the frozen unary-quality selector on the two remaining MAL seeds."""
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
    path = HERE / "frozen_uq_confirmation_status.json"
    if path.exists():
        raise RuntimeError("Existing confirmation status")
    jobs = [
        {"name": f"dfine_frozen_UQ_seed{seed}", "seed": seed, "status": "pending"}
        for seed in [20260930, 20261001]
    ]
    state = {"status": "running", "started": now(), "test_used": False, "jobs": jobs}

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(path)

    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    save()
    for job in jobs:
        job.update(status="running", started=now())
        save()
        command = [
            PYTHON,
            str(HERE / "train_frozen_selector.py"),
            "--mode",
            "unary",
            "--seed",
            str(job["seed"]),
        ]
        with (HERE / "logs" / f"{job['name']}.log").open("x") as log:
            result = subprocess.run(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
        run = HERE / "runs_frozen" / job["name"]
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
