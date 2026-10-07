"""Train BR and then BPS for the two remaining frozen MAL seeds."""
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
    path = HERE / "confirmation_status.json"
    if path.exists():
        raise RuntimeError("Existing confirmation status")
    jobs = []
    for seed in [20260930, 20261001]:
        jobs.extend(
            [
                {"name": f"dfine_frozen_BR_seed{seed}", "kind": "BR", "seed": seed, "status": "pending"},
                {"name": f"dfine_frozen_BPS_seed{seed}", "kind": "BPS", "seed": seed, "status": "pending"},
            ]
        )
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
        if job["kind"] == "BR":
            command = [
                PYTHON,
                str(HERE / "train_frozen_refiner.py"),
                "--mode",
                "box",
                "--seed",
                str(job["seed"]),
            ]
        else:
            br_job = next(
                item for item in jobs if item["seed"] == job["seed"] and item["kind"] == "BR"
            )
            if br_job["status"] != "trained":
                job.update(status="blocked_by_BR", finished=now())
                save()
                continue
            command = [
                PYTHON,
                str(HERE / "train_pair_selector.py"),
                "--seed",
                str(job["seed"]),
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
