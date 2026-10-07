"""Run the two pre-registered selector pilots sequentially on one GPU."""
from __future__ import annotations

import datetime
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).parent
PYTHON = "/home/viplab/contest/.detector-venv/bin/python"
INITIAL = "/home/viplab/contest/experiments/kamp_pilot_v1/dfine_s_init.pth"
SEED = 20260929
JOBS = [
    {
        "name": "dfine_UQ_seed20260929",
        "mode": "unary",
        "config": str(HERE / "configs/dfine_UQ_seed20260929.yml"),
    },
    {
        "name": "dfine_RQS_seed20260929",
        "mode": "relation",
        "config": str(HERE / "configs/dfine_RQS_seed20260929.yml"),
    },
]


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main() -> None:
    preflight = json.loads((HERE / "preflight.json").read_text())
    if not preflight["passed"]:
        raise RuntimeError("Preflight did not pass")
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "runs").mkdir(exist_ok=True)
    lock = (HERE / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = HERE / "queue_status.json"
    if status_path.exists():
        raise RuntimeError("Existing queue status; refusing to duplicate trials")
    jobs = []
    for entry in JOBS:
        command = [
            PYTHON,
            str(HERE / "train_dfine.py"),
            "-c",
            entry["config"],
            "-t",
            INITIAL,
            "--device",
            "cuda:0",
            "--seed",
            str(SEED),
            "--selector",
            entry["mode"],
            "--quality-weight",
            "1.0",
        ]
        jobs.append({**entry, "seed": SEED, "command": command, "status": "pending"})
    state = {
        "status": "running",
        "started": now(),
        "test_used": False,
        "validation_used_for_checkpoint_selection": True,
        "jobs": jobs,
    }

    def save() -> None:
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(status_path)

    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )
    save()
    for job in jobs:
        while True:
            free = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                        "-i",
                        "0",
                    ],
                    text=True,
                ).strip()
            )
            if free >= 8000:
                break
            job.update(status="waiting_for_vram", free_vram_MiB=free)
            save()
            time.sleep(30)
        job.update(status="running", started=now())
        save()
        log_path = HERE / "logs" / f"{job['name']}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(
                job["command"], cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            job["pid"] = process.pid
            save()
            return_code = process.wait()
        job.update(returncode=return_code, finished=now())
        run = HERE / "runs" / job["name"]
        try:
            if return_code:
                raise RuntimeError(f"Training exited with {return_code}")
            rows = [json.loads(line) for line in (run / "log.txt").read_text().splitlines() if line]
            if [row["epoch"] for row in rows] != list(range(30)):
                raise RuntimeError("Missing training epochs")
            if not (run / "best_stg1.pth").exists() or not (run / "last.pth").exists():
                raise RuntimeError("Missing checkpoint")
            if not all(
                math.isfinite(value)
                for row in rows
                for key, value in row.items()
                if key.startswith("train_") and isinstance(value, (float, int))
            ):
                raise RuntimeError("Nonfinite training metric")
            job["status"] = "trained"
        except Exception as error:
            job.update(status="failed", error=str(error))
        save()
    state.update(
        status="finished" if all(job["status"] == "trained" for job in jobs) else "finished_with_failures",
        finished=now(),
    )
    save()


if __name__ == "__main__":
    main()

