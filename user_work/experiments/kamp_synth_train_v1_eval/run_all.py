from __future__ import annotations

import concurrent.futures
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
PYTHON = ROOT / ".detector-venv/bin/python"
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def work(job: dict) -> tuple[str, bool, str | None]:
    name = job["name"]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="1")
    stages = [
        ("synthetic_inference", [str(PYTHON), str(HERE / "infer.py"), name, "--batch", "1"]),
        ("synthetic_analysis", [str(PYTHON), str(HERE / "analyze_run.py"), name]),
        ("real_evaluation", [str(PYTHON), str(HERE / "evaluate_real.py"), name]),
    ]
    for stage, command in stages:
        if stage == "synthetic_inference" and (HERE / "runs" / name / "complete.json").exists():
            continue
        if stage == "synthetic_analysis" and (HERE / "runs" / name / "analysis/complete.json").exists():
            continue
        if stage == "real_evaluation" and (HERE / "real" / name / "metrics.json").exists():
            continue
        with (HERE / f"{name}_{stage}.log").open("a") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        if result.returncode:
            return name, False, stage
    return name, True, None


def main() -> None:
    started = time.time()
    status = {"started": now(), "jobs": {job["name"]: {"status": "queued"} for job in PROTOCOL["jobs"]}}
    (HERE / "status.json").write_text(json.dumps(status, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(work, job): job["name"] for job in PROTOCOL["jobs"]}
        for future in concurrent.futures.as_completed(futures):
            name, ok, failed_stage = future.result()
            status["jobs"][name] = {"status": "complete" if ok else "failed", "failed_stage": failed_stage, "updated": now()}
            (HERE / "status.json").write_text(json.dumps(status, indent=2))
    status.update(finished=now(), seconds=time.time() - started, complete=all(item["status"] == "complete" for item in status["jobs"].values()))
    (HERE / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
