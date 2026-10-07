from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/viplab/contest")
HERE = Path(__file__).parent
PYTHON = ROOT / ".detector-venv/bin/python"
MAX_PARALLEL = 3


def command(job: dict) -> list[str]:
    if job["family"] == "yolo":
        return [str(PYTHON), str(HERE / "train_yolo.py"), "--checkpoint", job["checkpoint"], "--seed", str(job["seed"]), "--name", job["name"]]
    if job["family"] == "dfine":
        return [str(PYTHON), str(ROOT / "experiments/kamp_ablation_v3/train_dfine.py"), *job["flags"], "-c", job["config"], "-t", job["checkpoint"], "--device", "cuda:0", "--seed", str(job["seed"])]
    dependency = HERE / "runs" / job["depends_on"]
    detector_checkpoint = dependency / "best_stg1.pth"
    if not detector_checkpoint.exists():
        detector_checkpoint = dependency / "last.pth"
    return [str(PYTHON), str(HERE / "train_uq.py"), "--config", job["config"], "--detector-checkpoint", str(detector_checkpoint), "--uq-checkpoint", job["base_uq_checkpoint"], "--output", str(HERE / "runs" / job["name"]), "--seed", str(job["seed"])]


def save_status(status: dict) -> None:
    (HERE / "queue_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    jobs = protocol["jobs"]
    status = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_parallel": MAX_PARALLEL,
        "jobs": {job["name"]: {"status": "pending", "family": job["family"], "seed": job["seed"]} for job in jobs},
    }
    save_status(status)
    pending = {job["name"]: job for job in jobs}
    running: dict[str, tuple[subprocess.Popen, object, float]] = {}
    completed: set[str] = set()

    while pending or running:
        for name, job in list(pending.items()):
            if len(running) >= MAX_PARALLEL:
                break
            dependency = job.get("depends_on")
            if dependency and dependency not in completed:
                continue
            output = HERE / "runs" / name
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            log_path = HERE / "logs" / f"{name}.log"
            log_handle = log_path.open("w")
            cmd = command(job)
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
            running[name] = (process, log_handle, time.time())
            status["jobs"][name].update({"status": "running", "pid": process.pid, "command": cmd, "log": str(log_path)})
            pending.pop(name)
            save_status(status)

        for name, (process, log_handle, started) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            status["jobs"][name].update({"status": "complete" if return_code == 0 else "failed", "return_code": return_code, "wall_seconds": time.time() - started})
            running.pop(name)
            if return_code == 0:
                completed.add(name)
            else:
                # Do not start a dependent UQ run from a failed MAL job.
                for pending_name, pending_job in list(pending.items()):
                    if pending_job.get("depends_on") == name:
                        status["jobs"][pending_name].update({"status": "blocked", "reason": f"dependency {name} failed"})
                        pending.pop(pending_name)
            save_status(status)
        time.sleep(2)

    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status["complete"] = all(item["status"] == "complete" for item in status["jobs"].values())
    save_status(status)
    return_codes = [item.get("return_code", 1) for item in status["jobs"].values()]
    sys.exit(0 if all(code == 0 for code in return_codes) else 1)


if __name__ == "__main__":
    main()
