"""Evaluate trained selector pilots sequentially."""
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
status = json.loads((HERE / "queue_status.json").read_text())
for job in status["jobs"]:
    if job["status"] != "trained":
        continue
    output = HERE / "runs" / job["name"] / "common_eval/metrics.json"
    if output.exists():
        continue
    log_path = HERE / "logs" / f"{job['name']}_evaluation.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, str(HERE / "evaluate_selectors.py"), job["name"], job["mode"]],
            cwd=HERE,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Evaluation failed for {job['name']}")
    print(job["name"], "evaluated", flush=True)

