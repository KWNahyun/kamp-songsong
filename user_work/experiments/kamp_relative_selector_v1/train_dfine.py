"""D-FINE MAL training entry point for detached selector pilots."""
import argparse
import runpy
import sys
from pathlib import Path


ROOT = Path("/home/viplab/contest")
sys.path.insert(0, str(ROOT / "models/D-FINE"))
sys.path.insert(0, str(ROOT / "experiments/kamp_ablation_v3"))
sys.path.insert(0, str(Path(__file__).parent))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--selector", choices=["unary", "relation"], required=True)
parser.add_argument("--quality-weight", type=float, default=1.0)
args, remaining = parser.parse_known_args()

from loss_patch import install as install_mal
from selector_patch import install as install_selector

install_mal(0.0, True)
install_selector(args.selector, args.quality_weight, criterion=True)
sys.argv = [sys.argv[0]] + remaining
runpy.run_path(str(ROOT / "models/D-FINE/train.py"), run_name="__main__")

