"""Install the MAL loss patch and invoke the pinned D-FINE trainer."""
from pathlib import Path
import sys
import runpy


def main():
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root / 'vendor/dfine'), str(root / 'src')]
    from loss_patch import install
    install(0.0, True)
    runpy.run_path(str(root / 'vendor/dfine/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
