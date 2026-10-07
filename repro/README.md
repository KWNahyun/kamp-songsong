# 최종 MAL+UQ 실행 코드

설치, 데이터, 실행 방법, 검수 상태는 루트의 [README](../README.md)를 읽으세요.

통합 실행:

```bash
python run_all.py --mode check --output outputs/check
python run_all.py --mode predict --output outputs/frozen --device cuda:0
python run_all.py --mode train --output outputs/retrain --device cuda:0
```

`predict`에는 대표 checkpoint, `train`에는 프로젝트의 단일 클래스 초기 checkpoint가 필요합니다. `checkpoints/README.md`와 `manifests/`의 해시를 확인하세요.

세부 단계는 기존 `repro/run.sh`로도 실행할 수 있습니다. `PYTHON=.venv/bin/python`을 지정해 준비한 환경을 사용하세요.
