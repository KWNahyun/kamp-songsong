from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = "/home/viplab/contest/experiments/kamp_synth_train_v1/ultralytics_settings"

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer


class FixedSquareTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    import ultralytics.data.build as build
    import ultralytics.models.yolo.detect.train as detect_train

    source = inspect.getsource(build.build_dataloader)
    assert "6148914691236517205 + RANK" in source
    source = source.replace("6148914691236517205 + RANK", f"{args.seed} + RANK")
    namespace = dict(build.__dict__)
    exec(compile(source, "<seeded_build_dataloader>", "exec"), namespace)
    detect_train.build_dataloader = namespace["build_dataloader"]

    root = Path("/home/viplab/contest")
    model = YOLO(str(args.checkpoint))
    model.train(
        trainer=FixedSquareTrainer,
        data=str(root / "experiments/kamp_synth_train_v1/data640_mix25/data.yaml"),
        project=str(root / "experiments/kamp_synth_train_v1/runs"),
        name=args.name,
        epochs=12,
        patience=0,
        imgsz=640,
        batch=8,
        device=0,
        workers=1,
        seed=args.seed,
        optimizer="AdamW",
        lr0=5e-5,
        lrf=0.1,
        weight_decay=1e-4,
        nbs=8,
        warmup_epochs=0.5,
        warmup_bias_lr=0.0,
        amp=False,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        close_mosaic=0,
        multi_scale=False,
        rect=False,
        plots=True,
        save=True,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
