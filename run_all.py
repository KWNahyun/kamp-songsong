"""One command for fixed-model validation/test inference or three-seed MAL+UQ training."""
from pathlib import Path
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'repro' / 'scripts'))
from audit_dataset import audit, sha256


def run(command):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def checkpoint(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f'Required checkpoint is missing: {path}')
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Checkpoint SHA256 differs: {path}; expected={expected}, actual={actual}')
    return actual


def predict_evaluate(dataset, output, device, ck=None, seed=20260930):
    for split in ('val', 'test'):
        prediction = output / split / 'prediction'
        cmd = [sys.executable, ROOT / 'repro/predict.py', '--preprocessed', '--device', device,
               '--images', dataset / 'images' / split, '--output', prediction, '--seed', seed]
        if ck is not None:
            cmd += ['--checkpoint', ck, '--checkpoint-sha256', sha256(ck)]
        run(cmd)
        run([sys.executable, ROOT / 'repro/evaluate.py', '--predictions', prediction / 'predictions.json',
             '--annotations', dataset / 'coco' / f'instances_{split}.json', '--output', output / split / 'metrics'])
        export_csv(prediction)


def export_csv(folder):
    rows = json.loads((folder / 'predictions.json').read_text())
    fields = ['image_id', 'file_name', 'category_id', 'x', 'y', 'width', 'height', 'score']
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            box = row['bbox']
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             **dict(zip(['x', 'y', 'width', 'height'], box))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'predict', 'train'], default='predict')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/kamp_xray_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260929, 20260930, 20260931])
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    result = audit(dataset, ROOT / 'repro/manifests/split_manifest.csv')
    model = json.loads((ROOT / 'repro/manifests/model.json').read_text())
    frozen = ROOT / 'repro' / model['checkpoint']
    init = ROOT / 'repro/checkpoints/dfine_s_coco_init.pth'
    result['checkpoints'] = {'frozen_present': frozen.is_file(), 'initialization_present': init.is_file()}
    if args.mode == 'check':
        output.mkdir(parents=True, exist_ok=False)
        (output / 'dataset_audit.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
        print('DATA CHECK PASSED. This does not certify training or inference.')
        return
    if not (ROOT / 'repro/vendor/dfine/src/core/__init__.py').is_file():
        raise FileNotFoundError('D-FINE source is missing; initialize Git submodules or use the complete ZIP')
    if args.mode == 'predict':
        checkpoint(frozen, model['sha256'])
    else:
        if not args.device.startswith('cuda'):
            raise ValueError('Full training requires a CUDA GPU')
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Duplicate seeds')
        provenance = json.loads((ROOT / 'repro/manifests/source_provenance.json').read_text())
        init_hash = next(row['sha256'] for row in provenance['training_sources'] if row['source'].endswith('/dfine_s_init.pth'))
        checkpoint(init, init_hash)
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable; fixed-model inference also supports --device cpu')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'dataset_audit.json').write_text(json.dumps(result, indent=2))
    started = time.time()
    if args.mode == 'predict':
        predict_evaluate(dataset, output, args.device)
    else:
        data640 = output / 'data640'
        run([sys.executable, ROOT / 'repro/scripts/prepare_training.py', '--dataset', dataset, '--output', data640])
        results = []
        for seed in args.seeds:
            seed_output = output / f'seed_{seed}'
            config = seed_output / 'train.yml'
            run([sys.executable, ROOT / 'repro/scripts/configure_training.py', '--data640', data640, '--output', config])
            run([sys.executable, ROOT / 'repro/train_mal.py', '-c', config, '-t', init,
                 '--device', args.device, '--seed', seed])
            mal = seed_output / 'mal_run/best_stg1.pth'
            if not mal.is_file():
                raise FileNotFoundError(f'MAL training did not produce {mal}')
            uq = seed_output / 'uq_run'
            run([sys.executable, ROOT / 'repro/train_uq.py', '--config', config, '--base-checkpoint', mal,
                 '--output', uq, '--device', args.device, '--seed', seed])
            predict_evaluate(dataset, seed_output, args.device, uq / 'last.pth', seed)
            for split in ('val', 'test'):
                metrics = json.loads((seed_output / split / 'metrics/metrics.json').read_text())
                results.append({'seed': seed, 'split': split, **metrics})
        (output / 'seed_metrics.json').write_text(json.dumps(results, indent=2))
        summary = {}
        for split in ('val', 'test'):
            summary[split] = {}
            for metric in ('AP', 'AP50', 'AP75', 'recall', 'FP_per_image'):
                values = [r[metric] for r in results if r['split'] == split]
                summary[split][metric] = {'mean': statistics.mean(values),
                                        'sample_std': statistics.stdev(values) if len(values) > 1 else 0}
        (output / 'seed_summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'completed.json').write_text(json.dumps({'mode': args.mode, 'seconds': time.time()-started,
        'device': args.device, 'torch': torch.__version__, 'seeds': args.seeds if args.mode == 'train' else [model['seed']],
        'scope': 'MAL+UQ final-model path; historical baseline and synthesis experiments are separate',
        'threshold': model['val_threshold'], 'threshold_optimized_on_test': False}, indent=2))
    print(f'COMPLETED: {output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'FAILED: {error}', file=sys.stderr)
        sys.exit(1)
