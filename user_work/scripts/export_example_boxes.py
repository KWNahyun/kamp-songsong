"""Render COCO xywh GT/prediction boxes; no model inference or GT-based selection."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.0)
    args = p.parse_args()
    gt = json.loads(args.annotations.read_text())
    predictions = json.loads(args.predictions.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for record in gt['images']:
        source = Image.open(args.images / record['file_name']).convert('RGB')
        source.save(args.output / (Path(record['file_name']).stem + '_plain.png'))
        canvas = source.copy()
        draw = ImageDraw.Draw(canvas)
        for items, color, label in [(gt['annotations'], '#00a050', 'GT'), (predictions, '#e05020', 'Pred')]:
            for box in items:
                if box['image_id'] != record['id'] or (label == 'Pred' and box.get('score', 0) < args.threshold):
                    continue
                x, y, w, h = box['bbox']
                draw.rectangle((x, y, x+w, y+h), outline=color, width=1)
        canvas.save(args.output / (Path(record['file_name']).stem + '_boxes.png'))
    print(f"Rendered {len(gt['images'])} images: green=GT, orange=prediction. Output: {args.output}")

if __name__ == '__main__':
    main()
