"""Draw submission Figures 3 and 4 as reproducible SVG, PNG, and PDF."""
from __future__ import annotations

import csv
import html
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#24476B"
BLUE = "#3976A8"
TEAL = "#168A83"
GREEN = "#3A7D58"
ORANGE = "#C7772C"
INK = "#263442"
MUTED = "#657482"
PALE_BLUE = "#F1F6FA"
PALE_TEAL = "#EEF8F6"
PALE_GREEN = "#F1F7F2"
PALE_ORANGE = "#FFF7EE"
FONT = "Noto Sans CJK JP, Noto Sans CJK KR, sans-serif"


def text(x, y, value, size=24, fill=INK, weight="normal", anchor="start", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{html.escape(str(value))}</text>"
    )


def multi_text(x, y, lines, size=22, fill=INK, weight="normal", gap=34, anchor="middle"):
    tspans = "".join(
        f'<tspan x="{x}" dy="{0 if index == 0 else gap}">{html.escape(str(line))}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{tspans}</text>'
    )


def box(x, y, w, h, title, details, stroke, fill, title_size=24, detail_size=18):
    title_y = y + 52
    details_y = y + 102
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
        + text(x + w / 2, title_y, title, title_size, INK, "700", "middle")
        + multi_text(x + w / 2, details_y, details, detail_size, INK, gap=31)
    )


def arrow(x1, y1, x2, y2, color=BLUE, width=3, dashed=False):
    dash = ' stroke-dasharray="9 8"' if dashed else ""
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{dash} marker-end="url(#arrow-{color[1:]})"/>'
    )


def header(width, height, title, subtitle):
    marker_defs = "".join(
        f'<marker id="arrow-{color[1:]}" markerWidth="10" markerHeight="10" '
        f'refX="8" refY="5" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
        for color in (BLUE, MUTED, ORANGE)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><defs>{marker_defs}</defs>'
        f'<rect width="100%" height="100%" fill="white"/>'
        + text(60, 67, title, 36, INK, "700")
        + text(60, 116, subtitle, 21, MUTED)
    )


def export_svg(stem: str, source: str, width: int, height: int):
    svg_path = OUT / f"{stem}.svg"
    svg_path.write_text(source, encoding="utf-8")
    chrome = shutil.which("google-chrome")
    if chrome is None:
        raise RuntimeError("google-chrome is required to render Korean SVG text accurately")
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={OUT / f'{stem}.png'}",
            svg_path.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tempfile.TemporaryDirectory(prefix="kamp-report-figure-") as temp_dir:
        temp = Path(temp_dir)
        page_width = width / 96
        page_height = height / 96
        html_path = temp / "print.html"
        html_path.write_text(
            "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"@page{{size:{page_width}in {page_height}in;margin:0}}"
            f"html,body{{width:{width}px;height:{height}px;margin:0;padding:0}}"
            f"img{{display:block;width:{width}px;height:{height}px}}"
            f"</style></head><body><img src=\"{svg_path.as_uri()}\"></body></html>",
            encoding="utf-8",
        )
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-pdf-header-footer",
                f"--user-data-dir={temp / 'chrome-profile'}",
                f"--print-to-pdf={OUT / f'{stem}.pdf'}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def make_architecture():
    width, height = 1800, 940
    out = [
        header(
            width,
            height,
            "최종 검출 구조: MAL로 학습한 D-FINE-S와 후보 품질 재점수화(UQ)",
            "추론 흐름 — UQ는 박스 좌표를 다시 예측하지 않고, 후보의 위치 품질을 반영해 검출 점수를 조정한다.",
        ),
        text(60, 183, "추론", 20, BLUE, "700"),
    ]

    y, h = 235, 235
    nodes = [
        (50, 145, "X-ray 영상", ["640 × 640 입력"], BLUE, PALE_BLUE),
        (245, 270, "D-FINE-S", ["MAL 적용 학습 후 고정", "query decoder · FDR"], NAVY, PALE_BLUE),
        (575, 250, "검출 후보", ["박스 bᵢ · 기본 점수 pᵢ", "query 특징 · 경계 분포"], BLUE, PALE_BLUE),
        (890, 215, "UQ 예측기", ["283차원 입력", "62,007개 파라미터"], TEAL, PALE_TEAL),
        (1170, 210, "점수 결합", ["sᵢ = √(pᵢqᵢ)", "박스 bᵢ는 유지"], GREEN, PALE_GREEN),
        (1450, 260, "NMS · 임계값", ["NMS IoU 0.7", "고정 점수 기준", "최종 경보 박스"], GREEN, PALE_GREEN),
    ]
    for x, w, title_value, details, stroke, fill in nodes:
        out.append(box(x, y, w, h, title_value, details, stroke, fill,
                       title_size=24 if w > 180 else 22, detail_size=18))
    for (x1, w1, *_), (x2, *_rest) in zip(nodes[:-1], nodes[1:]):
        out.append(arrow(x1 + w1 + 5, y + h / 2, x2 - 9, y + h / 2))

    # Base score and box geometry join the UQ estimate only at score fusion.
    out.extend(
        [
            f'<path d="M 825 235 L 825 185 L 1275 185 L 1275 226" fill="none" '
            f'stroke="{MUTED}" stroke-width="2.5" marker-end="url(#arrow-{MUTED[1:]})"/>',
            text(1050, 173, "기본 점수 pᵢ와 박스 bᵢ도 점수 결합부로 전달", 18, MUTED, "normal", "middle"),
            text(60, 552, "UQ 학습 때만 사용하는 감독 신호", 20, ORANGE, "700"),
        ]
    )

    train_nodes = [
        (180, 620, 270, "공식 TXT 정답", ["학습 중 품질 목표 생성"], ORANGE, PALE_ORANGE),
        (620, 620, 320, "품질 목표 rᵢ", ["후보와 정답 사이의 최대 IoU"], ORANGE, PALE_ORANGE),
        (1120, 620, 410, "UQ 예측기만 학습", ["D-FINE-S와 박스 좌표는 고정"], ORANGE, PALE_ORANGE),
    ]
    for x, ty, w, title_value, details, stroke, fill in train_nodes:
        out.append(box(x, ty, w, 155, title_value, details, stroke, fill, title_size=22, detail_size=18))
    out.extend(
        [
            arrow(450, 698, 610, 698, ORANGE, dashed=True),
            arrow(830, 470, 780, 610, ORANGE, dashed=True),
            arrow(950, 698, 1110, 698, ORANGE, dashed=True),
            arrow(1320, 610, 995, 480, ORANGE, dashed=True),
            text(745, 580, "고정된 후보 박스", 16, ORANGE, "normal", "middle"),
            text(1315, 580, "학습 후 추론에서는 정답 미사용", 16, ORANGE, "normal", "middle"),
            text(
                60,
                850,
                "FDR: D-FINE의 경계 분포 정제  ·  pᵢ: 기존 검출 점수  ·  qᵢ: UQ가 예측한 위치 품질  ·  NMS: 겹치는 중복 박스 억제",
                17,
                MUTED,
            ),
            "</svg>",
        ]
    )
    export_svg("figure_07_method", "".join(out), width, height)


def make_ambiguity_comparison():
    source = PROJECT / "experiments/kamp_v2_ambiguity_uq/analysis/paired_UQ.csv"
    with source.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["control"] == "edge"]
    rows.sort(key=lambda row: int(row["seed"]))
    if len(rows) != 3:
        raise ValueError(f"Expected 3 paired edge-control rows, got {len(rows)}")

    width, height = 1800, 820
    out = [
        header(
            width,
            height,
            "경계 1픽셀 완화의 반복별 변화",
            "같은 난수 초기값끼리 비교: Soft1+UQ − 경계 좌표 대조군(edge+UQ). 양수는 Soft1+UQ가 높은 점수임을 뜻한다.",
        )
    ]
    charts = [
        {"metric": "AP", "title": "전체 검출 성능(AP)", "x": 95, "y": 240, "w": 730, "h": 430, "min": -1.25, "max": 1.6},
        {"metric": "AP75", "title": "엄격한 위치 정밀도(AP75)", "x": 975, "y": 240, "w": 730, "h": 430, "min": -1.0, "max": 3.45},
    ]
    for chart in charts:
        x, y, w, h = chart["x"], chart["y"], chart["w"], chart["h"]
        low, high = chart["min"], chart["max"]
        plot_top, plot_bottom = y + 62, y + h - 80
        plot_left, plot_right = x + 65, x + w - 22

        def yy(value):
            return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

        out.append(text(x + w / 2, y + 25, chart["title"], 22, INK, "700", "middle"))
        for tick in (-1, 0, 1, 2, 3):
            if tick < low or tick > high:
                continue
            py = yy(tick)
            out.append(f'<path d="M {plot_left} {py:.1f} L {plot_right} {py:.1f}" stroke="#DCE3E8" stroke-width="1.4"/>')
            out.append(text(plot_left - 14, py + 6, f"{tick:+d}", 15, MUTED, "normal", "end"))

        # Zero reference and mean reference.
        out.append(f'<path d="M {plot_left} {yy(0):.1f} L {plot_right} {yy(0):.1f}" stroke="{INK}" stroke-width="2"/>')
        values = [float(row[chart["metric"]]) for row in rows]
        mean = sum(values) / len(values)
        out.append(
            f'<path d="M {plot_left} {yy(mean):.1f} L {plot_right} {yy(mean):.1f}" '
            f'stroke="{BLUE}" stroke-width="2.4" stroke-dasharray="10 8"/>'
        )
        out.append(text(x + w / 2, y + 55, f"파란 점선: 3회 평균 {mean:+.2f}", 15, BLUE, "700", "middle"))

        centers = [plot_left + (i + 0.5) * (plot_right - plot_left) / 3 for i in range(3)]
        base = yy(0)
        for center, value, row in zip(centers, values, rows):
            top = yy(max(value, 0))
            bottom = yy(min(value, 0))
            color = GREEN if value > 0 else ORANGE
            bar_y = min(top, bottom)
            bar_h = max(abs(bottom - top), 2)
            out.append(
                f'<rect x="{center - 46:.1f}" y="{bar_y:.1f}" width="92" height="{bar_h:.1f}" '
                f'fill="{color}" rx="5"/>'
            )
            label_y = top - 13 if value >= 0 else bottom + 26
            out.append(text(center, label_y, f"{value:+.2f}", 19, INK, "700", "middle"))
            out.append(text(center, plot_bottom + 38, f"seed {row['seed']}", 15, MUTED, "normal", "middle"))

    out.extend(["</svg>"])
    export_svg("figure_08_ambiguity_control", "".join(out), width, height)


if __name__ == "__main__":
    make_architecture()
    make_ambiguity_comparison()
    print("Updated figure_07_method and figure_08_ambiguity_control (SVG + PNG + PDF)")
