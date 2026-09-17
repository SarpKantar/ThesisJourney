#!/usr/bin/env python
"""Create a two-page image-only PDF from the RN50 filter analysis outputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


PAGE_W = 3508
PAGE_H = 2480
MARGIN = 120
GUTTER = 48
CONTENT_W = PAGE_W - 2 * MARGIN
BG = (255, 255, 255)
INK = (22, 28, 36)
MUTED = (78, 86, 96)
LIGHT = (216, 221, 228)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = get_font(70, bold=True)
FONT_LABEL = get_font(31, bold=True)


def trim_whitespace(image: Image.Image, threshold: int = 8, pad: int = 10) -> Image.Image:
    rgb = image.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, BG)).convert("L")
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(rgb.width, right + pad)
    bottom = min(rgb.height, bottom + pad)
    return rgb.crop((left, top, right, bottom))


def draw_header(page: Image.Image, title: str) -> None:
    draw = ImageDraw.Draw(page)
    draw.text((MARGIN, 56), title, font=FONT_TITLE, fill=INK)
    report_text = f"Report | {date.today().isoformat()}"
    box = draw.textbbox((0, 0), report_text, font=FONT_LABEL)
    draw.text((PAGE_W - MARGIN - (box[2] - box[0]), 82), report_text, font=FONT_LABEL, fill=MUTED)
    draw.line((MARGIN, 180, PAGE_W - MARGIN, 180), fill=LIGHT, width=2)


def paste_image_panel(
    page: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
) -> None:
    x, y, w, h = box
    source = trim_whitespace(Image.open(path))
    fitted = ImageOps.contain(source, (w, h), method=Image.Resampling.LANCZOS)
    px = x + (w - fitted.width) // 2
    py = y + (h - fitted.height) // 2
    page.paste(fitted, (px, py))


def create_page_one(output_dir: Path) -> Image.Image:
    fig_dir = output_dir / "figures"
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw_header(page, "RN50 Filter Analysis")

    top_y = 220
    top_h = 600
    paste_image_panel(
        page,
        fig_dir / "drift_global_heatmap.png",
        (MARGIN, top_y, 920, top_h),
    )
    paste_image_panel(
        page,
        fig_dir / "drift_layer_pairs_by_depth.png",
        (MARGIN + 920 + GUTTER, top_y, CONTENT_W - 920 - GUTTER, top_h),
    )

    stage_y = 860
    stage_h = 520
    stage_w = (CONTENT_W - 3 * GUTTER) // 4
    for idx, stage in enumerate(["layer1", "layer2", "layer3", "layer4"]):
        paste_image_panel(
            page,
            fig_dir / f"drift_stage_{stage}_heatmap.png",
            (MARGIN + idx * (stage_w + GUTTER), stage_y, stage_w, stage_h),
        )

    bottom_y = 1435
    bottom_h = 875
    paste_image_panel(
        page,
        fig_dir / "component_drift_contributions.png",
        (MARGIN, bottom_y, CONTENT_W, bottom_h),
    )
    return page


def create_page_two(output_dir: Path) -> Image.Image:
    fig_dir = output_dir / "figures"
    page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw_header(page, "Filter Geometry and Layer Quality")

    paste_image_panel(
        page,
        fig_dir / "pca_eigenfilters_target_basis.png",
        (MARGIN, 220, CONTENT_W, 470),
    )

    mid_y = 735
    mid_h = 630
    ridge_w = 1845
    paste_image_panel(
        page,
        fig_dir / "ridge_global_three_models.png",
        (MARGIN, mid_y, ridge_w, mid_h),
    )
    paste_image_panel(
        page,
        fig_dir / "scatter_c0_c1_by_model.png",
        (MARGIN + ridge_w + GUTTER, mid_y, CONTENT_W - ridge_w - GUTTER, mid_h),
    )

    low_y = 1430
    low_h = 860
    low_w = (CONTENT_W - 2 * GUTTER) // 3
    for idx, name in enumerate(
        [
            "entropy_by_depth.png",
            "entropy_over_random_threshold_by_depth.png",
            "sparsity_by_depth.png",
        ]
    ):
        paste_image_panel(
            page,
            fig_dir / name,
            (MARGIN + idx * (low_w + GUTTER), low_y, low_w, low_h),
        )
    return page


def save_pdf(pages: Sequence[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = pages
    first.save(path, "PDF", resolution=300.0, save_all=True, append_images=rest)


def main() -> None:
    output_dir = Path("outputs/rn50_filter_analysis")
    pages = [
        create_page_one(output_dir),
        create_page_two(output_dir),
    ]
    clean_path = output_dir / "two_page_filter_report.pdf"
    save_pdf(pages, clean_path)
    print(clean_path)


if __name__ == "__main__":
    main()
