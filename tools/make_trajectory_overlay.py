#!/usr/bin/env python3
"""Create a trajectory-overlay figure from ordered Isaac Gym screenshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


def _parse_crop(value: str | None):
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be left,top,right,bottom")
    return tuple(parts)


def _fit_crop(images, crop):
    min_w = min(im.width for im in images)
    min_h = min(im.height for im in images)
    if crop is None:
        return (0, 0, min_w, min_h)
    left, top, right, bottom = crop
    return (
        max(0, left),
        max(0, top),
        min(min_w, right),
        min(min_h, bottom),
    )


def _motion_mask(base: Image.Image, image: Image.Image, threshold: int) -> Image.Image:
    diff = ImageChops.difference(base, image).convert("L")
    diff = ImageEnhance.Contrast(diff).enhance(2.5)
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    return mask


def make_overlay(paths, out_path: Path, crop, alpha: float, threshold: int):
    images = [Image.open(p).convert("RGB") for p in paths]
    crop_box = _fit_crop(images, crop)
    images = [im.crop(crop_box) for im in images]

    base = ImageEnhance.Contrast(images[0]).enhance(1.05).convert("RGBA")
    base = ImageEnhance.Brightness(base).enhance(0.93)

    colors = [
        (34, 139, 230),
        (64, 192, 87),
        (250, 176, 5),
        (245, 101, 101),
        (156, 108, 255),
        (255, 146, 43),
    ]

    canvas = base.copy()
    for idx, im in enumerate(images):
        color = colors[idx % len(colors)]
        mask = _motion_mask(images[0], im, threshold)
        tint = ImageOps.colorize(mask.convert("L"), black=(0, 0, 0), white=color).convert("RGBA")
        tint.putalpha(mask.point(lambda p: int(p * alpha)))
        canvas = Image.alpha_composite(canvas, tint)

        # Also retain a faint copy of each changed object, so the hexapod body
        # keeps its natural shape instead of becoming only a colored blob.
        natural = im.convert("RGBA")
        natural.putalpha(mask.point(lambda p: int(p * min(alpha + 0.08, 0.75))))
        canvas = Image.alpha_composite(canvas, natural)

    # Make the already-rendered trajectory lines easier to read after blending.
    rgb = canvas.convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path)
    return crop_box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="ordered screenshots")
    parser.add_argument("--out", required=True, help="output png")
    parser.add_argument("--crop", default=None, help="left,top,right,bottom")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--threshold", type=int, default=28)
    args = parser.parse_args()

    paths = [Path(p) for p in args.images]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing images: " + ", ".join(missing))

    crop_box = make_overlay(
        paths,
        Path(args.out),
        _parse_crop(args.crop),
        max(0.0, min(1.0, args.alpha)),
        max(1, args.threshold),
    )
    print(f"saved {args.out}")
    print(f"crop {crop_box}")


if __name__ == "__main__":
    main()
