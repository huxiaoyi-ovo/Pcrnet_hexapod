#!/usr/bin/env python3
"""Build a Fig.1 teaser from current PCR simulation screenshots."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(im, size, method=RESAMPLE_LANCZOS, centering=(0.5, 0.5))


def round_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
        return
    draw.rectangle(box, fill=fill, outline=outline)


def paste_panel(canvas: Image.Image, box, title: str, image: Image.Image | None = None, subtitle: str | None = None):
    d = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    round_rect(d, box, 28, (250, 250, 248), (40, 43, 48), 3)
    d.text((x0 + 26, y0 + 18), title, fill=(20, 24, 30), font=font(44, True))
    inner = (x0 + 26, y0 + 84, x1 - 26, y1 - 72)
    if image is not None:
        img = fit_cover(image, (inner[2] - inner[0], inner[3] - inner[1]))
        canvas.paste(img, inner[:2])
        ImageDraw.Draw(canvas).rectangle(inner, outline=(40, 43, 48), width=2)
    if subtitle:
        d.text((x0 + 26, y1 - 54), subtitle, fill=(55, 60, 70), font=font(28, False))


def arrow(draw: ImageDraw.ImageDraw, p0, p1, color, width=10):
    draw.line([p0, p1], fill=color, width=width)
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
    ux, uy = vx / norm, vy / norm
    px, py = -uy, ux
    head = 30
    base = (x1 - ux * head, y1 - uy * head)
    pts = [
        (x1, y1),
        (base[0] + px * head * 0.55, base[1] + py * head * 0.55),
        (base[0] - px * head * 0.55, base[1] - py * head * 0.55),
    ]
    draw.polygon(pts, fill=color)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene_path = OUT_DIR / "simple_aac_recording0_full_overlap_moving_regions.png"
    viewer_path = Path("/home/hxy/图片/snipaste/Snipaste_2026-06-18_15-15-14.png")
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)
    if not viewer_path.exists():
        raise FileNotFoundError(viewer_path)

    scene = Image.open(scene_path).convert("RGB")
    viewer = Image.open(viewer_path).convert("RGB")

    W, H = 3600, 1650
    canvas = Image.new("RGB", (W, H), (245, 246, 248))
    d = ImageDraw.Draw(canvas)

    margin = 70
    header_h = 150
    left_box = (margin, header_h, 2260, H - 80)
    right_x = 2320
    right_w = W - right_x - margin

    d.text((margin, 46), "PCR-Net: Command-Space Conflict Arbitration for Legged Following", fill=(15, 18, 24), font=font(58, True))
    d.text((margin, 112), "Follow the target while negotiating local obstacle rows with learned conflict priors.", fill=(65, 70, 80), font=font(32, False))

    # Left: main task scene.
    round_rect(d, left_box, 34, (255, 255, 255), (28, 32, 38), 3)
    sx0, sy0, sx1, sy1 = left_box
    scene_crop = fit_cover(scene, (sx1 - sx0 - 50, sy1 - sy0 - 120))
    canvas.paste(scene_crop, (sx0 + 25, sy0 + 75))
    d.rectangle((sx0 + 25, sy0 + 75, sx1 - 25, sy1 - 45), outline=(28, 32, 38), width=2)
    d.text((sx0 + 35, sy0 + 22), "(a) Target following through obstacle rows", fill=(20, 24, 30), font=font(42, True))
    d.text((sx0 + 45, sy1 - 40), "Rendered trajectory overlay: body poses, field-of-view rays, and executed path.", fill=(55, 60, 70), font=font(26, False))

    # Right top: perception examples from viewer.
    depth = viewer.crop((1440, 1040, 1880, 1510))
    occ = viewer.crop((1960, 1040, 2365, 1510))
    clear = viewer.crop((2440, 1040, 2845, 1510))

    panel_h = 420
    paste_panel(canvas, (right_x, header_h, right_x + right_w, header_h + panel_h), "(b) Real-time policy inputs", None)
    px0, py0, px1, py1 = (right_x, header_h, right_x + right_w, header_h + panel_h)
    inner_y = py0 + 95
    tile_w = (right_w - 86) // 3
    for i, (name, img) in enumerate([("Depth", depth), ("Occupancy", occ), ("Clearance", clear)]):
        tx = px0 + 26 + i * (tile_w + 17)
        tile = fit_cover(img, (tile_w, 245))
        canvas.paste(tile, (tx, inner_y))
        d.rectangle((tx, inner_y, tx + tile_w, inner_y + 245), outline=(40, 43, 48), width=2)
        d.text((tx + 8, inner_y + 258), name, fill=(35, 40, 50), font=font(28, True))

    # Right middle: method blocks.
    method_box = (right_x, header_h + panel_h + 38, right_x + right_w, header_h + panel_h + 520)
    round_rect(d, method_box, 28, (255, 255, 255), (28, 32, 38), 3)
    mx0, my0, mx1, my1 = method_box
    d.text((mx0 + 26, my0 + 18), "(c) High-level arbitration", fill=(20, 24, 30), font=font(42, True))
    blocks = {
        "follow": ("Follow expert", "u_F", (41, 105, 176), (mx0 + 55, my0 + 126)),
        "avoid": ("Avoid expert", "u_A", (42, 145, 110), (mx0 + 55, my0 + 282)),
        "risk": ("Risk query", "risk_F, risk_A", (105, 105, 110), (mx0 + 385, my0 + 126)),
        "gate": ("GatePolicy", "y, w", (36, 72, 105), (mx0 + 385, my0 + 282)),
        "fusion": ("Fusion", "u_mix", (28, 95, 120), (mx0 + 735, my0 + 204)),
    }
    bw = 260
    bh = 100
    coords = {key: (x, y, x + bw, y + bh) for key, (_, _, _, (x, y)) in blocks.items()}
    for a, b in [("follow", "risk"), ("avoid", "risk"), ("risk", "gate"), ("follow", "fusion"), ("avoid", "fusion"), ("gate", "fusion")]:
        x0, y0, x1, y1 = coords[a]
        u0, v0, u1, v1 = coords[b]
        arrow(d, (x1 + 3, (y0 + y1)//2), (u0 - 8, (v0 + v1)//2), (45, 48, 55), 5)
    for key, (title, out, color, (x, y)) in blocks.items():
        round_rect(d, (x, y, x + bw, y + bh), 18, tuple(min(c + 32, 255) for c in color), color, 3)
        d.text((x + 18, y + 20), title, fill=(255, 255, 255), font=font(30, True))
        d.text((x + 18, y + 62), out, fill=(235, 245, 255), font=font(28, False))
    d.text((mx0 + 57, my1 - 46), "solid = command / decision flow; risk query informs gate and fusion", fill=(55, 60, 70), font=font(25, False))

    # Right bottom: execution.
    exec_box = (right_x, my1 + 38, right_x + right_w, H - 80)
    round_rect(d, exec_box, 28, (255, 255, 255), (28, 32, 38), 3)
    ex0, ey0, ex1, ey1 = exec_box
    d.text((ex0 + 26, ey0 + 18), "(d) Fixed locomotion execution", fill=(20, 24, 30), font=font(42, True))
    round_rect(d, (ex0 + 70, ey0 + 120, ex0 + 430, ey0 + 260), 20, (232, 238, 248), (50, 55, 65), 3)
    d.text((ex0 + 108, ey0 + 154), "u_exec", fill=(30, 40, 60), font=font(44, True))
    arrow(d, (ex0 + 450, ey0 + 190), (ex0 + 600, ey0 + 190), (50, 55, 65), 8)
    round_rect(d, (ex0 + 620, ey0 + 100, ex1 - 70, ey0 + 285), 22, (246, 238, 214), (50, 55, 65), 3)
    d.text((ex0 + 660, ey0 + 132), "Frozen low-level policy", fill=(30, 28, 22), font=font(34, True))
    d.text((ex0 + 660, ey0 + 190), "joint targets -> hexapod", fill=(65, 60, 48), font=font(30, False))
    d.text((ex0 + 70, ey1 - 62), "PCR changes only the high-level body command; locomotion remains fixed.", fill=(55, 60, 70), font=font(28, False))

    # Save outputs.
    png_path = OUT_DIR / "fig1_teaser.png"
    pdf_path = OUT_DIR / "fig1_teaser.pdf"
    svg_path = OUT_DIR / "fig1_teaser.svg"
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)

    # Pillow does not write true vector SVG; create an SVG wrapper embedding the PNG.
    import base64
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\\n'
        f'  <image href="data:image/png;base64,{data}" width="{W}" height="{H}"/>\\n'
        f'</svg>\\n',
        encoding="utf-8",
    )
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    main()
