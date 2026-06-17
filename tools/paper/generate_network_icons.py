#!/usr/bin/env python3
"""Compile the compact TikZ network icons to transparent SVG files."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "tools" / "paper" / "tikz_network_icons"
OUTPUT_DIR = ROOT / "paper_assets" / "network_icons"
SOURCES = (
    "affordance_map_icon.tex",
    "learnedw_gate_icon.tex",
    "avoid_expert_icon.tex",
    "locomotion_policy_icon.tex",
)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"missing required tool: {name}")
    return path


def compile_icon(pdflatex: str, dvisvgm: str, source: Path, build_dir: Path) -> Path:
    latex_result = subprocess.run(
        [
            pdflatex,
            "-halt-on-error",
            "-interaction=nonstopmode",
            f"-output-directory={build_dir}",
            source.name,
        ],
        cwd=SOURCE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if latex_result.returncode != 0:
        raise SystemExit(f"TikZ compilation failed for {source.name}:\n{latex_result.stdout}")
    pdf_path = build_dir / f"{source.stem}.pdf"
    output_path = OUTPUT_DIR / f"{source.stem}.svg"
    svg_result = subprocess.run(
        [
            dvisvgm,
            "--pdf",
            "--no-fonts",
            "--exact",
            "--bbox=min",
            f"--output={output_path}",
            str(pdf_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if svg_result.returncode != 0:
        raise SystemExit(f"SVG conversion failed for {source.name}:\n{svg_result.stdout}")
    return output_path


def main() -> None:
    pdflatex = require_tool("pdflatex")
    dvisvgm = require_tool("dvisvgm")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tikz_network_icons_") as temp_dir:
        build_dir = Path(temp_dir)
        for filename in SOURCES:
            output = compile_icon(pdflatex, dvisvgm, SOURCE_DIR / filename, build_dir)
            print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
