# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

import typer

try:
    from PIL import Image
    from PIL import ImageDraw
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

app = typer.Typer(help="Render detection overlays on images.")


def _require_pillow() -> None:
    if Image is None or ImageDraw is None:  # pragma: no cover
        raise RuntimeError("Image rendering requires Pillow.")


def _load_detection_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        if isinstance(payload.get("detections"), list):
            return [d for d in payload["detections"] if isinstance(d, dict)]
        if isinstance(payload.get("page_elements"), list):
            return [d for d in payload["page_elements"] if isinstance(d, dict)]
        return [payload]
    if isinstance(payload, list):
        return [d for d in payload if isinstance(d, dict)]
    return []


def _extract_bbox_xyxy(det: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "bbox_xyxy", "bbox_xyxy_norm"):
        value = det.get(key)
        if not isinstance(value, Sequence) or len(value) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in value]
        except Exception:
            continue
        if key.endswith("_norm"):
            return (x1 * width, y1 * height, x2 * width, y2 * height)
        return (x1, y1, x2, y2)
    return None


def render_page_element_detections_for_image(
    image_path: Path,
    detections_path: Path,
    output_path: Path | None = None,
) -> Path:
    _require_pillow()
    output_path = output_path or image_path.with_name(f"{image_path.stem}.rendered{image_path.suffix}")
    detections = _load_detection_payload(detections_path)

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size

        for det in detections:
            bbox = _extract_bbox_xyxy(det, width, height)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            label = str(det.get("label_name") or det.get("label") or "")
            if label:
                draw.text((x1 + 2, y1 + 2), label, fill="red")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    return output_path


def render_page_element_detections_for_dir(
    input_dir: Path,
    detections_dir: Path,
    output_dir: Path,
) -> list[Path]:
    rendered: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(p for p in input_dir.iterdir() if p.is_file()):
        detections_path = detections_dir / f"{image_path.stem}.json"
        if not detections_path.is_file():
            continue
        rendered.append(
            render_page_element_detections_for_image(
                image_path=image_path,
                detections_path=detections_path,
                output_path=output_dir / image_path.name,
            )
        )

    return rendered


@app.command("image")
def render_image_command(
    image_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, path_type=Path),
    detections_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, path_type=Path),
    output_path: Path | None = typer.Option(None, "--output-path", path_type=Path),
) -> None:
    rendered = render_page_element_detections_for_image(image_path, detections_path, output_path)
    typer.echo(str(rendered))


@app.command("dir")
def render_dir_command(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, path_type=Path),
    detections_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, path_type=Path),
    output_dir: Path = typer.Argument(..., file_okay=False, dir_okay=True, path_type=Path),
) -> None:
    rendered = render_page_element_detections_for_dir(input_dir, detections_dir, output_dir)
    typer.echo(f"Rendered {len(rendered)} images to {output_dir}")
