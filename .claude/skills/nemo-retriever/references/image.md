# retriever image

Visualization helpers: render YOLOX page-element / chart-element detection
overlays on page images so you can sanity-check the detector by eye.

If flags below look stale, re-check `retriever image render --help`.

## When to use this

- A page-element or chart detector returned suspect boxes and you want to
  see them overlaid on the source page image.
- You're tuning thresholds and need a quick visual diff.

**Use a different command when:**

- You need the actual extraction output, not a picture → [[pdf]] or
  [[chart]].
- You want benchmarks over the detector → [[benchmark]] (`page-elements`).

## Canonical invocations

Overlay a single page:

```bash
retriever image render image \
  page_001.png \
  page_001.detections.json \
  --output-path page_001.overlay.png
```

Overlay every page in a directory:

```bash
retriever image render dir \
  pages/ detections/ overlays/
```

## Inputs

- **`render image`**: a PNG/JPEG `image_path` plus a `detections_path` JSON
  (YOLOX-shaped output).
- **`render dir`**: parallel `input_dir` / `detections_dir`, output written
  per-image to `output_dir`. Files are matched by basename.

## Outputs

- A single composite image with bounding boxes + class labels drawn on top
  of the source. **Not** a side-by-side / split layout; if you want
  original-vs-overlay panels, compose them yourself (e.g. via `ffmpeg
  hstack` or `PIL`). `render image` writes to `--output-path`; `render dir`
  writes into `output_dir`.

## Common failure modes

- **No boxes appear** — the detections JSON shape doesn't match what
  `render` expects. Use the JSON that `retriever pdf stage page-elements`
  (or [[pipeline]]) emitted, not a hand-rolled file.
- **Mismatched coordinates** — detections were produced against a different
  page render scale than the image you're overlaying on. Re-render at the
  same DPI/`render-mode` you ran the detector with.

## Related

- [[pdf]] — produce the detections JSON that this command renders.
