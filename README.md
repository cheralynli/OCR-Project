# OCR Camera Reader

Live camera and image-file OCR for reading short screen codes with PaddleOCR.

The script uses the lightweight PaddleOCR mobile models:

- `PP-OCRv5_mobile_det`
- `PP-OCRv5_mobile_rec`

It can read from a USB camera, one image, or a folder of test images. It also includes code cleanup for common OCR mistakes in code-shaped text, such as the `L` plus five digits field.

## Setup

Install dependencies with Poetry:

```bash
poetry install
```

The first OCR run may download PaddleOCR model files into:

```text
~/.paddlex/official_models/
```

## Camera Mode

Open camera `0`:

```bash
poetry run python camera_ocr.py
```

Or explicitly:

```bash
poetry run python camera_ocr.py --camera 0
```

Use digital zoom if the screen text is small:

```bash
poetry run python camera_ocr.py --camera 0 --zoom 1.8
```

Camera window controls:

```text
SPACE    capture and OCR
+ / =    zoom in
- / _    zoom out
q / ESC  quit
```

## Image Mode

Run OCR on one image:

```bash
poetry run python camera_ocr.py --image sample2.PNG
```

Run OCR on a folder:

```bash
poetry run python camera_ocr.py --images zoomin
```

Example output:

```text
BA5C24 (Inbound)
L01813 (CM) - WZ5G03
```

## Useful Options

Print only the `L` plus five digit code:

```bash
poetry run python camera_ocr.py --image sample2.PNG --code-only
```

Use a fixed crop region:

```bash
poetry run python camera_ocr.py --camera 0 --roi 420,250,700,220
```

Lower the OCR confidence filter if text is missing:

```bash
poetry run python camera_ocr.py --camera 0 --min-confidence 0.25
```

Raise it if too much junk appears:

```bash
poetry run python camera_ocr.py --camera 0 --min-confidence 0.5
```

Disable aggressive code cleanup:

```bash
poetry run python camera_ocr.py --camera 0 --no-aggressive-code-normalize
```

## PaddleOCR Tuning

Current detector defaults are tuned for camera/screen photos:

```text
--det-limit-type max
--det-limit-side-len 960
--det-thresh 0.3
--det-box-thresh 0.6
--det-unclip-ratio 1.5
```

If PaddleOCR misses text in large camera images, try increasing the side length:

```bash
poetry run python camera_ocr.py --images zoomin --det-limit-side-len 1280
```

If it detects too many false boxes, try raising the box threshold:

```bash
poetry run python camera_ocr.py --camera 0 --det-box-thresh 0.7
```

## Notes

- PaddleOCR is more accurate than the previous EasyOCR setup on `sample2.PNG`.
- Live camera OCR can vary frame to frame because of autofocus, exposure, screen flicker, blur, and glare.
- Better focus, less glare, and a larger on-screen code usually help more than heavy post-processing.
