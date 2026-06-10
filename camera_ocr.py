#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


CODE_RE = re.compile(r"\bL\s*(\d{5})\b")
MISSING_L_CODE_RE = re.compile(r"(?<![A-Z0-9])[:|I1]?\s*(\d{5})(?=\s*\(\s*CM\s*\))", re.IGNORECASE)
CM_CODE_RE = re.compile(r"(?:\bL\s*)?(\d{5})\s*\(\s*CM\s*\)", re.IGNORECASE)
VAR1_RE = re.compile(r"\b([A-Z0-9]{5,8})\b\s*\(\s*inbound\s*\)", re.IGNORECASE)
FALLBACK_VAR1_RE = re.compile(
    r"\b(?!L\d{5}\b|W[Z2][A-Z0-9]{4,}\b)(?=[A-Z0-9]*\d)([A-Z0-9]{5,8})\b",
    re.IGNORECASE,
)
VAR3_RE = re.compile(r"\b(W[Z2][A-Z0-9]{4,})\b", re.IGNORECASE)
DASH_CODE_RE = re.compile(r"[-–—]\s*([A-Z0-9]{3,8})\b", re.IGNORECASE)
OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 11",
    "--oem 3 --psm 12",
    "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()-:/ ",
]


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def check_dependencies() -> None:
    if cv2 is None:
        fail("Missing Python package: opencv-python. Install it with: poetry add opencv-python")
    if pytesseract is None:
        fail("Missing Python package: pytesseract. Install it with: poetry add pytesseract")

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        fail("Missing Tesseract OCR engine. Install it with: brew install tesseract")


def preprocess_variants(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted = cv2.bitwise_not(thresh)
    return [frame, gray, scaled, big, thresh, adaptive, inverted]


def crop_frame(frame, roi: tuple[int, int, int, int] | None):
    if not roi:
        return frame

    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def detect_screen_crop(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    frame_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < frame_area * 0.08:
            continue
        if w < width * 0.20 or h < height * 0.15:
            continue

        aspect = w / h
        if not 0.6 <= aspect <= 3.5:
            continue

        if area > best_area:
            margin_x = int(w * 0.03)
            margin_y = int(h * 0.03)
            x1 = max(0, x + margin_x)
            y1 = max(0, y + margin_y)
            x2 = min(width, x + w - margin_x)
            y2 = min(height, y + h - margin_y)
            best = frame[y1:y2, x1:x2]
            best_area = area

    return best


def auto_ocr_frames(frame, roi: tuple[int, int, int, int] | None):
    if roi:
        return [("roi", crop_frame(frame, roi))]

    height, width = frame.shape[:2]
    screen = detect_screen_crop(frame)
    crops = [
        ("screen", screen),
        ("center", frame[int(height * 0.20) : int(height * 0.80), int(width * 0.15) : int(width * 0.85)]),
        ("upper", frame[int(height * 0.05) : int(height * 0.55), int(width * 0.10) : int(width * 0.90)]),
        ("lower", frame[int(height * 0.45) : int(height * 0.95), int(width * 0.10) : int(width * 0.90)]),
        ("full", frame),
    ]
    return [(name, crop) for name, crop in crops if crop is not None and crop.size]


def ocr_best_frame(frame, roi: tuple[int, int, int, int] | None):
    best = None
    for name, target in auto_ocr_frames(frame, roi):
        code, text = extract_text_and_code(target)
        fields = extract_fields(text)
        candidate = (row_status(fields), code, text, fields, name)
        if row_status(fields) == "parsed":
            return candidate
        if best is None:
            best = candidate
        elif row_status(fields) == "partial" and best[0] == "unparsed":
            best = candidate
        elif code and not best[1]:
            best = candidate

    return best


def candidate_rank(text: str, fields: dict[str, str], confidence: float = 0.0) -> float:
    status_bonus = {"parsed": 1000, "partial": 400, "unparsed": 0}[row_status(fields)]
    field_bonus = sum(80 for key in ("var1", "var2", "var3") if fields[key])
    return status_bonus + field_bonus + text_score(text) + confidence


def save_debug_target(frame, roi: tuple[int, int, int, int] | None, path: Path) -> None:
    frames = auto_ocr_frames(frame, roi)
    if not frames:
        return
    _, target = frames[0]
    cv2.imwrite(str(path), target)


def find_code(text: str) -> str | None:
    match = CODE_RE.search(text)
    if match:
        return f"L{match.group(1)}"

    match = MISSING_L_CODE_RE.search(text)
    if match:
        return f"L{match.group(1)}"

    match = CM_CODE_RE.search(text)
    if match:
        return f"L{match.group(1)}"

    return None


def extract_fields(text: str) -> dict[str, str]:
    normalized = normalize_text(text).upper()
    var1_match = VAR1_RE.search(normalized)
    if not var1_match:
        var1_match = FALLBACK_VAR1_RE.search(normalized)
    var3_match = VAR3_RE.search(normalized)
    if not var3_match:
        var3_match = DASH_CODE_RE.search(normalized)

    return {
        "var1": var1_match.group(1) if var1_match else "",
        "var2": find_code(normalized) or "",
        "var3": var3_match.group(1).replace("W2", "WZ", 1) if var3_match else "",
        "has_inbound": "INBOUND" in normalized,
        "has_cm": bool(re.search(r"\(\s*CM\s*\)", normalized)),
    }


def row_status(fields: dict[str, str]) -> str:
    found_count = sum(1 for key in ("var1", "var2", "var3") if fields[key])
    if found_count == 3:
        return "parsed"
    if found_count:
        return "partial"
    return "unparsed"


def normalize_text(text: str) -> str:
    return MISSING_L_CODE_RE.sub(lambda match: f"L{match.group(1)}", text)


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def text_score(text: str) -> float:
    compact = "".join(char for char in text if not char.isspace())
    if len(compact) < 3:
        return 0

    useful = sum(char.isalnum() or char in "()-:/" for char in compact)
    useful_ratio = useful / len(compact)
    line_bonus = min(len(text.splitlines()), 4) * 2
    length_penalty = max(0, len(compact) - 160) * 0.25
    line_penalty = max(0, len(text.splitlines()) - 8) * 8
    return useful_ratio * min(len(compact), 160) + line_bonus - length_penalty - line_penalty


def ocr_text_with_confidence(image, config: str) -> tuple[str, float, float, int]:
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    words = []
    confidences = []

    for word, confidence in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue
        try:
            confidence_value = float(confidence)
        except ValueError:
            continue
        if confidence_value < 0:
            continue

        words.append(word)
        confidences.append(confidence_value)

    text = clean_text(" ".join(words))
    if not confidences:
        return text, 0.0, 0.0, 0
    return text, sum(confidences) / len(confidences), max(confidences), len(confidences)


def is_readable_text(text: str, average_confidence: float, max_confidence: float, word_count: int) -> bool:
    compact = "".join(char for char in text if not char.isspace())
    if len(compact) < 4 or word_count == 0:
        return False

    alnum_count = sum(char.isalnum() for char in compact)
    if alnum_count / len(compact) < 0.75:
        return False

    return average_confidence >= 28 or (word_count <= 8 and max_confidence >= 55)


def compact_lines(text: str) -> list[str]:
    cleaned = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def limit_output(text: str, max_lines: int = 6, max_chars: int = 320) -> str:
    lines = compact_lines(text)
    if len(lines) <= max_lines:
        return "\n".join(lines)[:max_chars]
    return "\n".join(lines[:max_lines])[:max_chars]


def looks_like_noise(text: str) -> bool:
    lines = compact_lines(text)
    compact = "".join(lines)
    if len(compact) < 4:
        return True
    if len(lines) > 12 or len(compact) > 700:
        return True

    alnum_count = sum(char.isalnum() for char in compact)
    punctuation_count = sum(not char.isalnum() for char in compact)
    if alnum_count == 0:
        return True
    if punctuation_count / max(len(compact), 1) > 0.45:
        return True

    words = re.findall(r"[A-Za-z0-9]{2,}", text)
    return len(words) == 0


def extract_text_and_code(frame) -> tuple[str | None, str]:
    best_text = ""
    best_fields = {"var1": "", "var2": "", "var3": ""}
    best_rank = 0.0
    best_is_readable = False
    seen = set()

    for variant in preprocess_variants(frame):
        for config in OCR_CONFIGS:
            raw_text = clean_text(pytesseract.image_to_string(variant, config=config))
            if raw_text and raw_text not in seen:
                seen.add(raw_text)
                normalized_raw = normalize_text(raw_text)
                raw_fields = extract_fields(normalized_raw)
                if row_status(raw_fields) == "parsed":
                    return raw_fields["var2"], normalized_raw

                if not looks_like_noise(raw_text):
                    raw_rank = candidate_rank(normalized_raw, raw_fields)
                    if raw_rank > best_rank:
                        best_text = normalized_raw
                        best_fields = raw_fields
                        best_rank = raw_rank
                        best_is_readable = True

            text, average_confidence, max_confidence, word_count = ocr_text_with_confidence(variant, config)
            if not text or text in seen:
                continue

            seen.add(text)
            normalized = normalize_text(text)
            fields = extract_fields(normalized)
            if row_status(fields) == "parsed":
                return fields["var2"], normalized

            if is_readable_text(text, average_confidence, max_confidence, word_count):
                rank = candidate_rank(normalized, fields, average_confidence)
                if rank > best_rank:
                    best_text = normalized
                    best_fields = fields
                    best_rank = rank
                    best_is_readable = True

    if best_is_readable:
        if row_status(best_fields) == "parsed":
            return best_fields["var2"], best_text
        if row_status(best_fields) == "partial":
            return best_fields["var2"] or None, limit_output(best_text)
        return None, limit_output(best_text)

    return None, ""


def extract_code(frame) -> tuple[str | None, str]:
    code, text = extract_text_and_code(frame)
    if code:
        return code, text
    return None, text


def format_ocr_result(text: str, fields: dict[str, str]) -> str:
    if row_status(fields) == "parsed":
        first_line = fields["var1"]
        if fields.get("has_inbound"):
            first_line += " (inbound)"

        second_line = fields["var2"]
        if fields.get("has_cm"):
            second_line += " (CM)"
        second_line += f" - {fields['var3']}"

        return f"{first_line}\n{second_line}"

    lines = []
    if fields["var1"]:
        line = fields["var1"]
        if fields.get("has_inbound"):
            line += " (inbound)"
        lines.append(line)

    second_line_parts = []
    if fields["var2"]:
        line = fields["var2"]
        if fields.get("has_cm"):
            line += " (CM)"
        second_line_parts.append(line)
    if fields["var3"]:
        second_line_parts.append(fields["var3"])
    if second_line_parts:
        lines.append(" - ".join(second_line_parts))

    return "\n".join(lines) if lines else "No text detected"


def draw_overlay(frame, status: str, roi: tuple[int, int, int, int] | None) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "SPACE: capture OCR   Q/ESC: quit",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 120),
        2,
        cv2.LINE_AA,
    )
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read text from a live camera with Tesseract OCR.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Try 1 or 2 for USB cameras.")
    parser.add_argument("--image", type=Path, help="OCR one image instead of opening the camera.")
    parser.add_argument("--list-cameras", action="store_true", help="Scan camera indexes and report which ones open.")
    parser.add_argument("--snapshot-cameras", action="store_true", help="Save one frame from each camera index.")
    parser.add_argument("--scan-max", type=int, default=2, help="Highest camera index to scan with --list-cameras.")
    parser.add_argument("--code-only", action="store_true", help="Print only the extracted L+5 digit code.")
    parser.add_argument("--roi", help="Optional fixed crop rectangle as x,y,w,h.")
    parser.add_argument("--debug-image", type=Path, help="Save the first OCR target image to this path on SPACE.")
    return parser.parse_args()


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    try:
        x, y, w, h = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        fail("Invalid --roi. Use x,y,w,h, for example: --roi 420,250,700,220")
    if w <= 0 or h <= 0:
        fail("Invalid --roi. Width and height must be positive.")
    return x, y, w, h


def list_cameras(max_index: int = 10) -> None:
    check_dependencies()
    found = False
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        if opened:
            found = True
            ret, frame = cap.read()
            if ret:
                print(f"camera {index}: available ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"camera {index}: opens but cannot read frames")
        cap.release()

    if not found:
        print("No cameras found by OpenCV.")


def snapshot_cameras(max_index: int = 2) -> None:
    check_dependencies()
    found = False
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"camera {index}: opens but cannot read frames")
            continue

        found = True
        path = Path(f"camera_{index}_snapshot.jpg")
        cv2.imwrite(str(path), frame)
        print(f"camera {index}: saved {path}")

    if not found:
        print("No camera snapshots saved.")


def main() -> None:
    args = parse_args()
    roi = parse_roi(args.roi)

    if args.list_cameras:
        list_cameras(args.scan_max)
        return

    if args.snapshot_cameras:
        snapshot_cameras(args.scan_max)
        return

    check_dependencies()

    if args.image:
        if not args.image.exists():
            fail(f"Image file not found: {args.image}")
        frame = cv2.imread(str(args.image))
        if frame is None:
            fail(f"Could not read image file: {args.image}")
        result = ocr_best_frame(frame, roi)
        if result is None:
            fail("No OCR result.")
        _, code, text, fields, _ = result
        if args.code_only and not code:
            fail("No L + 5 digit code detected.")
        print(code if args.code_only else format_ocr_result(text, fields))
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        fail(f"Cannot open camera index {args.camera}. Try --camera 1 or --camera 2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window_name = "Live camera OCR"
    status = "Ready"

    print("Live camera OCR ready")
    print("Press SPACE to capture and OCR")
    print("Press ESC or q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                fail("Failed to read a frame from the camera.")

            display = frame.copy()
            draw_overlay(display, status, roi)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                if args.debug_image:
                    save_debug_target(frame, roi, args.debug_image)
                result = ocr_best_frame(frame, roi)
                if result is None:
                    status = "No OCR result"
                    print("No OCR result")
                    continue

                _, code, text, fields, _ = result
                output = code if args.code_only else format_ocr_result(text, fields)
                if output == "No text detected":
                    status = "No text detected"
                elif row_status(fields) == "parsed":
                    status = f"Found {code}"
                else:
                    status = "Text detected"
                print(output)
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
