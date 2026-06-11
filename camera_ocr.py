#!/usr/bin/env python3
import argparse
import contextlib
import io
import logging
import re
import sys
import warnings
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true.*")
warnings.filterwarnings("ignore", message="No ccache found.*")
for logger_name in ("paddle", "paddleocr", "paddlex", "ppocr"):
    logging.getLogger(logger_name).disabled = True
    logging.getLogger(logger_name).setLevel(logging.ERROR)


CODE_RE = re.compile(r"\bL\s*(\d{5})\b")
MISSING_L_CODE_RE = re.compile(r"(?<![A-Z0-9])[:|I1]?\s*(\d{5})(?=\s*\(\s*CM\s*\))", re.IGNORECASE)
CM_CODE_RE = re.compile(r"(?:\bL\s*)?(\d{5})\s*\(\s*CM\s*\)", re.IGNORECASE)
VAR1_RE = re.compile(r"\b([A-Z0-9]{5,8})\b\s*\(\s*inbound\s*\)", re.IGNORECASE)
FALLBACK_VAR1_RE = re.compile(
    r"\b(?!L\d{5}\b|W[Z2][A-Z0-9]{4,}\b)(?=[A-Z0-9]*\d)([A-Z0-9]{5,8})\b",
    re.IGNORECASE,
)
VAR3_RE = re.compile(r"\b(W[Z2][A-Z0-9]{4,})\b", re.IGNORECASE)
DASH_CODE_RE = re.compile(r"[-–—_:]\s*([A-Z0-9]{3,8})\b", re.IGNORECASE)
AFTER_L_CODE_RE = re.compile(
    r"\bL\s*\d{5}\b\s*(?:\(\s*CM\s*\))?\s*[-–—_:]?\s*([A-Z0-9]{3,8})\b",
    re.IGNORECASE,
)
LIKELY_L_CODE_RE = re.compile(r"\b[Ll]([0-9OQDIIlTtSioqds]{5})\b")
SPACED_L_CODE_RE = re.compile(r"\b[Ll]\s*([0-9OQDIIlTtSioqds](?:\s*[0-9OQDIIlTtSioqds]){4})\b")
CODEISH_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{4,8}\b")
PADDLE_OCR = None
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PADDLE_DET_MODEL = "PP-OCRv5_mobile_det"
PADDLE_REC_MODEL = "PP-OCRv5_mobile_rec"
PADDLE_DET_LIMIT_SIDE_LEN = 960
PADDLE_DET_LIMIT_TYPE = "max"
PADDLE_DET_THRESH = 0.3
PADDLE_DET_BOX_THRESH = 0.6
PADDLE_DET_UNCLIP_RATIO = 1.5
PADDLE_REC_SCORE_THRESH = 0.0
AGGRESSIVE_CODE_NORMALIZE = True


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def check_dependencies() -> None:
    if cv2 is None:
        fail("Missing Python package: opencv-python. Install it with: poetry add opencv-python")
    if PaddleOCR is None:
        fail("Missing Python package: paddleocr. Install it with: poetry add paddleocr paddlepaddle")


def get_ocr():
    global PADDLE_OCR
    if PADDLE_OCR is None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                PADDLE_OCR = PaddleOCR(
                    text_detection_model_name=PADDLE_DET_MODEL,
                    text_recognition_model_name=PADDLE_REC_MODEL,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                text_det_limit_side_len=PADDLE_DET_LIMIT_SIDE_LEN,
                text_det_limit_type=PADDLE_DET_LIMIT_TYPE,
                text_det_thresh=PADDLE_DET_THRESH,
                    text_det_box_thresh=PADDLE_DET_BOX_THRESH,
                    text_det_unclip_ratio=PADDLE_DET_UNCLIP_RATIO,
                    text_rec_score_thresh=PADDLE_REC_SCORE_THRESH,
                )
        except Exception as exc:
            fail(f"Could not initialize PaddleOCR: {exc}")
    return PADDLE_OCR


def initialize_ocr() -> None:
    try:
        get_ocr()
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"Could not initialize PaddleOCR: {exc}")


def preprocess_variants(frame):
    return [frame]


def raw_preprocess_variants(frame):
    return [("normal", frame)]


def crop_frame(frame, roi: tuple[int, int, int, int] | None):
    if not roi:
        return frame

    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


def digital_zoom(frame, zoom: float):
    if zoom <= 1.0:
        return frame

    height, width = frame.shape[:2]
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))
    x1 = max(0, (width - crop_width) // 2)
    y1 = max(0, (height - crop_height) // 2)
    cropped = frame[y1 : y1 + crop_height, x1 : x1 + crop_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)


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


def raw_ocr_frames(frame, roi: tuple[int, int, int, int] | None):
    if roi:
        return [("roi", crop_frame(frame, roi))]

    frames = [("full", frame)]
    screen = detect_screen_crop(frame)
    if screen is not None and screen.size:
        frames.append(("screen", screen))
    return frames


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


def paddleocr_results(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        predictions = get_ocr().predict(image)
    results = []

    for prediction in predictions:
        if isinstance(prediction, dict):
            boxes = prediction.get("rec_polys") or prediction.get("dt_polys") or []
            texts = prediction.get("rec_texts") or []
            scores = prediction.get("rec_scores") or []
            for box, text, score in zip(boxes, texts, scores):
                results.append((box.tolist() if hasattr(box, "tolist") else box, text, float(score)))
            continue

        # Compatibility fallback for older PaddleOCR tuple/list result shapes.
        for item in prediction or []:
            if not item or len(item) < 2:
                continue
            box = item[0]
            text_info = item[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text, score = text_info[0], text_info[1]
                results.append((box, text, float(score)))

    return results


def box_center(box):
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def box_height(box):
    ys = [point[1] for point in box]
    return max(ys) - min(ys)


def box_width(box):
    xs = [point[0] for point in box]
    return max(xs) - min(xs)


def results_to_lines(results, min_confidence: float = 0.35, min_height: float = 18) -> str:
    items = []
    for box, text, confidence in results:
        text = re.sub(r"\s+", " ", text).strip()
        if not text or confidence < min_confidence:
            continue
        height = max(box_height(box), 1)
        width = max(box_width(box), 1)
        if height < min_height:
            continue
        x_center, y_center = box_center(box)
        items.append(
            {
                "text": text,
                "x": x_center,
                "y": y_center,
                "height": height,
                "width": width,
            }
        )

    if not items:
        return ""

    items.sort(key=lambda item: item["y"])
    lines = []
    for item in items:
        for line in lines:
            tolerance = max(line["height"], item["height"]) * 0.65
            if abs(item["y"] - line["y"]) <= tolerance:
                line["items"].append(item)
                line["y"] = sum(part["y"] for part in line["items"]) / len(line["items"])
                line["height"] = max(line["height"], item["height"])
                break
        else:
            lines.append({"y": item["y"], "height": item["height"], "items": [item]})

    output_lines = []
    for line in sorted(lines, key=lambda line: line["y"]):
        parts = sorted(line["items"], key=lambda item: item["x"])
        output_lines.append(" ".join(part["text"] for part in parts))
    return cleanup_raw_text("\n".join(output_lines))


DIGIT_SLOT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "D": "0",
        "d": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "T": "1",
        "t": "1",
        "S": "5",
        "s": "5",
    }
)


def normalize_digit_slots(text: str) -> str:
    return re.sub(r"\s+", "", text).translate(DIGIT_SLOT_TRANSLATION)


def normalize_l_code(match: re.Match) -> str:
    digits = normalize_digit_slots(match.group(1))
    return f"L{digits}" if digits.isdigit() else match.group(0).upper()


def normalize_codeish_token(match: re.Match) -> str:
    token = match.group(0).upper()
    if token.startswith("L") and len(token) == 6:
        return LIKELY_L_CODE_RE.sub(normalize_l_code, token)
    if AGGRESSIVE_CODE_NORMALIZE and any(char.isdigit() for char in token):
        token = re.sub(r"S$", "5", token)
    return token


def normalize_trailing_code_token(match: re.Match) -> str:
    prefix, token = match.groups()
    token = normalize_codeish_token(re.match(r".+", token))
    if AGGRESSIVE_CODE_NORMALIZE:
        token = re.sub(r"S$", "5", token)
        token = re.sub(r"A$", "4", token)
    return f"{prefix}{token}"


def cleanup_raw_text(text: str) -> str:
    lines = []
    for line in compact_lines(text):
        line = SPACED_L_CODE_RE.sub(normalize_l_code, line)
        cleaned = LIKELY_L_CODE_RE.sub(normalize_l_code, line)
        cleaned = CODEISH_TOKEN_RE.sub(normalize_codeish_token, cleaned)
        cleaned = re.sub(r"\b([A-Z0-9]{4,8})\s*\(([^)]*)\)", r"\1 (\2)", cleaned)
        cleaned = re.sub(r"\b([A-Z0-9]{4,8})\s*[~_=]\s*([A-Z0-9]{3,8})\b", r"\1 - \2", cleaned)
        cleaned = re.sub(r"\b(L\d{5})\s*[-–—]\s*([A-Z0-9]{3,8})\b", r"\1 - \2", cleaned)
        cleaned = re.sub(r"\b(L\d{5})\s+(?:[S5Xx]\s+)?([A-Z0-9]{3,8})\b", r"\1 - \2", cleaned)
        cleaned = re.sub(r"\)\s*-\s*", ") - ", cleaned)
        cleaned = re.sub(r"(\bL\d{5}\s+-\s+)([A-Z0-9]{3,8})\b", normalize_trailing_code_token, cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        lines.append(cleaned)

    if len(lines) > 1:
        lines = [line for line in lines if not re.fullmatch(r"[A-Z0-9]", line)]

    cleaned_lines = []
    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if re.search(r"\s+[A-Z0-9]$", line) and re.match(r"L\d{5}\b", next_line):
            line = re.sub(r"\s+[A-Z0-9]$", "", line)
        cleaned_lines.append(line)

    if any(re.search(r"\bL\d{5}\b", line) for line in cleaned_lines):
        cleaned_lines = [
            line
            for line in cleaned_lines
            if not (
                re.fullmatch(r"[A-Z]{4,}", line)
                and "INBOUND" not in line.upper()
                and "CM" not in line.upper()
            )
        ]

    return "\n".join(cleaned_lines)


def read_raw_text(
    frame,
    roi: tuple[int, int, int, int] | None,
    min_confidence: float = 0.35,
    min_height: float = 18,
) -> str:
    best_text = ""
    best_score = 0.0
    seen = set()
    crops = raw_ocr_frames(frame, roi)
    variant_passes = [
        [("normal", target) for crop_name, target in crops],
        [
            (variant_name, variant)
            for crop_name, target in crops
            for variant_name, variant in raw_preprocess_variants(target)
            if variant_name != "normal"
        ],
    ]

    for pass_index, variants in enumerate(variant_passes):
        for variant_name, variant in variants:
            text = results_to_lines(
                paddleocr_results(variant),
                min_confidence=min_confidence,
                min_height=min_height,
            )
            if not text or text in seen:
                continue
            seen.add(text)

            score = text_score(text)
            if looks_like_noise(text):
                score *= 0.35
            if variant_name.startswith("bw"):
                score -= 2

            if score > best_score:
                best_text = text
                best_score = score
        if pass_index == 0 and best_score >= 18 and len(compact_lines(best_text)) >= 2:
            break
    return best_text.strip()


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


def normalize_code_token(token: str) -> str:
    if not any(char.isdigit() for char in token):
        return token
    return token.translate(str.maketrans({"O": "0", "S": "5", "I": "1"}))


def extract_fields(text: str) -> dict[str, str]:
    normalized = normalize_text(text).upper()
    var1_match = VAR1_RE.search(normalized)
    if not var1_match:
        var1_match = FALLBACK_VAR1_RE.search(normalized)
    var3_match = VAR3_RE.search(normalized)
    if not var3_match:
        var3_match = DASH_CODE_RE.search(normalized)
    if not var3_match:
        var3_match = AFTER_L_CODE_RE.search(normalized)

    return {
        "var1": normalize_code_token(var1_match.group(1)) if var1_match else "",
        "var2": find_code(normalized) or "",
        "var3": normalize_code_token(var3_match.group(1).replace("W2", "WZ", 1)) if var3_match else "",
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


def ocr_text_with_confidence(image) -> tuple[str, float, float, int]:
    results = paddleocr_results(image)
    words = []
    confidences = []

    for _, word, confidence in results:
        word = word.strip()
        if not word:
            continue
        confidence_value = float(confidence) * 100
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
        text, average_confidence, max_confidence, word_count = ocr_text_with_confidence(variant)
        if not text or text in seen:
            continue

        seen.add(text)
        normalized = normalize_text(text)
        fields = extract_fields(normalized)
        if row_status(fields) == "parsed":
            return fields["var2"], normalized

        if is_readable_text(text, average_confidence, max_confidence, word_count) and not looks_like_noise(text):
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


def draw_overlay(frame, status: str, roi: tuple[int, int, int, int] | None, zoom: float = 1.0) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"SPACE: OCR   +/-: zoom {zoom:.1f}x   Q/ESC: quit",
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
    parser = argparse.ArgumentParser(description="Read text from a live camera or sample images with PaddleOCR.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live camera mode.")
    parser.add_argument("--image", type=Path, help="OCR one image instead of opening the camera.")
    parser.add_argument("--images", type=Path, help="OCR an image file or every image in a folder.")
    parser.add_argument("--list-cameras", action="store_true", help="Scan camera indexes and report which ones open.")
    parser.add_argument("--snapshot-cameras", action="store_true", help="Save one frame from each camera index.")
    parser.add_argument("--scan-max", type=int, default=2, help="Highest camera index to scan with --list-cameras.")
    parser.add_argument("--code-only", action="store_true", help="Print only the extracted L+5 digit code.")
    parser.add_argument("--structured", action="store_true", help="Format known code screens into normalized lines.")
    parser.add_argument("--roi", help="Optional fixed crop rectangle as x,y,w,h.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Digital center zoom for live camera mode.")
    parser.add_argument("--debug-image", type=Path, help="Save the first OCR target image to this path on SPACE.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Ignore PaddleOCR boxes below this confidence.")
    parser.add_argument("--min-height", type=float, default=18, help="Ignore PaddleOCR boxes shorter than this many pixels.")
    parser.add_argument("--det-limit-side-len", type=int, default=960, help="PaddleOCR detection limit side length.")
    parser.add_argument("--det-limit-type", choices=("min", "max"), default="max", help="How PaddleOCR scales images for detection.")
    parser.add_argument("--det-thresh", type=float, default=0.3, help="PaddleOCR text detection threshold.")
    parser.add_argument("--det-box-thresh", type=float, default=0.6, help="PaddleOCR detected box confidence threshold.")
    parser.add_argument("--det-unclip-ratio", type=float, default=1.5, help="PaddleOCR detection box expansion ratio.")
    parser.add_argument("--rec-score-thresh", type=float, default=0.0, help="PaddleOCR recognition score threshold.")
    parser.add_argument(
        "--no-aggressive-code-normalize",
        action="store_true",
        help="Disable extra OCR cleanup for mixed code tokens, such as final S/5 and A/4.",
    )
    return parser.parse_args()


def configure_paddleocr(args: argparse.Namespace) -> None:
    global PADDLE_DET_LIMIT_SIDE_LEN, PADDLE_DET_THRESH, PADDLE_DET_BOX_THRESH
    global PADDLE_DET_LIMIT_TYPE, PADDLE_DET_UNCLIP_RATIO, PADDLE_REC_SCORE_THRESH
    global AGGRESSIVE_CODE_NORMALIZE
    PADDLE_DET_LIMIT_SIDE_LEN = args.det_limit_side_len
    PADDLE_DET_LIMIT_TYPE = args.det_limit_type
    PADDLE_DET_THRESH = args.det_thresh
    PADDLE_DET_BOX_THRESH = args.det_box_thresh
    PADDLE_DET_UNCLIP_RATIO = args.det_unclip_ratio
    PADDLE_REC_SCORE_THRESH = args.rec_score_thresh
    AGGRESSIVE_CODE_NORMALIZE = not args.no_aggressive_code_normalize


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


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )
    fail(f"Image path not found: {path}")


def ocr_image(
    path: Path,
    roi: tuple[int, int, int, int] | None,
    structured: bool,
    code_only: bool,
    min_confidence: float,
    min_height: float,
) -> str:
    frame = cv2.imread(str(path))
    if frame is None:
        return "Could not read image file."

    if not structured and not code_only:
        return read_raw_text(frame, roi, min_confidence, min_height) or "No text detected"

    result = ocr_best_frame(frame, roi)
    if result is None:
        return "No OCR result."

    _, code, text, fields, _ = result
    if code_only:
        return code or "No L + 5 digit code detected."
    return format_ocr_result(text, fields)


def process_images(
    path: Path,
    roi: tuple[int, int, int, int] | None,
    structured: bool,
    code_only: bool,
    min_confidence: float,
    min_height: float,
) -> None:
    paths = image_paths(path)
    if not paths:
        fail(f"No image files found in: {path}")

    multiple = len(paths) > 1
    for index, image_path in enumerate(paths):
        if multiple:
            if index:
                print()
            print(f"=== {image_path.name} ===")
        print(ocr_image(image_path, roi, structured, code_only, min_confidence, min_height))


def default_image_path() -> Path:
    for path in (Path("zoomin"), Path("nonglare"), Path("tests"), Path("sample2.PNG")):
        if path.exists():
            return path
    fail("No image path provided. Use --images folder_name, --image file_name, or --camera 0.")


def main() -> None:
    args = parse_args()
    roi = parse_roi(args.roi)
    zoom = max(1.0, args.zoom)
    configure_paddleocr(args)

    if args.list_cameras:
        list_cameras(args.scan_max)
        return

    if args.snapshot_cameras:
        snapshot_cameras(args.scan_max)
        return

    check_dependencies()
    initialize_ocr()

    if args.image or args.images:
        process_images(
            args.images or args.image,
            roi,
            args.structured,
            args.code_only,
            args.min_confidence,
            args.min_height,
        )
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

            ocr_frame = digital_zoom(frame, zoom)
            display = ocr_frame.copy()
            draw_overlay(display, status, roi, zoom)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                if args.debug_image:
                    save_debug_target(ocr_frame, roi, args.debug_image)

                if not args.structured and not args.code_only:
                    output = read_raw_text(
                        ocr_frame,
                        roi,
                        min_confidence=args.min_confidence,
                        min_height=args.min_height,
                    ) or "No text detected"
                    status = "No text detected" if output == "No text detected" else "Text detected"
                    print(output)
                    continue

                result = ocr_best_frame(ocr_frame, roi)
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
            elif key in (ord("+"), ord("=")):
                zoom = min(4.0, zoom + 0.2)
                status = f"Zoom {zoom:.1f}x"
            elif key in (ord("-"), ord("_")):
                zoom = max(1.0, zoom - 0.2)
                status = f"Zoom {zoom:.1f}x"
            elif key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
