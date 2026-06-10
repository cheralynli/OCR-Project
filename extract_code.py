#!/usr/bin/env python3
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


IMAGE_PATH = Path("sample2.PNG")
CSV_PATH = Path("extracted_codes.csv")
CODE_RE = re.compile(r"\bL\s*(\d{5})\b")
CM_CODE_RE = re.compile(r"(?:\bL\s*)?(\d{5})\s*\(\s*CM\s*\)", re.IGNORECASE)


def fail(message: str, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def read_text_with_tesseract(path: Path) -> str:
    if not path.exists():
        fail(f"Image file not found: {path}")
    if not path.is_file():
        fail(f"Image path is not a file: {path}")

    try:
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "6"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        fail("Tesseract is not installed or is not on PATH.")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        fail(f"Tesseract OCR failed: {detail}")

    if not result.stdout.strip():
        fail("Tesseract did not return any text.")
    return result.stdout


def extract_code(text: str) -> str:
    match = CODE_RE.search(text)
    if not match:
        match = CM_CODE_RE.search(text)
    if not match:
        fail("No code matching L followed by 5 digits was found.")
    return f"L{match.group(1)}"


def save_code(code: str) -> None:
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["timestamp", "code"])
        writer.writerow([timestamp, code])


def main() -> None:
    text = read_text_with_tesseract(IMAGE_PATH)
    code = extract_code(text)
    save_code(code)
    print(code)


if __name__ == "__main__":
    main()
