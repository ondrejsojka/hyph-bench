#!/usr/bin/env python3
"""
Filter unique Ukrainian words longer than 4 characters from a text file.
 
Usage:
    python3 filter_ukrainian_words.py <input_file> [output_file] [--count N]
 
If output_file is omitted, results are printed to stdout.
If --count is omitted, defaults to 500 words.
"""
 
import argparse
import re
import sys
from pathlib import Path
 
 
# Ukrainian alphabet (lowercase + uppercase, including apostrophe-like chars handled separately)
UKRAINIAN_LETTERS = (
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
    "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
)
 
# Apostrophe variants used in Ukrainian text:
#   ' (U+0027 apostrophe)
#   ’ (U+2019 right single quotation mark) — most common in modern typography
#   ʼ (U+02BC modifier letter apostrophe) — recommended by Ukrainian orthography
#   ` (U+0060 grave accent) — sometimes used as a fallback
APOSTROPHES = "'\u2019\u02BC`"
 
# A "word" here = a run of Ukrainian letters, optionally containing apostrophes
# between letters (e.g. комп'ютер, м'який, об'єкт). A leading or trailing
# apostrophe is NOT included in the match. Anything else (digits, punctuation,
# latin letters, hyphens, etc.) acts as a separator and is excluded.
WORD_RE = re.compile(
    f"[{UKRAINIAN_LETTERS}]+(?:[{APOSTROPHES}][{UKRAINIAN_LETTERS}]+)*"
)
 
 
def extract_words(text: str, min_length: int = 5) -> list[str]:
    """
    Extract unique Ukrainian words longer than (min_length - 1) characters.
    Words are lowercased, apostrophe variants are normalized to U+2019,
    and results are returned in order of first appearance.
    """
    seen = set()
    result = []
    for match in WORD_RE.finditer(text):
        word = match.group(0).lower()
        # Normalize all apostrophe variants to a single canonical form (’)
        # so "комп'ютер" and "комп’ютер" don't count as two different words.
        for ch in "'\u02BC`":
            word = word.replace(ch, "\u2019")
        if len(word) >= min_length and word not in seen:
            seen.add(word)
            result.append(word)
    return result
 
 
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract unique Ukrainian words longer than 4 characters."
    )
    parser.add_argument("input", help="Path to the input text file (UTF-8).")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional output file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="How many words to keep (default: 500).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Minimum word length, inclusive (default: 5, i.e. 'longer than 4').",
    )
    args = parser.parse_args()
 
    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1
 
    try:
        text = in_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files encoded in cp1251 (common for older Ukrainian texts)
        text = in_path.read_text(encoding="cp1251")
 
    words = extract_words(text, min_length=args.min_length)
    selected = words[: args.count]
 
    output_text = "\n".join(selected) + "\n"
 
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(
            f"Wrote {len(selected)} words to {args.output} "
            f"(found {len(words)} unique words total).",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(output_text)
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
