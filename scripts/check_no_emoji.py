#!/usr/bin/env python3
"""
scripts/check_no_emoji.py - Scanner to detect emojis across the ParkLens codebase.

Scans all text files (.py, .md, .css, .html, .json, .ipynb, .toml, .txt, .yaml)
excluding data/, outputs/, .venv/, and cache folders.
Exits 0 if no emojis are found, or exits 1 with line-by-line findings.
"""

import sys
import re
from pathlib import Path

# Paths to ignore
IGNORE_DIRS = {
    ".venv", ".git", ".pytest_cache", ".mpl_cache", "__pycache__",
    "data", "outputs", ".edge_temp", "node_modules"
}

ALLOWED_EXTENSIONS = {
    ".py", ".md", ".css", ".html", ".json", ".ipynb", ".toml", ".txt", ".yaml", ".yml"
}

# Regex matching Unicode emoji ranges:
# U+1F300-U+1FAFF: Miscellaneous Symbols and Pictographs, Emoticons, Supplemental Symbols
# U+1F1E6-U+1F1FF: Regional indicator symbols (flags)
# U+2600-U+27BF: Miscellaneous Symbols, Dingbats
# U+2B50, U+2B1B, U+2B1C: Stars, squares
# U+231A, U+231B: Watch, hourglass
# U+23E9-U+23FA: Media / AV control symbols
# U+FE0F: Variation selector-16 (emoji presentation)
# U+200D: Zero width joiner
EMOJI_PATTERN = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"
    r"\U0001F1E6-\U0001F1FF"
    r"\u2600-\u27BF"
    r"\u2B50\u2B1B\u2B1C"
    r"\u231A\u231B"
    r"\u23E9-\u23FA"
    r"\uFE0F\u200D"
    r"]"
)

# Common HTML entities / CSS escapes for emojis
HTML_CSS_EMOJI_PATTERN = re.compile(
    r"(&#(?:12[89]\d{3}|1[3-9]\d{4}|x1[fF][0-9a-fA-F]{3});|\\[0-1][fF][0-9a-fA-F]{3})"
)


def scan_file(path: Path) -> list:
    findings = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return findings

    for line_num, line in enumerate(content.splitlines(), start=1):
        matches = EMOJI_PATTERN.findall(line)
        html_css_matches = HTML_CSS_EMOJI_PATTERN.findall(line)
        all_matches = matches + html_css_matches
        if all_matches:
            findings.append((line_num, line.strip(), all_matches))
    return findings


def main():
    root = Path(__file__).resolve().parent.parent
    findings_by_file = {}
    total_findings = 0

    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # Check if inside an ignored directory
        parts = p.relative_to(root).parts
        if any(part in IGNORE_DIRS for part in parts):
            continue
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        file_findings = scan_file(p)
        if file_findings:
            rel_path = str(p.relative_to(root)).replace("\\", "/")
            findings_by_file[rel_path] = file_findings
            total_findings += len(file_findings)

    if total_findings == 0:
        print("PASS: Zero emoji characters found across scanned files.")
        sys.exit(0)
    else:
        print(f"FAIL: Found {total_findings} emoji instances across {len(findings_by_file)} files:\n")
        for fpath, lines in findings_by_file.items():
            print(f"--- {fpath} ({len(lines)} occurrences) ---")
            for line_no, text, matched in lines[:5]:
                codepoints = [f"U+{ord(c):04X}" if len(c) == 1 else c for c in matched]
                safe_text = text.encode("ascii", "replace").decode("ascii")[:80]
                print(f"  L{line_no}: {safe_text} (matches: {codepoints})")
            if len(lines) > 5:
                print(f"  ... and {len(lines) - 5} more")
            print()
        sys.exit(1)


if __name__ == "__main__":
    main()
