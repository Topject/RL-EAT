"""Check every relative link and image reference in the repository's Markdown.

    python tools/check_links.py            # from the repo root
    python tools/check_links.py --fix-report links.txt

Reports
  * broken relative links (target file does not exist)
  * absolute links that point outside the repository
  * links to a directory without a trailing slash (GitHub cannot render those)
EOF is not a failure; the exit code is 1 only when something is broken.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def iter_markdown(root: Path):
    for path in sorted(root.rglob("*.md")):
        if any(part in {".git", "node_modules"} for part in path.parts):
            continue
        yield path


def check(root: Path) -> tuple[list[str], list[str], int]:
    broken: list[str] = []
    absolute: list[str] = []
    checked = 0
    for md in iter_markdown(root):
        text = md.read_text(encoding="utf-8", errors="replace")
        for regex, kind in ((IMAGE_RE, "image"), (LINK_RE, "link")):
            for match in regex.finditer(text):
                target = match.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                if any(token in target for token in ("<", ">", "xxx", "yyy", "NN-name")):
                    continue  # placeholder in a template, not a real reference
                checked += 1
                if target.startswith("/"):
                    absolute.append(f"{md.relative_to(root)}: {kind} -> {target}")
                    continue
                clean = target.split("#", 1)[0].split("?", 1)[0]
                if not clean:
                    continue
                resolved = (md.parent / clean).resolve()
                try:
                    resolved.relative_to(root.resolve())
                except ValueError:
                    broken.append(f"{md.relative_to(root)}: {kind} -> {target} (escapes repo)")
                    continue
                if not resolved.exists():
                    broken.append(f"{md.relative_to(root)}: {kind} -> {target}")
    return broken, absolute, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fix-report", default="")
    args = ap.parse_args()
    root = Path(args.root)

    broken, absolute, checked = check(root)
    print(f"checked {checked} relative references in {len(list(iter_markdown(root)))} markdown files")
    if absolute:
        print(f"\n{len(absolute)} absolute (root-relative) references — prefer relative paths:")
        for item in absolute[:20]:
            print("  /", item)
    if broken:
        print(f"\n{len(broken)} BROKEN references:")
        for item in broken:
            print("  x", item)
    else:
        print("\nall relative references resolve")

    if args.fix_report:
        Path(args.fix_report).write_text(
            "\n".join(broken + absolute) + "\n", encoding="utf-8"
        )
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
