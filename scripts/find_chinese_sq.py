#!/usr/bin/env python3
"""
find_chinese_sq.py

Scan `scenarios/streaming_queue` for .py and .yml/.yaml files that contain Chinese characters.
- For Python files: checks comments (via tokenize) and docstrings (via ast)
- For YAML files: scans the full file text
- Excludes any path that contains `/data/`, `/result/` or `/results/` segments

Usage:
    python3 scripts/find_chinese_sq.py --path scenarios/streaming_queue --report report.json

Outputs a human-readable summary to stdout and optionally writes a JSON report.
"""

import argparse
import ast
import json
import os
import re
import sys
import tokenize
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
# Paths (substrings) to exclude from scanning. Keep variants with/without trailing sep where useful.
EXCLUDE_SEGMENTS = (
    os.sep + 'data' + os.sep,
    os.sep + 'result',
    os.sep + 'results' + os.sep,
    # Exclude the gaia workspaces directory as requested
    'scenarios' + os.sep + 'gaia' + os.sep + 'workspaces' + os.sep,
)


def contains_chinese_in_text(text: str) -> bool:
    return bool(CHINESE_RE.search(text))


def scan_python_file(path: Path) -> Dict[str, Any]:
    """Return details about Chinese occurrences in comments and docstrings for a Python file."""
    details: Dict[str, Any] = {"path": str(path), "comments": [], "docstrings": []}
    try:
        raw = path.read_bytes()
    except Exception as e:
        details["error"] = f"read_error: {e}"
        return details

    # Extract comment tokens
    try:
        for tok in tokenize.tokenize(BytesIO(raw).readline):
            if tok.type == tokenize.COMMENT:
                if CHINESE_RE.search(tok.string):
                    details["comments"].append({"line": tok.start[0], "text": tok.string})
    except Exception:
        # Tokenize may fail on incomplete files; fall back to line scan
        try:
            text = raw.decode('utf-8', errors='ignore')
            for i, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith('#') and CHINESE_RE.search(line):
                    details["comments"].append({"line": i, "text": line.strip()})
        except Exception:
            pass

    # Parse AST for docstrings
    try:
        node = ast.parse(raw.decode('utf-8'), filename=str(path))
        # Module docstring
        mod_doc = ast.get_docstring(node)
        if mod_doc and CHINESE_RE.search(mod_doc):
            details["docstrings"].append({"type": "module", "text": mod_doc})

        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(child)
                if doc and CHINESE_RE.search(doc):
                    obj_type = 'class' if isinstance(child, ast.ClassDef) else 'function'
                    name = getattr(child, 'name', '<anon>')
                    details["docstrings"].append({"type": obj_type, "name": name, "text": doc})
    except Exception:
        # If AST parse fails, try to approximate by searching triple-quoted strings
        try:
            text = raw.decode('utf-8', errors='ignore')
            triple_re = re.compile(r"([\"']{3})(.*?)(\1)", re.S)
            for m in triple_re.finditer(text):
                content = m.group(2)
                if CHINESE_RE.search(content):
                    details["docstrings"].append({"type": "approx", "text": content[:200]})
        except Exception:
            pass

    return details


def scan_yaml_file(path: Path) -> Dict[str, Any]:
    details: Dict[str, Any] = {"path": str(path), "matches": []}
    try:
        text = path.read_text(encoding='utf-8')
    except Exception as e:
        details["error"] = f"read_error: {e}"
        return details

    for i, line in enumerate(text.splitlines(), start=1):
        if CHINESE_RE.search(line):
            details["matches"].append({"line": i, "text": line.strip()})
    return details


def should_exclude(path: Path) -> bool:
    s = str(path)
    for seg in EXCLUDE_SEGMENTS:
        if seg in s:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='scenarios', help='Root path to scan')
    parser.add_argument('--report', '-r', default=None, help='Optional JSON report path')
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Path not found: {root}")
        return 2

    py_files: List[Path] = []
    yaml_files: List[Path] = []

    for p in root.rglob('*'):
        if p.is_file():
            if should_exclude(p):
                continue
            if p.suffix == '.py':
                py_files.append(p)
            elif p.suffix in ('.yaml', '.yml'):
                yaml_files.append(p)

    results: Dict[str, Any] = {"scanned_path": str(root), "py_total": len(py_files), "yaml_total": len(yaml_files), "py": [], "yaml": []}

    for p in sorted(py_files):
        det = scan_python_file(p)
        if det.get('comments') or det.get('docstrings') or det.get('error'):
            results['py'].append(det)

    for y in sorted(yaml_files):
        det = scan_yaml_file(y)
        if det.get('matches') or det.get('error'):
            results['yaml'].append(det)

    total_files = results['py_total'] + results['yaml_total']
    files_with_chinese = len(results['py']) + len(results['yaml'])

    print('Scanning root:', results['scanned_path'])
    print('Total files scanned (.py + .yaml/.yml):', total_files)
    print('Files containing Chinese in comments/docstrings or content:', files_with_chinese)

    if files_with_chinese > 0:
        print('\nDetailed list:')
        for item in results['py']:
            print('\n[PY]', item['path'])
            if 'error' in item:
                print('  ERROR:', item['error'])
            for c in item.get('comments', []):
                print(f"  COMMENT L{c['line']}: {c['text']}")
            for d in item.get('docstrings', []):
                t = d.get('text') if 'text' in d else None
                name = d.get('name', d.get('type', ''))
                print(f"  DOCSTRING ({name}): { (t[:120] + '...') if t and len(t)>120 else t }")

        for item in results['yaml']:
            print('\n[YAML]', item['path'])
            if 'error' in item:
                print('  ERROR:', item['error'])
            for m in item.get('matches', []):
                print(f"  L{m['line']}: {m['text']}")

    else:
        print('No Chinese found in .py comments/docstrings or .yaml files under the scanned path.')

    if args.report:
        try:
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print('\nWrote report to', args.report)
        except Exception as e:
            print('Failed to write report:', e)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
