#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter records with non-empty 'file_name' from metadata.jsonl to multimodal.jsonl.
Usage (from repo root or same dir):
    python script/gaia/dataset/2023/validation/filter_multimodal.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

SRC = Path(__file__).with_name("metadata.jsonl")
DST = Path(__file__).with_name("multimodal.jsonl")


def has_non_empty_file_name(rec: Dict[str, Any]) -> bool:
    if not isinstance(rec, dict):
        return False
    if "file_name" not in rec:
        return False
    v = rec.get("file_name")
    if v is None:
        return False
    if isinstance(v, str):
        return len(v.strip()) > 0
    # accept non-str truthy values too
    return bool(v)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Source file not found: {SRC}")

    kept = 0
    total = 0

    with SRC.open("r", encoding="utf-8") as fin, DST.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # skip bad lines silently
                continue
            if has_non_empty_file_name(rec):
                json.dump(rec, fout, ensure_ascii=False)
                fout.write("\n")
                kept += 1

    print(f"Done. total={total}, kept={kept}, src='{SRC.name}', dst='{DST.name}', dst_size={DST.stat().st_size if DST.exists() else 0}")


if __name__ == "__main__":
    main()
