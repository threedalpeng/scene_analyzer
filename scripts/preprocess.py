import argparse
import json
import re
import unicodedata

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def safe_filename(name: str) -> str:
    # Normalize and remove file-system unfriendly chars
    s = unicodedata.normalize("NFKC", name)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "-")
    s = re.sub(r"[^A-Za-z0-9 _\.-]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s or "UNKNOWN_MOVIE"


def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # collapse whitespace
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def parse_segment_etree(segment_elem: ET.Element) -> str:
    lines: List[str] = []
    current_char: Optional[str] = None
    # Iterate in order to preserve character-dialogue sequence
    for child in list(segment_elem):
        tag = child.tag.lower()
        txt = (child.text or "").strip()
        if not txt:
            continue
        if tag == "stage_direction":
            lines.append(txt)
            current_char = None
        elif tag == "segment_description":
            lines.append(txt)
        elif tag == "character":
            current_char = txt
        elif tag == "dialogue":
            if current_char:
                lines.append(f"{current_char}: {txt}")
            else:
                lines.append(txt)
        else:
            # Unknown tag: keep as plain text
            lines.append(txt)
    return "\n".join(normalize_text(l) for l in lines if l)


def parse_script(script_text: str) -> List[str]:
    """Return list of segment texts in order."""
    if not script_text:
        return []
    s = script_text.strip()
    # Ensure there's a single root; many entries already have <script>…</script>
    if not re.search(r"<\s*script\s*>", s, re.I):
        s = "<script>\n" + s + "\n</script>"
    try:
        root = ET.fromstring(s)
        segments = []
        for sc in root.findall(".//segment"):
            segments.append(parse_segment_etree(sc))
        if segments:
            return segments
    except ET.ParseError:
        pass

    # Fallback: regex-based extraction of segments
    segments = []
    for m in re.finditer(r"<segment>(.*?)</segment>", s, re.I | re.S):
        block = m.group(1)
        # Collect stage_direction & description
        parts: List[str] = []
        for tag in ["stage_direction", "segment_description"]:
            mm = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.I | re.S)
            if mm and mm.group(1).strip():
                parts.append(re.sub(r"\s+", " ", mm.group(1).strip()))
        # Character / dialogue in order (simplified: sequential scan)
        # We'll scan tags in order and pair character+dialogue when encountered
        tokens = re.findall(
            r"<(stage_direction|segment_description|character|dialogue)>(.*?)</\1>",
            block,
            re.I | re.S,
        )
        current_char = None
        for tag, val in tokens:
            tag = tag.lower()
            val = re.sub(r"\s+", " ", val.strip())
            if tag == "character":
                current_char = val
            elif tag == "dialogue":
                if current_char:
                    parts.append(f"{current_char}: {val}")
                else:
                    parts.append(val)
            # stage_direction/segment_description are already added at top, skip duplicates
        text = "\n".join(p for p in parts if p)
        segments.append(text.strip())
    return segments


def write_jsonl(movie_name: str, segments: List[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = safe_filename(movie_name) + ".jsonl"
    out_path = out_dir / fname
    with open(out_path, "w", encoding="utf-8") as f:
        for i, sc in enumerate(segments, 1):
            obj = {"segment": i, "text": sc}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def process_dataset(
    dataset_name: str, out_dir: str, splits=("train", "validation", "test")
):
    from datasets import load_dataset  # lazy import

    ds_dict = load_dataset(dataset_name)
    total_movies = 0
    for split in splits:
        if split not in ds_dict:
            continue
        ds = ds_dict[split]
        for row in ds:
            movie_name = row.get("movie_name") or row.get("imdb_id") or "UNKNOWN_MOVIE"
            script = row.get("script") or ""
            segments = parse_script(script)
            write_jsonl(movie_name, segments, Path(out_dir))
            total_movies += 1
    return total_movies


def main():
    ap = argparse.ArgumentParser(
        description="Split MovieSum scripts into segment-level JSONL files."
    )
    ap.add_argument("--dataset", default="rohitsaxena/MovieSum", help="HF dataset path")
    ap.add_argument("--out_dir", default="./movies", help="Output directory")
    ap.add_argument(
        "--no_hf", action="store_true", help="Skip HF load (useful for dry run)"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_hf:
        print("Skipping HF dataset load (--no_hf).")
        return

    n = process_dataset(args.dataset, str(out_dir))
    print(f"Done. Wrote JSONL files for {n} movies to {out_dir}")


if __name__ == "__main__":
    main()
