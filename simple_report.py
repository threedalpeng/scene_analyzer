# analyze_cooperation_simple.py
# Usage:
#   export GOOGLE_API_KEY="YOUR_KEY"
#   pip install google-genai
#   python analyze_cooperation_simple.py scenes.json --out report.md

import argparse
import json
import os
from typing import List, Dict
from google import genai
from google.genai import types

SYSTEM = """You are a careful narrative analyst. Your only sources of truth are the scene texts provided to you. 
Your tasks:
- Identify ONLY cooperative or de-escalating interactions that are *explicitly evidenced* in the text.
- Distinguish voluntary cooperation from coerced compliance.
- When (and only when) trust/solidarity between two characters clearly shifts, record a relationship shift with a small signed delta.
- Always cite line references (Lx or Lx–Ly) and keep quotes short.

Constraints:
- No outside knowledge. No speculation. If unclear, omit.
- Use the fixed label set for acts: [Info-Share | Coord-Action | Protect-Rescue | De-escalate | Comfort | Resource-Share | Apology | Negotiation | Compliance | Coercion]
- Volition: {Voluntary | Coerced}
- Outcome: {Success | Partial | Fail}
- Relationship Δ: one of {-2, -1, 0, +1, +2}; use only if a shift is clearly evidenced, else write "None".
- Be concise and structured. Return strictly in the requested Markdown format.
"""

SCENE_PROMPT = """Scene ID: {scene_id}

Text with line numbers:
{numbered_text}

Produce Markdown in the exact format:

## Scene {scene_id}
Cooperative Acts:
- Actor → Target — {{Label from [Info-Share|Coord-Action|Protect-Rescue|De-escalate|Comfort|Resource-Share|Apology|Negotiation|Compliance|Coercion]}} ({{Voluntary|Coerced}}; Outcome={{Success|Partial|Fail}}) — "short evidence quote" (Lx–Ly)
- (add more bullets as needed; omit if none)

Relationship Shifts:
- Pair(A ↔ B) — Δ={{-2|-1|0|+1|+2}} — one-line reason grounded in evidence — "short quote" (Lx–Ly)
- (add more only if clearly evidenced; write "None" if no shift)

Summary (2–3 sentences):
- Focus ONLY on (i) what concrete cooperative results were achieved, and (ii) whether any relationship shift occurred. Keep it under 90 words.

"""

FINAL_PROMPT = """You will write a concise overall report about cooperation across scenes.

Inputs (per-scene digests in fixed format):
---
{scene_digests}
---

Write a single Markdown report with the following sections:

# Cooperation Report

**Overview (4–6 sentences)**
- Describe who tends to cooperate with whom and typical cooperative patterns (rescue, protection, coordination, apology, resource sharing, de-escalation).
- Explicitly distinguish *voluntary cooperation* vs *coerced compliance*.
- Base claims ONLY on the digests.

## Key Cooperative Pairs
- 3–6 bullets. Each bullet: Pair — why they matter — cite at least one scene like [Scene 12].

## Notable Events
- 5–10 bullets. Each: [Scene N] — one-line description of the cooperative event and its effect on the situation.

## Relationship Shifts
- Summarize clear trust/solidarity shifts recorded in the digests.
- Mention direction (strengthening vs weakening) and cite scenes like [Scene N].
- If shifts are scarce or ambiguous, say so.

## Current State
- 2–4 sentences describing the present cooperation trend, tensions that remain, and any temporary truces or ongoing protections.
- No outside facts.

Formatting rules:
- Use ONLY information from the digests.
- Reference scenes with square brackets: [Scene N].
- Be precise and economical with words; avoid repetition.

"""


def add_line_numbers(text: str) -> str:
    out = []
    n = 1
    for line in text.splitlines():
        if line.strip():
            out.append(f"L{n}: {line}")
            n += 1
        else:
            out.append(line)
    return "\n".join(out)


def summarize_scene(client: genai.Client, model: str, scene_id: int, text: str) -> str:
    numbered = add_line_numbers(text)
    prompt = SCENE_PROMPT.format(scene_id=scene_id, numbered_text=numbered)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
            max_output_tokens=900,
        ),
    )
    txt = getattr(resp, "text", None)
    print(txt)
    if isinstance(txt, str) and txt.strip():
        return f"{txt.strip()}"
    # fallback through candidates
    candidates = getattr(resp, "candidates", None) or []
    for c in candidates:
        content = getattr(c, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    return f"{t.strip()}"
    raise RuntimeError(f"No text returned for scene {scene_id}")


def build_final_report(
    client: genai.Client, model: str, scene_digests: List[str]
) -> str:
    joined = "\n\n".join(scene_digests)
    prompt = FINAL_PROMPT.format(scene_digests=joined)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.3,
            max_output_tokens=1500,
        ),
    )
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    candidates = getattr(resp, "candidates", None) or []
    for c in candidates:
        content = getattr(c, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
    raise RuntimeError("No text returned for final report")


def main():
    ap = argparse.ArgumentParser(
        description="LLM-only simple cooperation analyzer per scene."
    )
    ap.add_argument(
        "scenes_path", help="scenes.json (list of {scene_id,int, text,str})"
    )
    ap.add_argument("--out", default="report.md", help="Output Markdown file")
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash-preview-09-2025",
        help="Gemini model (default: gemini-2.5-flash-preview-09-2025)",
    )
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY env var.")

    with open(args.scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, list):
        raise SystemExit("Input must be a list of {scene_id, text} objects.")

    client = genai.Client(api_key=api_key)

    digests: List[str] = []
    for obj in scenes:
        sid = obj.get("scene_id") or obj.get("scene") or obj.get("id")
        text = obj.get("text") or obj.get("content") or ""
        if not sid or not text:
            # skip malformed entries
            continue
        print(f"Scene: {sid} summarizing...")
        digest = summarize_scene(client, args.model, int(sid), text)
        digests.append(digest)

    final_report = build_final_report(client, args.model, digests)

    with open(args.out, "w", encoding="utf-8") as wf:
        wf.write("# Scene-by-Scene Cooperation Digests\n\n")
        wf.write("\n\n".join(digests))
        wf.write("\n\n---\n\n")
        wf.write(final_report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
