import json

from google import genai
from google.genai import types

from schema import AliasGroups
from utils import norm_pair, parse_model

S2_SYSTEM_PROMPT = """
You merge aliases into canonical person identities.
Merge ONLY if at least TWO hold:
(1) co-occurrence/interaction context,
(2) explicit aka/title/codename,
(3) consistent role/relationship pattern.
Be conservative. No guessing.
Return strict JSON: { "groups": [ { "canonical": "...", "aliases": ["...", "..."] } ] }.
""".strip()


def s2_user_prompt(unique_names: list[str], appearances: dict[str, list[int]]) -> str:
    return json.dumps(
        {"UNIQUE_NAMES": unique_names, "APPEARANCES": appearances, "HINTS": []},
        ensure_ascii=False,
        indent=2,
    )


def run_s2_alias(
    client: genai.Client, unique_names: list[str], appearances: dict[str, list[int]]
) -> AliasGroups:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [{"text": s2_user_prompt(unique_names, appearances)}],
            },
        ],
        config=types.GenerateContentConfig(
            system_instruction=S2_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=AliasGroups,
        ),
    )
    raw = getattr(resp, "parsed", None)
    if raw is not None:
        return parse_model(AliasGroups, raw)
    if resp.text is not None:
        return parse_model(AliasGroups, resp.text)
    raise RuntimeError("Alias generation returned no usable payload.")


def build_alias_map(groups: AliasGroups) -> dict[str, str]:
    mp: dict[str, str] = {}
    for g in groups.groups:
        for a in g.aliases:
            mp[a] = g.canonical
    return mp


def alias_name(n: str, amap: dict[str, str]) -> str:
    low = n.lower()
    for k, v in amap.items():
        if low == k.lower():
            return v
    return n


def alias_pair(p: tuple[str, str], amap: dict[str, str]) -> tuple[str, str]:
    return norm_pair(alias_name(p[0], amap), alias_name(p[1], amap))
