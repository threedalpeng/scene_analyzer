import time
from collections import defaultdict

from google import genai
from google.genai import types
from pydantic import ValidationError

from schema import ActClue, IntraScenePayload, IntraScenePayloadAPI, ToMClue
from utils import (
    log_status,
    norm_pair,
    parse_model,
)

S1_SYSTEM_PROMPT = """
You extract structured character signals from a single scene transcript (dialogue + SDH cues like [sighs], [gasps]). 
Return ONLY JSON that satisfies the provided response schema exactly. Do not include any extra fields or text.

TASK
- Identify participants (all person names in the scene).
- Extract two types of items:
  1) tom_clues (Theory of Mind events)
  2) act_clues (interaction events)

HARD RULES (NO GUESSING)
- Use ONLY what is explicit in dialogue, described actions, or SDH cues. Do NOT infer from assumed camera work or unstated plot devices.
- Names must be copied as written in the scene text (no invention).
- Evidence is MANDATORY for every clue: a direct quote substring (≤ 200 characters) copied from the input scene.

PAIR & DIRECTION (VERY IMPORTANT)
- Every clue must include `pair`, an array of EXACTLY two person names [length=2] = [subject/actor, object/target].
- If an ACT mentions multiple actors and/or multiple targets, SPLIT into multiple act_clues, one per supported actor→target link.
  - Create a link ONLY if the actor→target connection is explicitly supported by the text (direct address, described action, or SDH).
  - If you cannot confidently identify BOTH sides, OMIT that clue.
- For ToM (Theory of Mind), pair = [thinker, target]. If the target is not explicit, OMIT the ToM clue.

ID FORMAT (STRICT)
- Every clue MUST provide an `id` using this exact pattern:
  - tom_clues -> `tom_s1_{scene:03d}_{index:04d}`
  - act_clues -> `act_s1_{scene:03d}_{index:04d}`
- `scene` is the scene number (zero-padded to 3 digits) and `index` starts at 1 for each scene and increments by 1 for each clue of that type.
- Do NOT reuse ids or invent any other format.

TOM (MENTAL EVENTS)
- Allowed kinds: BelievesAbout, FeelsTowards, IntendsTo, DesiresFor.
- `claim` must be a short, literal statement grounded in the scene (no speculation).
- Only produce a ToM clue if a line/action/SDH directly indicates belief/feeling/intention/desire.

ACT (INTERACTION EVENTS)
- `stance` must be one of: cooperation, hostility.
- `subtype` is a short label of the interaction (e.g., “rescues”, “shares intel”, “threatens”, “attacks”, “betrays”, “interrogates”).
- `axes` must use EXACT labels:
  - salience: {low, medium, high}
  - stakes: {minor, moderate, major}
  - durability: {momentary, temporary, persistent}
  - volition: {voluntary, coerced, accidental}
  - goal_alignment: {aligned, orthogonal, opposed}
  - consequence_refs: [] (array of integers; use [] if none)
- If any of the following are present, set durability="persistent":
  institutional pledge/order, affiliation change, death, identity reveal, legal verdict, irreversible destruction of a core asset.

PARTICIPANTS
- List all person names appearing in the scene (uniques). Do not include non-person entities.

QUALITY GUARDS
- Keep quotes ≤ 200 chars (hard clip).
- Omit any clue you are not fully confident about (no partial/approximate pairs).
- Keep JSON compact. No explanations.

OUTPUT SHAPE (must match the response schema)
{
  "participants": [string, ...],
  "tom_clues": [
    {
      "id": "tom_s1_{scene:03d}_{index:04d}",
      "scene": <integer>,
      "pair": ["NameA","NameB"],      // EXACTLY two names
      "modality": "tom",
      "evidence": "quoted <=200 chars",
      "kind": "BelievesAbout|FeelsTowards|IntendsTo|DesiresFor",
      "claim": "short literal statement"
    }
  ],
  "act_clues": [
    {
      "id": "act_s1_{scene:03d}_{index:04d}",
      "scene": <integer>,
      "pair": ["NameA","NameB"],      // EXACTLY two names = [actor, target]
      "modality": "act",
      "evidence": "quoted <=200 chars",
      "actors": ["NameA", ...],       // optional; include if clearly enumerated in text
      "targets": ["NameB", ...],      // optional; include if clearly enumerated in text
      "stance": "cooperation|hostility",
      "subtype": "short label",
      "axes": {
        "salience": "low|medium|high",
        "stakes": "minor|moderate|major",
        "durability": "momentary|temporary|persistent",
        "volition": "voluntary|coerced|accidental",
        "goal_alignment": "aligned|orthogonal|opposed",
        "consequence_refs": []
      }
    }
  ]
}
""".strip()


def s1_user_prompt(scene_id: int, text: str) -> str:
    return f"""SCENE_ID: {scene_id}
TEXT:
{text}

Extract:
- participants (all person names in this scene)
- tom_clues and act_clues following the system rules.
""".strip()


def build_s1_inline_requests(scenes: list[dict]) -> list[types.InlinedRequestDict]:
    reqs: list[types.InlinedRequestDict] = []
    for item in scenes:
        sid = int(item["scene"])
        text = str(item["text"])
        reqs.append(
            types.InlinedRequestDict(
                contents=[
                    {"role": "user", "parts": [{"text": s1_user_prompt(sid, text)}]},
                ],
                config=types.GenerateContentConfigDict(
                    system_instruction=S1_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=IntraScenePayloadAPI,
                ),
            )
        )
    return reqs


def run_s1_batch(
    client: genai.Client, scenes: list[dict], chunk: int = 10
) -> list[IntraScenePayload]:
    outs: list[IntraScenePayload] = []

    total = (len(scenes) + chunk - 1) // chunk if scenes else 0
    if total == 0:
        log_status("S1: no scenes to process.")
        return outs

    for i in range(0, len(scenes), chunk):
        sub = scenes[i : i + chunk]
        batch_idx = (i // chunk) + 1
        log_status(
            f"S1 batch {batch_idx}/{total}: submitting {len(sub)} scenes to Gemini"
        )
        inlined: list[types.InlinedRequestDict] = build_s1_inline_requests(sub)
        job = client.batches.create(
            model="gemini-2.5-flash",
            src=types.BatchJobSourceDict(inlined_requests=inlined),
            config=types.CreateBatchJobConfigDict(display_name=f"s1-{i // chunk:03d}"),
        )

        assert job.name is not None
        done = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        last_state = None
        while True:
            bj = client.batches.get(name=job.name)
            assert bj.state is not None
            state_name = bj.state.name
            if state_name != last_state:
                log_status(f"S1 batch {batch_idx}/{total}: {state_name.lower()}")
                last_state = state_name
            if state_name in done:
                if state_name != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError(f"S1 batch failed: {state_name} {bj.error}")
                break
            time.sleep(3)

        assert bj.dest is not None
        assert bj.dest.inlined_responses is not None
        parsed = 0
        for idx, resp in enumerate(bj.dest.inlined_responses, start=1):
            if resp.error:
                log_status(
                    f"S1 batch {batch_idx}/{total}: inline {idx} error -> {resp.error}"
                )
                continue
            if not resp.response:
                log_status(
                    f"S1 batch {batch_idx}/{total}: inline {idx} has no response field"
                )
                continue

            payload = getattr(resp.response, "parsed", None)
            if payload is not None:
                api_payload = parse_model(IntraScenePayloadAPI, payload)
                outs.append(api_payload.to_internal())
                parsed += 1
                continue

            text = getattr(resp.response, "text", None)
            if text:
                api_payload = parse_model(IntraScenePayloadAPI, text)
                outs.append(api_payload.to_internal())
                parsed += 1
                continue

            log_status(
                f"S1 batch {batch_idx}/{total}: inline {idx} missing text/parsed payload"
            )
        log_status(
            f"S1 batch {batch_idx}/{total}: parsed {parsed} responses "
            f"(running total {len(outs)})"
        )

    return outs


def s1_flatten_validate(
    payloads: list[IntraScenePayload],
) -> tuple[list[ToMClue], list[ActClue], list[str]]:
    toms: list[ToMClue] = []
    acts: list[ActClue] = []
    logs: list[str] = []
    running: dict[int, int] = defaultdict(int)

    for sp in payloads:
        combined = [*sp.tom_clues, *sp.act_clues]
        scene_id: int | None = None
        for it in combined:
            if getattr(it, "scene", None) is not None:
                scene_id = int(getattr(it, "scene"))
                break
        if scene_id is None:
            logs.append("warn: missing scene id in S1 items")
            continue

        for it in combined:
            m = it.model_dump()
            m["scene"] = int(m.get("scene", scene_id))

            # id 없으면 생성
            if not m.get("id"):
                running[m["scene"]] += 1
                prefix = "tom" if m.get("modality") == "tom" else "act"
                m["id"] = f"{prefix}_s1_{m['scene']:03d}_{running[m['scene']]:04d}"

            # pair 보정(없으면 actors/targets로; ToM은 없으면 드랍)
            if not m.get("pair"):
                if m.get("modality") == "act":
                    actors = m.get("actors") or []
                    targets = m.get("targets") or []
                    if actors and targets:
                        m["pair"] = norm_pair(actors[0], targets[0])
                    else:
                        logs.append(f"drop act(no pair/actors/targets): {m.get('id')}")
                        continue
                else:
                    logs.append(f"drop tom(no pair): {m.get('id')}")
                    continue

            try:
                if m.get("modality") == "act":
                    acts.append(ActClue.model_validate(m))
                elif m.get("modality") == "tom":
                    toms.append(ToMClue.model_validate(m))
                else:
                    logs.append(f"drop item(unknown modality): {m.get('id')}")
            except ValidationError as e:
                logs.append(f"drop item(validation): {m.get('id')} -> {e.errors()}")

    return toms, acts, logs
