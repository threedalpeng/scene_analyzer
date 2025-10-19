import json
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from schema import ActClue, LLMAdjudication, LLMEventRole, ToMClue, TurningEntry
from utils import act_score, log_status, parse_model

S3_SYSTEM_PROMPT = """
You adjudicate relationship state across scenes for a dyad.
Use ONLY the provided packs and return strict JSON per schema.

A turning MUST reference a representative act from the input (with stance and axes).
If no such act exists, do NOT mark a turning; downgrade to supporting and log "no-act-anchor".

Decide in this order:
1) Turning detection:
   turning if (stance reversal OR durability upgrade OR implicit_turn) AND (threshold met).
   - implicit_turn: Intends/Desires followed within ≤2 scenes by a consistent act for this dyad.
   - threshold: stakes="major" OR (salience="high" & durability="persistent") OR has consequence_refs.
2) De-duplication:
   - window-one: same-direction turns in close succession -> keep only the latest.
   - near-merge: merge similar turns within ±2 scenes; combine quotes/causes.
3) Pre/Post state:
   - pre_state: dyad state just before this scene (neutral by default or last post_state).
   - post_state: taken from the representative act's stance and durability.
4) Final relation for the dyad: cooperation|hostility|mixed|neutral.

ID rules:
- Preserve every id exactly as given in the input data. Do not invent or reformat ids.
- event_roles[].id must match an act or tom id from the packs.
- turning_timeline[].representative_act_id must be one of the supplied representative act ids.
""".strip()


def s3_user_payload(
    dossier_lines: list[str], tom_lines: list[str], causal_json: dict, stats_json: dict
) -> str:
    return f"""DYAD_DOSSIER:
{chr(10).join(dossier_lines)}

TOM_THREADS:
{chr(10).join(tom_lines)}

CAUSAL_CHAIN:
{json.dumps(causal_json, ensure_ascii=False, indent=2)}

STATS:
{json.dumps(stats_json, ensure_ascii=False, indent=2)}
""".strip()


@dataclass
class DyadBag:
    pair: tuple[str, str]
    acts_rep: list[ActClue] = field(default_factory=list)  # 번들 대표(판정용)
    acts_all: list[ActClue] = field(default_factory=list)  # 전체 유향(역할부착/그래프)
    toms: list[ToMClue] = field(default_factory=list)


def build_bags(
    toms: list[ToMClue], acts_rep: list[ActClue], acts_all: list[ActClue]
) -> dict[tuple[str, str], DyadBag]:
    bags: dict[tuple[str, str], DyadBag] = {}

    def get(p: tuple[str, str]) -> DyadBag:
        if p not in bags:
            bags[p] = DyadBag(pair=p)
        return bags[p]

    for a in acts_rep:
        get(a.pair).acts_rep.append(a)
    for a in acts_all:
        get(a.pair).acts_all.append(a)
    for t in toms:
        get(t.pair).toms.append(t)
    for b in bags.values():
        b.acts_rep.sort(key=lambda x: x.scene)
        b.acts_all.sort(key=lambda x: x.scene)
        b.toms.sort(key=lambda x: x.scene)
    return bags


def _sample_dossier(acts_rep: list[ActClue]) -> list[ActClue]:
    if not acts_rep:
        return []
    items = sorted(acts_rep, key=act_score, reverse=True)
    rep = items[0]
    supports = [a for a in items[1:] if a.stance != rep.stance][:1]
    if len(supports) < 1 and len(items) > 1:
        supports = [items[1]]
    return [rep, *supports]


def _packs_for_pair(bag: DyadBag) -> tuple[list[str], list[str], dict, dict]:
    dossier = _sample_dossier(bag.acts_rep)
    dossier_lines = [json.dumps(a.model_dump(), ensure_ascii=False) for a in dossier]
    tom_lines = [json.dumps(t.model_dump(), ensure_ascii=False) for t in bag.toms[-6:]]

    causal_edges = []
    for a in bag.acts_all:
        for r in a.axes.consequence_refs or []:
            causal_edges.append(
                {"src_scene": a.scene, "dst_scene": int(r), "via": [a.id]}
            )
    causal_json = {"edges": causal_edges[:50]}

    from collections import Counter

    stance = Counter(a.stance for a in bag.acts_all)
    sal = Counter(a.axes.salience for a in bag.acts_all)
    stk = Counter(a.axes.stakes for a in bag.acts_all)
    dur = Counter(a.axes.durability for a in bag.acts_all)
    coerced = sum(1 for a in bag.acts_all if a.axes.volition == "coerced")
    stats_json = {
        "acts_count": len(bag.acts_all),
        "stance_freq": dict(stance),
        "salience_freq": dict(sal),
        "stakes_freq": dict(stk),
        "durability_freq": dict(dur),
        "coerced_ratio": (coerced / len(bag.acts_all)) if bag.acts_all else 0.0,
    }
    return dossier_lines, tom_lines, causal_json, stats_json


def llm_adjudicate_pair(client: genai.Client, bag: DyadBag) -> LLMAdjudication:
    dossier, toms, causal, stats = _packs_for_pair(bag)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [{"text": s3_user_payload(dossier, toms, causal, stats)}],
            },
        ],
        config=types.GenerateContentConfigDict(
            system_instruction=S3_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMAdjudication,
        ),
    )
    raw = getattr(resp, "parsed", None)
    if raw is not None:
        return parse_model(LLMAdjudication, raw)
    if resp.text is not None:
        return parse_model(LLMAdjudication, resp.text)
    raise RuntimeError("LLM adjudication returned no usable payload.")


def build_s3_inline_requests(
    items: list[tuple[tuple[str, str], DyadBag]]
) -> tuple[list[types.InlinedRequestDict], list[tuple[str, str]]]:
    requests: list[types.InlinedRequestDict] = []
    order: list[tuple[str, str]] = []
    for pair, bag in items:
        dossier, toms, causal, stats = _packs_for_pair(bag)
        requests.append(
            types.InlinedRequestDict(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": s3_user_payload(
                                    dossier, toms, causal, stats
                                )
                            }
                        ],
                    }
                ],
                config=types.GenerateContentConfigDict(
                    response_schema=LLMAdjudication,
                    response_mime_type="application/json",
                ),
            )
        )
        order.append(pair)
    return requests, order


def run_s3_batch(
    client: genai.Client,
    items: list[tuple[tuple[str, str], DyadBag]],
    chunk: int = 10,
) -> dict[tuple[str, str], LLMAdjudication]:
    results: dict[tuple[str, str], LLMAdjudication] = {}
    if not items:
        return results

    done_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    for i in range(0, len(items), chunk):
        sub = items[i : i + chunk]
        batch_idx = (i // chunk) + 1
        reqs, order = build_s3_inline_requests(sub)
        log_status(
            f"S3 batch {batch_idx}: submitting {len(sub)} dyads to Gemini batches"
        )
        job = client.batches.create(
            model="gemini-2.5-flash",
            src=types.BatchJobSourceDict(inlined_requests=reqs),
            config=types.CreateBatchJobConfigDict(display_name=f"s3-{batch_idx:03d}"),
        )
        assert job.name is not None

        last_state = None
        while True:
            bj = client.batches.get(name=job.name)
            assert bj.state is not None
            state = bj.state.name
            if state != last_state:
                log_status(f"S3 batch {batch_idx}: {state.lower()}")
                last_state = state
            if state in done_states:
                if state != "JOB_STATE_SUCCEEDED":
                    log_status(f"S3 batch {batch_idx}: failed -> {bj.error}")
                break
            time.sleep(3)

        if not bj.dest or not bj.dest.inlined_responses:
            log_status(f"S3 batch {batch_idx}: no responses")
            continue

        for pair, resp in zip(order, bj.dest.inlined_responses, strict=False):
            if resp.error:
                log_status(
                    f"S3 batch {batch_idx}: {pair[0]}<->{pair[1]} error {resp.error}"
                )
                continue
            if not resp.response:
                log_status(
                    f"S3 batch {batch_idx}: {pair[0]}<->{pair[1]} missing response"
                )
                continue

            payload = getattr(resp.response, "parsed", None)
            try:
                if payload is not None:
                    results[pair] = parse_model(LLMAdjudication, payload)
                    continue
                text = getattr(resp.response, "text", None)
                if text:
                    results[pair] = parse_model(LLMAdjudication, text)
                    continue
            except Exception as err:
                log_status(
                    f"S3 batch {batch_idx}: parse error for {pair[0]}<->{pair[1]} -> {err}"
                )
                continue

            log_status(
                f"S3 batch {batch_idx}: {pair[0]}<->{pair[1]} missing payload content"
            )

    return results


TURN_TYPES = {
    "betrayal",
    "reconciliation",
    "coerced_hostility",
    "aligned_mission",
    "revealed_identity",
    "implicit_turn",
}


def validate_llm_decision(bag: DyadBag, out: LLMAdjudication) -> LLMAdjudication:
    valid_rep_ids = {a.id for a in bag.acts_rep}

    # turning_timeline: 대표 act 없는 건 제거, type 보정(필요 시)
    kept: list[TurningEntry] = []
    last: TurningEntry | None = None
    for e in out.turning_timeline:
        if e.representative_act_id not in valid_rep_ids:
            continue
        if e.type not in TURN_TYPES:
            e.type = (
                "reconciliation" if e.post_state.stance == "cooperation" else "betrayal"
            )
        if last and abs(e.scene - last.scene) <= 2 and e.type == last.type:
            last.summary = f"{last.summary} | {e.summary}"
            last.representative_act_id = e.representative_act_id
            last.caused_by_tom_ids = sorted(
                set(last.caused_by_tom_ids + e.caused_by_tom_ids)
            )
            last.post_state = e.post_state
        else:
            kept.append(e)
            last = kept[-1]
    out.turning_timeline = kept

    # event_roles: window-one 반영(타임라인에 남은 대표만 turning)
    keep_turn = {e.representative_act_id for e in out.turning_timeline}
    fixed: list[LLMEventRole] = []
    for er in out.event_roles:
        if er.event_role == "turning" and er.id not in keep_turn:
            fixed.append(
                LLMEventRole(scene=er.scene, id=er.id, event_role="supporting")
            )
        else:
            fixed.append(er)
    out.event_roles = fixed

    # final_relation 도메인 체크
    if out.final_relation not in ("cooperation", "hostility", "mixed", "neutral"):
        if out.turning_timeline:
            last_post = out.turning_timeline[-1].post_state.stance
            pols = {t.post_state.stance for t in out.turning_timeline}
            out.final_relation = (
                "mixed"
                if ("cooperation" in pols and "hostility" in pols)
                else last_post
            )
        else:
            out.final_relation = "neutral"
    return out
