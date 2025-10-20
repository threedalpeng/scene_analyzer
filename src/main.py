import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from google import genai

from inter_scene_analyze import DyadBag, build_bags, run_s3_batch, validate_llm_decision
from intra_scene_analyze import run_s1_batch, s1_flatten_validate
from merge_entity import alias_name, alias_pair, build_alias_map, run_s2_alias
from schema import ActClue, AliasGroups, DyadFinal, ToMClue
from utils import act_score, jsonl_write, log_status


def explode_directed_acts(acts: list[ActClue]) -> list[ActClue]:
    """
    1→N / N→1 act를 1→1 유향 act들로 분해.
    - id에 "__src->dst" suffix 추가
    - pair는 (src, dst) 정렬 무향키이지만, actors/targets로 유향 유지
    - 자기지시(src==dst)는 드롭
    """
    out: list[ActClue] = []
    for a in acts:
        actors = a.actors or [a.pair[0]]
        targets = a.targets or [a.pair[1]]
        for src in actors:
            for dst in targets:
                if src.lower() == dst.lower():
                    continue
                out.append(
                    a.model_copy(
                        update={
                            "id": f"{a.id}__{src}->{dst}",
                            "actors": [src],
                            "targets": [dst],
                            "pair": tuple(sorted((src, dst))),
                        }
                    )
                )
    return out


def bundle_same_scene(acts_dir: list[ActClue]) -> list[ActClue]:
    """
    같은 scene에서 (src,dst,stance,subtype) 동일한 act들을 하나로 대표 선택.
    - 선택 기준: _act_strength_score 내림차순
    - 대표만 남겨 중복 노이즈 제거 (증거/라벨은 대표에 보존)
    """
    buckets: dict[tuple[int, str, str, str, str], list[ActClue]] = defaultdict(list)

    for a in acts_dir:
        src = a.actors[0] if a.actors else a.pair[0]
        dst = a.targets[0] if a.targets else a.pair[1]
        buckets[(a.scene, src, dst, a.stance, a.subtype)].append(a)

    reps: list[ActClue] = []
    for items in buckets.values():
        items.sort(key=act_score, reverse=True)
        reps.append(items[0])
    return reps


def write_graph_light(
    out_dir: Path, toms: list[ToMClue], acts_all: list[ActClue]
) -> None:
    """
    graph_light/
      - nodes_person.jsonl
      - nodes_scene.jsonl
      - edges_tom.jsonl
      - edges_act.jsonl (src/dst 필드 포함)
      - edges_consequence.jsonl (consequence_refs로 scene 간 인과 에지)
    """
    gdir = out_dir / "graph_light"
    gdir.mkdir(parents=True, exist_ok=True)

    persons: set[str] = set()
    scenes: set[int] = set()

    for t in toms:
        persons.update(t.pair)
        scenes.add(t.scene)

    for a in acts_all:
        src = a.actors[0] if a.actors else a.pair[0]
        dst = a.targets[0] if a.targets else a.pair[1]
        persons.update([src, dst])
        scenes.add(a.scene)

    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write("\n")

    _write_jsonl(
        gdir / "nodes_person.jsonl",
        [{"id": p, "name": p} for p in sorted(persons)],
    )
    _write_jsonl(
        gdir / "nodes_scene.jsonl",
        [{"id": s} for s in sorted(scenes)],
    )
    _write_jsonl(
        gdir / "edges_tom.jsonl",
        [t.model_dump() for t in toms],
    )
    _write_jsonl(
        gdir / "edges_act.jsonl",
        [
            {
                **a.model_dump(),
                "src": (a.actors[0] if a.actors else a.pair[0]),
                "dst": (a.targets[0] if a.targets else a.pair[1]),
            }
            for a in acts_all
        ],
    )

    cons_edges: list[dict] = []
    for a in acts_all:
        for r in a.axes.consequence_refs or []:
            cons_edges.append(
                {
                    "src_scene": a.scene,
                    "dst_scene": int(r),
                    "via": [a.id],
                }
            )
    _write_jsonl(gdir / "edges_consequence.jsonl", cons_edges)


def _s3_cache_path(out_dir: Path) -> Path:
    return out_dir / "s3_results.jsonl"


def load_s3_cache(out_dir: Path) -> dict[tuple[str, str], tuple[DyadFinal, set[str]]]:
    cache: dict[tuple[str, str], tuple[DyadFinal, set[str]]] = {}
    path = _s3_cache_path(out_dir)
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pair_raw = obj.get("pair", [])
                if not isinstance(pair_raw, list) or len(pair_raw) != 2:
                    continue
                pair = (str(pair_raw[0]), str(pair_raw[1]))
                final = DyadFinal.model_validate(obj.get("final", {}))
                turning_ids = set(str(x) for x in obj.get("turning_ids", []))
            except Exception:
                continue
            cache[pair] = (final, turning_ids)
    return cache


def append_s3_cache(
    out_dir: Path, pair: tuple[str, str], final: DyadFinal, turning_ids: Iterable[str]
) -> None:
    path = _s3_cache_path(out_dir)
    record = {
        "pair": list(pair),
        "final": final.model_dump(),
        "turning_ids": sorted(set(turning_ids)),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def load_cached_s1(
    out_dir: Path,
) -> tuple[list[ToMClue], list[ActClue], list[str]] | None:
    path = out_dir / "signals_s1.jsonl"
    if not path.exists():
        return None

    toms: list[ToMClue] = []
    acts: list[ActClue] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                modality = obj.get("modality")
                if modality == "tom":
                    toms.append(ToMClue.model_validate(obj))
                elif modality == "act":
                    acts.append(ActClue.model_validate(obj))
            except Exception:
                continue

    log_path = out_dir / "s1_postprocess_log.txt"
    logs: list[str] = []
    if log_path.exists():
        logs = log_path.read_text(encoding="utf-8").splitlines()

    return toms, acts, logs


def load_cached_aliases(out_dir: Path) -> AliasGroups | None:
    path = out_dir / "aliases.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AliasGroups.model_validate(data)
    except Exception:
        return None


def run_pipeline(input_path: Path, out_dir: Path, batch_chunk: int = 10) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[dict] = json.loads(input_path.read_text(encoding="utf-8"))

    client = genai.Client()

    # S1
    log_status(f"Pipeline start: {len(scenes)} scenes -> {out_dir}")
    cached_s1 = load_cached_s1(out_dir)
    if cached_s1:
        toms_raw, acts_raw, s1_logs = cached_s1
        log_status(
            f"S1: loaded cached signals ({len(toms_raw)} tom clues & "
            f"{len(acts_raw)} act clues)"
        )
    else:
        log_status(f"S1: extracting intra-scene signals in chunks of {batch_chunk}")
        s1_payloads = run_s1_batch(client, scenes, chunk=batch_chunk)
        toms_raw, acts_raw, s1_logs = s1_flatten_validate(s1_payloads)
        log_status(
            f"S1: collected {len(toms_raw)} tom clues & {len(acts_raw)} act clues "
            f"({len(s1_logs)} post-process notes)"
        )
    jsonl_write(
        out_dir / "signals_s1.jsonl",
        [t.model_dump() for t in toms_raw] + [a.model_dump() for a in acts_raw],
    )

    # S2: alias
    aliases = load_cached_aliases(out_dir)
    if aliases:
        log_status(f"S2: loaded cached alias groups ({len(aliases.groups)} groups)")
    else:
        log_status("S2: preparing alias candidates")
        appearances: dict[str, list[int]] = defaultdict(list)
        for t in toms_raw:
            for n in t.pair:
                appearances[n].append(t.scene)
        for a in acts_raw:
            for n in set([*a.pair, *a.actors, *a.targets]):
                appearances[n].append(a.scene)

        unique_names = sorted(appearances.keys())
        log_status(f"S2: resolving {len(unique_names)} unique names")
        aliases = run_s2_alias(
            client, unique_names, {k: sorted(set(v)) for k, v in appearances.items()}
        )
        (out_dir / "aliases.json").write_text(
            json.dumps(aliases.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        alias_log = [f"{g.canonical} <- {', '.join(g.aliases)}" for g in aliases.groups]
        (out_dir / "alias_log.txt").write_text("\n".join(alias_log), "utf-8")

    if aliases is None:
        raise RuntimeError("Failed to obtain alias groups")

    amap = build_alias_map(aliases)
    log_status(f"S2: canonicalized {len(amap)} alias mappings")

    toms = [t.model_copy(update={"pair": alias_pair(t.pair, amap)}) for t in toms_raw]
    acts_alias = [
        a.model_copy(
            update={
                "pair": alias_pair(a.pair, amap),
                "actors": [alias_name(x, amap) for x in (a.actors or [])],
                "targets": [alias_name(x, amap) for x in (a.targets or [])],
            }
        )
        for a in acts_raw
    ]

    # Directed & bundling
    log_status("Post: exploding directed acts and selecting representatives")
    acts_dir = explode_directed_acts(acts_alias)
    acts_rep = bundle_same_scene(acts_dir)
    log_status(f"Post: {len(acts_dir)} directed acts, {len(acts_rep)} representatives")

    # Graph
    log_status("Graph: writing lightweight graph artifacts")
    write_graph_light(out_dir, toms, acts_dir)

    # S3: dyad bags
    log_status("S3: building dyad bags")
    bags = build_bags(toms, acts_rep, acts_dir)
    total_pairs = len(bags)
    log_status(f"S3: adjudicating {total_pairs} dyads")

    # LLM adjudication + validate (with caching)
    cache_data = load_s3_cache(out_dir)
    if cache_data:
        log_status(f"S3: loaded cache for {len(cache_data)} pairs")

    event_role_map: dict[str, str] = {}
    finals_dict: dict[tuple[str, str], DyadFinal] = {}

    for pair, (cached_final, cached_turns) in cache_data.items():
        finals_dict[pair] = cached_final
        bag = bags.get(pair)
        if not bag:
            continue
        for a in bag.acts_rep:
            event_role_map[a.id] = "turning" if a.id in cached_turns else "supporting"
        for t in bag.toms:
            event_role_map[t.id] = "supporting"

    pending: list[tuple[int, tuple[str, str], DyadBag]] = []
    for idx, (pair, bag) in enumerate(sorted(bags.items()), start=1):
        cached = cache_data.get(pair)
        if cached:
            log_status(f"S3 {idx}/{total_pairs}: {pair[0]} <-> {pair[1]} (cached)")
            continue
        log_status(f"S3 {idx}/{total_pairs}: {pair[0]} <-> {pair[1]} queued")
        pending.append((idx, pair, bag))

    batched_results = run_s3_batch(
        client, [(pair, bag) for _, pair, bag in pending], chunk=batch_chunk
    )

    for idx, pair, bag in pending:
        out = batched_results.get(pair)
        if out:
            try:
                out = validate_llm_decision(bag, out)
                turning_ids = {e.representative_act_id for e in out.turning_timeline}
                for a in bag.acts_rep:
                    event_role_map[a.id] = (
                        "turning" if a.id in turning_ids else "supporting"
                    )
                for t in bag.toms:
                    event_role_map[t.id] = "supporting"
                final = DyadFinal(
                    pair=pair,
                    final_relation=out.final_relation,
                    turning_timeline=out.turning_timeline,
                    state_timeline_ref=None,
                )
                finals_dict[pair] = final
                cache_data[pair] = (final, turning_ids)
                append_s3_cache(out_dir, pair, final, turning_ids)
                log_status(
                    f"S3 {idx}/{total_pairs}: {pair[0]} <-> {pair[1]} adjudicated"
                )
                log_status(
                    f"S3 {idx}/{total_pairs}: assigned roles to {len(bag.acts_rep)} acts"
                )
                continue
            except Exception as err:
                log_status(
                    f"S3 {idx}/{total_pairs}: validation error -> {err}; defaulting to neutral"
                )

        # Fallback path when no valid result
        for a in bag.acts_rep:
            event_role_map[a.id] = "supporting"
        for t in bag.toms:
            event_role_map[t.id] = "supporting"
        final = DyadFinal(
            pair=pair,
            final_relation="neutral",
            turning_timeline=[],
            state_timeline_ref=None,
        )
        finals_dict[pair] = final
        cache_data[pair] = (final, set())
        append_s3_cache(out_dir, pair, final, set())
        log_status(
            f"S3 {idx}/{total_pairs}: {pair[0]} <-> {pair[1]} defaulted to neutral/supporting"
        )

    finals = [finals_dict[p] for p in sorted(finals_dict)]

    # signals_s3.jsonl — ToM 전량 + 분해된 전체 유향 act 전량
    log_status("Outputs: assembling signals_s3.jsonl")
    rows = []
    for t in toms:
        r = t.model_dump()
        r["event_role"] = event_role_map.get(t.id, "supporting")
        rows.append(r)
    rep_ids = set(event_role_map.keys())
    for a in acts_dir:
        r = a.model_dump()
        r["event_role"] = (
            "turning"
            if (a.id in rep_ids and event_role_map.get(a.id) == "turning")
            else "supporting"
        )
        rows.append(r)
    jsonl_write(out_dir / "signals_s3.jsonl", rows)

    # final_report.json
    log_status("Outputs: writing final report")
    (out_dir / "final_report.json").write_text(
        json.dumps(
            {"pairs": [f.model_dump() for f in finals]}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    # logs
    (out_dir / "s1_postprocess_log.txt").write_text("\n".join(s1_logs), "utf-8")
    log_status("Pipeline complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./outputs"))
    ap.add_argument("--batch-chunk", type=int, default=50)
    args = ap.parse_args()
    run_pipeline(args.input, args.out, batch_chunk=args.batch_chunk)


if __name__ == "__main__":
    main()
