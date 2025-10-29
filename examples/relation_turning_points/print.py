#!/usr/bin/env python3
"""
Relationship analysis report generator.

Builds a layered narrative report for a specific character dyad using
extracted Act/ToM clues and the adjudicated turning timeline.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from clues.act import ActClue
from clues.tom import ToMClue
from processors.synthesis import _select_acts_for_dossier, _select_toms_for_dossier
from schema import FinalRelation, TurningEntry

from framework.utils import norm_pair


@dataclass
class TimelineEvent:
    segment: int
    event_type: str  # "act" or "tom"
    clue: ActClue | ToMClue
    is_turning: bool = False
    is_supporting: bool = False


@dataclass
class DyadInfo:
    turning_timeline: list[TurningEntry]
    final_relation: FinalRelation
    event_roles: dict[str, str]


def load_clues(output_dir: Path) -> tuple[list[ActClue], list[ToMClue]]:
    acts: list[ActClue] = []
    toms: list[ToMClue] = []

    act_path = output_dir / "act_clues.jsonl"
    if act_path.exists():
        with act_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    acts.append(ActClue.model_validate(json.loads(line)))

    tom_path = output_dir / "tom_clues.jsonl"
    if tom_path.exists():
        with tom_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    toms.append(ToMClue.model_validate(json.loads(line)))

    return acts, toms


def load_dyad_info(output_dir: Path, char1: str, char2: str) -> DyadInfo | None:
    result_path = output_dir / "dyad_results.jsonl"
    if not result_path.exists():
        return None

    target_pair = norm_pair(char1, char2)

    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            result_pair = norm_pair(record["char1"], record["char2"])
            if result_pair != target_pair:
                continue

            adjudication = record["adjudication"]
            event_roles = {
                item["id"]: item["event_role"]
                for item in adjudication.get("event_roles", [])
            }
            return DyadInfo(
                turning_timeline=[
                    TurningEntry.model_validate(raw)
                    for raw in adjudication.get("turning_timeline", [])
                ],
                final_relation=FinalRelation.model_validate(
                    adjudication.get("final_relation", {"valence": "neutral"})
                ),
                event_roles=event_roles,
            )
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _print_header(char1: str, char2: str) -> None:
    pair = norm_pair(char1, char2)
    title = f"RELATIONSHIP ANALYSIS REPORT: {pair[0]} ↔ {pair[1]}"
    line = "=" * 100
    print(line)
    print(title.center(100))
    print(line)
    print()


def _print_section(title: str) -> None:
    print("┌" + "─" * 98 + "┐")
    print("│" + f" {title} ".center(98) + "│")
    print("└" + "─" * 98 + "┘")
    print()


def _segment_span(segments: Sequence[int]) -> str:
    if not segments:
        return "-"
    lo, hi = segments[0], segments[-1]
    return f"{lo}-{hi}" if lo != hi else f"{lo}"


def _percent(part: int, total: int) -> float:
    return (part / total) * 100 if total else 0.0


def _describe_valence(ratio: float) -> str:
    if ratio >= 0.66:
        return "positive"
    if ratio >= 0.33:
        return "mixed"
    return "negative"


def _role_marker(clue_id: str, roles: dict[str, str]) -> str:
    role = roles.get(clue_id)
    if role == "turning":
        return "⭐ TURNING"
    if role == "supporting":
        return "• SUPPORTING"
    return ""


def _build_timeline(
    acts: list[ActClue], toms: list[ToMClue], roles: dict[str, str] | None
) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    role_map = roles or {}

    for act in acts:
        events.append(
            TimelineEvent(
                segment=act.segment,
                event_type="act",
                clue=act,
                is_turning=role_map.get(act.id) == "turning",
                is_supporting=role_map.get(act.id) == "supporting",
            )
        )
    for tom in toms:
        events.append(
            TimelineEvent(
                segment=tom.segment,
                event_type="tom",
                clue=tom,
                is_turning=role_map.get(tom.id) == "turning",
                is_supporting=role_map.get(tom.id) == "supporting",
            )
        )

    events.sort(key=lambda event: (event.segment, event.event_type))
    return events


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def _print_executive_summary(
    acts: list[ActClue],
    toms: list[ToMClue],
    selected_acts: list[ActClue],
    selected_toms: list[ToMClue],
    dyad_info: DyadInfo | None,
) -> None:
    _print_section("EXECUTIVE SUMMARY")

    segments = sorted({*{a.segment for a in acts}, *{t.segment for t in toms}})
    acts_segments = sorted({a.segment for a in acts})
    tom_segments = sorted({t.segment for t in toms})
    turning_entries = dyad_info.turning_timeline if dyad_info else []

    # Relationship trajectory
    if turning_entries:
        pre_state = turning_entries[0].pre_state
        initial_label = f' "{pre_state.label}"' if pre_state.label else ""
        initial_line = (
            f"  Initial State: {pre_state.valence} / {pre_state.durability}"
            f"{initial_label} (from turning analysis)"
        )
        first_turn = turning_entries[0]
        first_turn_line = f"  #1 Segment {first_turn.segment} → {first_turn.category}"
    else:
        if toms:
            earliest_segments = sorted({tom.segment for tom in toms})[:3]
            clue = next(
                (tom for tom in sorted(toms, key=lambda t: t.segment) if tom.segment in earliest_segments),
                None,
            )
            if clue:
                initial_line = (
                    f'  Initial State: inferred from ToM — {clue.thinker} about '
                    f"{clue.target}: \"{clue.claim.strip()}\""
                )
            else:
                initial_line = "  Initial State: insufficient evidence"
        else:
            initial_line = "  Initial State: insufficient evidence"
        first_turn_line = "  Turning Points: none detected"

    final_rel = dyad_info.final_relation if dyad_info else FinalRelation(valence="neutral")
    final_label = f' "{final_rel.label}"' if final_rel.label else ""
    final_line = f"  Final State: {final_rel.valence}{final_label}"

    print("Relationship Trajectory:")
    print(initial_line)
    print(first_turn_line)
    print(final_line)
    print()

    # Key statistics
    total_acts = len(acts)
    total_toms = len(toms)
    total_turns = len(turning_entries)
    positive_acts = sum(1 for act in acts if act.valence == "positive")
    early_segments = acts_segments[:2] if acts_segments else tom_segments[:2]
    late_segments = acts_segments[-2:] if acts_segments else tom_segments[-2:]

    early_acts = [act for act in acts if act.segment in early_segments]
    late_acts = [act for act in acts if act.segment in late_segments]
    early_ratio = _percent(sum(1 for a in early_acts if a.valence == "positive"), len(early_acts)) / 100 if early_acts else 0.0
    late_ratio = _percent(sum(1 for a in late_acts if a.valence == "positive"), len(late_acts)) / 100 if late_acts else 0.0

    turning_segment = turning_entries[0].segment if turning_entries else (acts_segments[0] if acts_segments else (tom_segments[0] if tom_segments else "-"))

    print("Key Statistics:")
    print(f"  • Total Interactions: {total_acts} acts, {total_toms} ToMs across segments {_segment_span(segments)}")
    print(f"  • Turning Points: {total_turns if total_turns else 'none'}")
    print(f"  • Valence Shift: {early_ratio*100:.1f}% → {late_ratio*100:.1f}% positive")
    print(f"  • Critical Segment: {turning_segment}")
    print()

    # Narrative arc
    print("Narrative Arc:")
    if segments:
        early_span = _segment_span(segments[: max(1, min(3, len(segments)))])
        if turning_entries:
            crisis = ", ".join(str(turn.segment) for turn in turning_entries)
        else:
            crisis = "n/a"
        late_span = _segment_span(segments[-max(1, min(3, len(segments))):])
        print(f"  {early_span}: Early context from ToM clues and initial acts")
        print(f"  Segment(s) {crisis}: Turning point sequence")
        print(f"  {late_span}: Relationship aftermath and final state")
    else:
        print("  (No timeline data available)")
    print()


def _print_turning_points(
    turning_timeline: list[TurningEntry],
    act_lookup: dict[str, ActClue],
    tom_lookup: dict[str, ToMClue],
) -> None:
    _print_section("TURNING POINTS: Detailed Analysis")

    if not turning_timeline:
        print("  (No turning points detected)")
        print()
        return

    for idx, turn in enumerate(turning_timeline, start=1):
        print(f"#{idx} SEGMENT {turn.segment} — {turn.category}")
        print("═" * 100)
        print("  Summary:")
        print(f"    {turn.summary.strip()}")
        print()

        print("  ┌─ State Transition " + "─" * 64 + "┐")
        before_label = turn.pre_state.label or ""
        after_label = turn.post_state.label or ""
        print(
            f"  │ BEFORE: {turn.pre_state.valence:>8} / {turn.pre_state.durability:<10}"
            f"{('  ' + before_label) if before_label else ''}".ljust(73)
            + "│"
        )
        print(
            f"  │ AFTER : {turn.post_state.valence:>8} / {turn.post_state.durability:<10}"
            f"{('  ' + after_label) if after_label else ''}".ljust(73)
            + "│"
        )
        print("  └" + "─" * 96 + "┘")
        print()

        act = act_lookup.get(turn.representative_act_id)
        if act:
            print(f"  Representative Action [{act.id}]:")
            print(
                f"    ⚔ {act.source} → {act.target}: {act.pattern} "
                f"(valence={act.valence})"
            )
            print(
                f"    Stakes={act.axes.stakes}, Salience={act.axes.salience}, "
                f"Durability={act.axes.durability}"
            )
            print(f'    Evidence: "{act.evidence.strip()}"')
        else:
            print(f"  Representative Action [{turn.representative_act_id}]: (missing)")
        print()

        if turn.caused_by_tom_ids:
            print("  Psychological Causes (ToM):")
            for tom_id in turn.caused_by_tom_ids:
                tom = tom_lookup.get(tom_id)
                if not tom:
                    print(f"    • [{tom_id}] (missing)")
                    continue
                print(
                    f"    • [{tom.id}] {tom.kind} — "
                    f"{tom.thinker} about {tom.target}"
                )
                print(f'      Claim: "{tom.claim.strip()}"')
                print(f'      Evidence: "{tom.evidence.strip()}"')
            print()

        print(f"  Pattern: {turn.pattern}")
        print()
        print("─" * 100)
        print()


def _print_selected_clues(
    acts: list[ActClue],
    toms: list[ToMClue],
    selected_acts: list[ActClue],
    selected_toms: list[ToMClue],
    roles: dict[str, str],
) -> None:
    _print_section("SELECTED CLUES (Provided to LLM)")

    print("Selection Strategy:")
    print(
        "  Acts:  major stakes + first 2 segments + last 2 segments + score-ranked fill (max 30)"
    )
    print(
        "  ToMs:  IntendsTo + act segments ±1 + first 3 ToM segments + last 2 segments + priority (max 20)"
    )
    print()

    selected_act_ids = {act.id for act in selected_acts}
    selected_tom_ids = {tom.id for tom in selected_toms}
    act_segments = sorted({act.segment for act in acts})
    tom_segments = sorted({tom.segment for tom in toms})

    early_act_segments = set(act_segments[:2])
    late_act_segments = set(act_segments[-2:])
    major_segments = {act.segment for act in selected_acts if act.axes.stakes == "major"}

    print(
        f"ACT CLUES ({len(selected_acts)} selected out of {len(acts)} total):"
    )
    grouped_acts = defaultdict(list)
    for act in selected_acts:
        grouped_acts[act.segment].append(act)

    for segment in sorted(grouped_acts.keys()):
        items = grouped_acts[segment]
        labels: list[str] = []
        if segment in early_act_segments:
            labels.append("early segment")
        if segment in late_act_segments:
            labels.append("late segment")
        if segment in major_segments:
            labels.append("MAJOR stakes")
        prefix = "⭐" if segment in major_segments else ("✓" if labels else "•")
        label_text = f" - {', '.join(labels)}" if labels else ""
        print(
            f"  {prefix} Segment {segment:03d} [{len(items)} act{'s' if len(items)>1 else ''}]{label_text}"
        )
        for act in sorted(items, key=lambda a: a.id):
            role = _role_marker(act.id, roles)
            role_suffix = f" [{role}]" if role else ""
            print(
                f"    • {act.id}: {act.pattern} "
                f"({act.valence}, stakes={act.axes.stakes}){role_suffix}"
            )
        print()

    print(
        f"THEORY-OF-MIND CLUES ({len(selected_toms)} selected out of {len(toms)} total):"
    )
    selected_act_segments = {act.segment for act in selected_acts}
    adjacent_segments = {
        seg + offset
        for seg in selected_act_segments
        for offset in (-1, 1)
    }
    early_tom_segments = set(tom_segments[:3])
    late_tom_segments = set(tom_segments[-2:])

    grouped_toms = defaultdict(list)
    for tom in selected_toms:
        grouped_toms[tom.segment].append(tom)

    for segment in sorted(grouped_toms.keys()):
        items = grouped_toms[segment]
        labels: list[str] = []
        if segment in selected_act_segments:
            labels.append("same segment as selected act")
        elif segment in adjacent_segments:
            labels.append("adjacent to selected act")
        if segment in early_tom_segments:
            labels.append("early ToM segment")
        if segment in late_tom_segments:
            labels.append("late ToM segment")

        prefix = "⚡" if segment in selected_act_segments or segment in adjacent_segments else ("✓" if labels else "•")
        label_text = f" - {', '.join(labels)}" if labels else ""
        print(
            f"  {prefix} Segment {segment:03d} [{len(items)} ToM{'' if len(items)==1 else 's'}]{label_text}"
        )
        for tom in sorted(items, key=lambda t: t.id):
            role = _role_marker(tom.id, roles)
            role_suffix = f" [{role}]" if role else ""
            print(
                f"    • {tom.id}: {tom.kind} — {tom.thinker} about {tom.target}{role_suffix}"
            )
        print()

    print("Selection Coverage:")
    act_cov = _percent(len(selected_acts), len(acts))
    tom_cov = _percent(len(selected_toms), len(toms))
    covered_segments = sorted({act.segment for act in selected_acts} | {tom.segment for tom in selected_toms})
    print(f"  • Acts: {len(selected_acts)}/{len(acts)} ({act_cov:.1f}%)")
    print(f"  • ToMs: {len(selected_toms)}/{len(toms)} ({tom_cov:.1f}%)")
    print(f"  • Segments covered: {', '.join(str(seg) for seg in covered_segments) or 'n/a'}")
    print()


def _print_statistics_and_patterns(
    acts: list[ActClue],
    toms: list[ToMClue],
    turning_timeline: list[TurningEntry],
) -> None:
    _print_section("STATISTICS & PATTERNS")

    if not acts and not toms:
        print("  (No clues available)")
        print()
        return

    total_acts = len(acts)
    total_toms = len(toms)
    positives = sum(1 for act in acts if act.valence == "positive")
    negatives = total_acts - positives

    stakes_counter = Counter(act.axes.stakes for act in acts)
    segments = sorted({act.segment for act in acts})
    span = _segment_span(segments)

    print("Act Analysis:")
    print(f"  Total: {total_acts} acts")
    if total_acts:
        print("  Valence Distribution:")
        print(f"    • Positive: {positives} ({_percent(positives, total_acts):.1f}%)")
        print(f"    • Negative: {negatives} ({_percent(negatives, total_acts):.1f}%)")
        print()
        print("  Stakes Distribution:")
        for level in ("major", "moderate", "minor"):
            value = stakes_counter.get(level, 0)
            print(f"    • {level.title()}: {value} ({_percent(value, total_acts):.1f}%)")
        print()

        if segments:
            density = total_acts / max(1, segments[-1] - segments[0] + 1)
            print(f"  Temporal Density: {density:.2f} acts/segment (span {span})")
            print()

    print("Theory-of-Mind Analysis:")
    print(f"  Total: {total_toms} ToMs")
    if total_toms:
        kind_counter = Counter(tom.kind for tom in toms)
        for kind, count in kind_counter.most_common():
            print(f"    • {kind}: {count} ({_percent(count, total_toms):.1f}%)")
        print(f"  ToM/Act Ratio: {total_toms / total_acts:.2f}" if total_acts else "  ToM/Act Ratio: n/a")
        print()

    print("Relationship Dynamics:")
    if turning_timeline:
        for turn in turning_timeline:
            print(
                f"  • Segment {turn.segment}: {turn.category} "
                f"({turn.pre_state.valence} → {turn.post_state.valence})"
            )
    else:
        print("  • No explicit turning points detected")
    print()

    insights: list[str] = []
    if acts and segments:
        major_count = stakes_counter.get("major", 0)
        if major_count / total_acts >= 0.5:
            insights.append(
                f"⚠ Major stakes concentration ({major_count}/{total_acts} acts)"
            )
    if acts and toms:
        min_act_seg = min(segments) if segments else None
        min_tom_seg = min(tom.segment for tom in toms)
        if min_act_seg is None or min_act_seg > min_tom_seg:
            insights.append("⚠ Early relationship context inferred from ToM clues only")
    if not insights:
        insights.append("• No additional warnings")

    print("Key Insights:")
    for line in insights:
        print(f"  {line}")
    print()


def _print_full_timeline(
    acts: list[ActClue],
    toms: list[ToMClue],
    selected_act_ids: set[str],
    selected_tom_ids: set[str],
    roles: dict[str, str],
) -> None:
    _print_section("FULL TIMELINE: All Clues (Reference)")
    print(
        "Note: Clues marked with [SELECTED] were provided to the LLM. "
        "⭐/• denote turning/supporting roles."
    )
    print()

    events = _build_timeline(acts, toms, roles)
    if not events:
        print("  (No timeline data)")
        print()
        return

    current_segment: int | None = None
    for event in events:
        if event.segment != current_segment:
            if current_segment is not None:
                print()
            current_segment = event.segment
            print("─" * 100)
            print(f"SEGMENT {event.segment:03d}")
            print("─" * 100)

        if event.event_type == "act":
            act = event.clue  # type: ignore[assignment]
            assert isinstance(act, ActClue)
            selected = " [SELECTED]" if act.id in selected_act_ids else ""
            role = _role_marker(act.id, roles)
            role_suffix = f" [{role}]" if role else ""
            print(f"  ACT [{act.id}]{selected}{role_suffix}")
            print(
                f"    [{act.valence}] {act.source} → {act.target}: {act.pattern}"
            )
            print(
                f"    Stakes={act.axes.stakes}, Salience={act.axes.salience}, "
                f"Durability={act.axes.durability}"
            )
            print(f'    Evidence: "{act.evidence.strip()}"')
        else:
            tom = event.clue  # type: ignore[assignment]
            assert isinstance(tom, ToMClue)
            selected = " [SELECTED]" if tom.id in selected_tom_ids else ""
            role = _role_marker(tom.id, roles)
            role_suffix = f" [{role}]" if role else ""
            print(f"  ToM [{tom.id}]{selected}{role_suffix}")
            print(
                f"    [{tom.kind}] {tom.thinker} about {tom.target}: "
                f'"{tom.claim.strip()}"'
            )
            print(f'    Evidence: "{tom.evidence.strip()}"')
    print("─" * 100)
    print("End of Timeline")
    print("─" * 100)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(output_dir: Path, char1: str, char2: str) -> None:
    acts, toms = load_clues(output_dir)

    pair = norm_pair(char1, char2)
    acts = [act for act in acts if norm_pair(act.source, act.target) == pair]
    toms = [
        tom
        for tom in toms
        if norm_pair(tom.thinker, tom.target) == pair
    ]

    dyad_info = load_dyad_info(output_dir, char1, char2)
    roles = dyad_info.event_roles if dyad_info else {}

    selected_acts = _select_acts_for_dossier(acts, limit=30)
    selected_toms = _select_toms_for_dossier(toms, selected_acts, limit=20)

    selected_act_ids = {act.id for act in selected_acts}
    selected_tom_ids = {tom.id for tom in selected_toms}

    _print_header(char1, char2)
    _print_executive_summary(acts, toms, selected_acts, selected_toms, dyad_info)
    _print_turning_points(
        dyad_info.turning_timeline if dyad_info else [],
        {act.id: act for act in acts},
        {tom.id: tom for tom in toms},
    )
    _print_selected_clues(acts, toms, selected_acts, selected_toms, roles)
    _print_statistics_and_patterns(
        acts,
        toms,
        dyad_info.turning_timeline if dyad_info else [],
    )
    _print_full_timeline(acts, toms, selected_act_ids, selected_tom_ids, roles)
    print("=" * 100)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a structured relationship analysis report."
    )
    parser.add_argument("output_dir", type=Path, help="Pipeline output directory")
    parser.add_argument("char1", help="First character name")
    parser.add_argument("char2", help="Second character name")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional file path to write the report",
    )
    args = parser.parse_args()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            import sys

            previous = sys.stdout
            sys.stdout = handle
            try:
                generate_report(args.output_dir, args.char1, args.char2)
            finally:
                sys.stdout = previous
        print(f"Report saved to {args.output}")
    else:
        generate_report(args.output_dir, args.char1, args.char2)


if __name__ == "__main__":
    main()
