#!/usr/bin/env python3
"""
특정 character pair에 대한 전체 관계 리포트 생성
Act, ToM, Dyad 정보를 timeline 순서로 표시
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clues.act import ActClue
from clues.tom import ToMClue
from framework.utils import norm_pair
from schema import TurningEntry, FinalRelation


@dataclass
class TimelineEvent:
    """Timeline에 표시할 이벤트"""

    segment: int
    event_type: str  # "act" or "tom"
    clue: ActClue | ToMClue
    is_turning: bool = False
    is_supporting: bool = False


@dataclass
class DyadInfo:
    """Dyad 분석 정보"""

    turning_timeline: list[TurningEntry]
    final_relation: FinalRelation
    event_roles: dict[str, str]  # {clue_id: "turning" or "supporting"}


def load_clues(output_dir: Path) -> tuple[list[ActClue], list[ToMClue]]:
    """JSONL 파일에서 clue 로드"""
    acts = []
    toms = []

    act_file = output_dir / "act_clues.jsonl"
    if act_file.exists():
        with act_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    acts.append(ActClue.model_validate(json.loads(line)))

    tom_file = output_dir / "tom_clues.jsonl"
    if tom_file.exists():
        with tom_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    toms.append(ToMClue.model_validate(json.loads(line)))

    return acts, toms


def load_dyad_info(output_dir: Path, char1: str, char2: str) -> DyadInfo | None:
    """특정 pair의 dyad 정보 로드"""
    dyad_file = output_dir / "dyad_results.jsonl"
    if not dyad_file.exists():
        return None

    target_pair = norm_pair(char1, char2)

    with dyad_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pair = norm_pair(data["char1"], data["char2"])

            if pair == target_pair:
                adj = data["adjudication"]

                # event_roles를 dict로 변환
                event_roles = {
                    item["id"]: item["event_role"]
                    for item in adj.get("event_roles", [])
                }

                return DyadInfo(
                    turning_timeline=[
                        TurningEntry.model_validate(t)
                        for t in adj.get("turning_timeline", [])
                    ],
                    final_relation=FinalRelation.model_validate(
                        adj.get("final_relation", {"valence": "neutral"})
                    ),
                    event_roles=event_roles,
                )

    return None


def build_timeline(
    acts: list[ActClue], toms: list[ToMClue], dyad_info: DyadInfo | None
) -> list[TimelineEvent]:
    """Act와 ToM을 segment 순서대로 정렬한 timeline 생성"""
    events = []

    for act in acts:
        role = dyad_info.event_roles.get(act.id, "") if dyad_info else ""
        events.append(
            TimelineEvent(
                segment=act.segment,
                event_type="act",
                clue=act,
                is_turning=(role == "turning"),
                is_supporting=(role == "supporting"),
            )
        )

    for tom in toms:
        role = dyad_info.event_roles.get(tom.id, "") if dyad_info else ""
        events.append(
            TimelineEvent(
                segment=tom.segment,
                event_type="tom",
                clue=tom,
                is_turning=(role == "turning"),
                is_supporting=(role == "supporting"),
            )
        )

    # segment 순서로 정렬
    events.sort(key=lambda e: e.segment)
    return events


def format_act_brief(act: ActClue, show_role: bool = False) -> str:
    """Act를 간략하게 포맷"""
    actors = "→".join(act.actors) if act.actors else act.pair[0]
    targets = "→".join(act.targets) if act.targets else act.pair[1]

    role_marker = ""
    if show_role:
        role_marker = " [TURNING]" if hasattr(act, "_is_turning") else ""
        role_marker = " [SUPPORT]" if hasattr(act, "_is_supporting") else role_marker

    return (
        f"    [{act.valence:>8}] {actors} → {targets}: {act.pattern}\n"
        f"                Stakes={act.axes.stakes}, Salience={act.axes.salience}, "
        f"Durability={act.axes.durability}{role_marker}\n"
        f'                "{act.evidence[:100]}..."'
    )


def format_tom_brief(tom: ToMClue) -> str:
    """ToM을 간략하게 포맷"""
    return (
        f"    [{tom.kind:>12}] {tom.pair[0]} about {tom.pair[1]}\n"
        f'                Claim: "{tom.claim}"\n'
        f'                Evidence: "{tom.evidence[:100]}..."'
    )


def print_header(char1: str, char2: str):
    """리포트 헤더 출력"""
    pair = norm_pair(char1, char2)
    print("=" * 100)
    print(f"RELATIONSHIP ANALYSIS REPORT: {pair[0]} ↔ {pair[1]}".center(100))
    print("=" * 100)
    print()


def print_timeline(events: list[TimelineEvent]):
    """Timeline 섹션 출력"""
    print("┌" + "─" * 98 + "┐")
    print("│" + " TIMELINE: All Events in Chronological Order ".center(98) + "│")
    print("└" + "─" * 98 + "┘")
    print()

    if not events:
        print("  (No events found)")
        print()
        return

    current_segment = None
    for event in events:
        if current_segment != event.segment:
            if current_segment is not None:
                print()
            current_segment = event.segment
            print(f"SEGMENT {event.segment:03d}")
            print("─" * 100)

        role_marker = ""
        if event.is_turning:
            role_marker = " ⭐ TURNING POINT"
        elif event.is_supporting:
            role_marker = " • supporting"

        if event.event_type == "act":
            print(f"  ACT [{event.clue.id}]{role_marker}")
            print(format_act_brief(event.clue))
        else:
            print(f"  ToM [{event.clue.id}]{role_marker}")
            print(format_tom_brief(event.clue))
        print()

    print()


def print_turning_points(turning_timeline: list[TurningEntry]):
    """Turning Points 섹션 출력"""
    print("┌" + "─" * 98 + "┐")
    print("│" + " TURNING POINTS: Key Relationship Changes ".center(98) + "│")
    print("└" + "─" * 98 + "┘")
    print()

    if not turning_timeline:
        print("  (No turning points detected)")
        print()
        return

    for i, turn in enumerate(turning_timeline, 1):
        print(f"#{i} SEGMENT {turn.segment} — {turn.category}")
        print("─" * 100)
        print(f"  Pattern: {turn.pattern}")
        print(f"  Summary: {turn.summary}")
        print(f'  Quote: "{turn.quote}"')
        print()
        print(
            f"  Pre-State:  valence={turn.pre_state.valence:>8}, "
            f"durability={turn.pre_state.durability:>10}",
            end="",
        )
        if turn.pre_state.label:
            print(f', label="{turn.pre_state.label}"')
        else:
            print()

        print(
            f"  Post-State: valence={turn.post_state.valence:>8}, "
            f"durability={turn.post_state.durability:>10}",
            end="",
        )
        if turn.post_state.label:
            print(f', label="{turn.post_state.label}"')
        else:
            print()

        print(f"  Representative Act: {turn.representative_act_id}")
        if turn.caused_by_tom_ids:
            print(f"  Caused by ToM: {', '.join(turn.caused_by_tom_ids)}")
        print()

    print()


def print_statistics(
    acts: list[ActClue], toms: list[ToMClue], dyad_info: DyadInfo | None
):
    """통계 섹션 출력"""
    print("┌" + "─" * 98 + "┐")
    print("│" + " STATISTICS & FINAL STATE ".center(98) + "│")
    print("└" + "─" * 98 + "┘")
    print()

    # Act 통계
    print("Act Clues:")
    print(f"  Total: {len(acts)}")
    if acts:
        pos = sum(1 for a in acts if a.valence == "positive")
        neg = sum(1 for a in acts if a.valence == "negative")
        print(
            f"  Valence: {pos} positive, {neg} negative ({pos / (pos + neg) * 100:.1f}% positive)"
        )

        from collections import Counter

        stakes = Counter(a.axes.stakes for a in acts)
        print(
            f"  Stakes: major={stakes['major']}, moderate={stakes['moderate']}, "
            f"minor={stakes['minor']}"
        )
    print()

    # ToM 통계
    print("Theory-of-Mind Clues:")
    print(f"  Total: {len(toms)}")
    if toms:
        from collections import Counter

        kinds = Counter(t.kind for t in toms)
        print(f"  Kinds: ", end="")
        print(", ".join(f"{k}={v}" for k, v in kinds.most_common()))
    print()

    # Final Relation
    if dyad_info:
        print("Final Relationship State:")
        print(f"  Valence: {dyad_info.final_relation.valence}")
        if dyad_info.final_relation.label:
            print(f'  Label: "{dyad_info.final_relation.label}"')
        print(f"  Turning Points: {len(dyad_info.turning_timeline)}")
    print()


def generate_report(output_dir: Path, char1: str, char2: str):
    """전체 리포트 생성"""
    # 데이터 로드
    print(f"Loading data from {output_dir}...")
    acts, toms = load_clues(output_dir)

    # 페어 필터링
    target_pair = norm_pair(char1, char2)
    acts = [a for a in acts if a.pair == target_pair]
    toms = [t for t in toms if t.pair == target_pair]

    # Dyad 정보 로드
    dyad_info = load_dyad_info(output_dir, char1, char2)

    # Timeline 구성
    timeline = build_timeline(acts, toms, dyad_info)

    # 리포트 출력
    print_header(char1, char2)
    print_timeline(timeline)

    if dyad_info:
        print_turning_points(dyad_info.turning_timeline)

    print_statistics(acts, toms, dyad_info)

    print("=" * 100)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="특정 character pair의 전체 관계 분석 리포트 생성"
    )
    parser.add_argument("output_dir", type=Path, help="Pipeline output directory")
    parser.add_argument("char1", help="First character name")
    parser.add_argument("char2", help="Second character name")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save report to file (default: print to stdout)",
    )
    args = parser.parse_args()

    if args.output:
        import sys

        original_stdout = sys.stdout
        with args.output.open("w", encoding="utf-8") as f:
            sys.stdout = f
            generate_report(args.output_dir, args.char1, args.char2)
        sys.stdout = original_stdout
        print(f"Report saved to {args.output}")
    else:
        generate_report(args.output_dir, args.char1, args.char2)


if __name__ == "__main__":
    main()
