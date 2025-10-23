from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from clues.registry import ClueRegistry
from framework.aliasing import AliasResolver
from framework.synthesis import DyadSynthesizer, build_bags
from framework.temporal import FabulaReconstructor
from framework.validation import ValidationContext, ValidationPipeline
from schema import BaseSignal


ActTransform = Callable[[Sequence[BaseSignal]], Sequence[BaseSignal]]


class Pipeline:
    """High-level orchestration for the narrative relationship framework."""

    def __init__(
        self,
        registry: ClueRegistry,
        *,
        validator: ValidationPipeline | None = None,
        temporal: FabulaReconstructor | None = None,
        alias_resolver: AliasResolver | None = None,
        synthesizer: DyadSynthesizer | None = None,
        act_bundle: ActTransform | None = None,
        act_explode: ActTransform | None = None,
    ) -> None:
        if synthesizer is None:
            raise ValueError("Pipeline requires a DyadSynthesizer instance")

        self.registry = registry
        self.validator = validator or ValidationPipeline(registry)
        self.temporal = temporal or FabulaReconstructor()
        self.aliaser = alias_resolver or AliasResolver()
        self.synthesizer = synthesizer
        self._bundle_acts: ActTransform = act_bundle or (lambda acts: list(acts))
        self._explode_acts: ActTransform = act_explode or (lambda acts: list(acts))

    def run(
        self,
        scenes: list[dict],
        *,
        metadata: dict[int, dict] | None = None,
    ) -> dict:
        metadata = metadata or {}
        scene_pairs = [(int(item["scene"]), str(item["text"])) for item in scenes]
        scene_ids = [sid for sid, _ in scene_pairs]

        signals_by_modality: dict[str, list[BaseSignal]] = defaultdict(list)
        all_signals: list[BaseSignal] = []
        for _, extractor in self.registry.items():
            outputs = list(extractor.batch_extract(scene_pairs))
            all_signals.extend(outputs)
            for signal in outputs:
                modality = getattr(signal, "modality", "")
                signals_by_modality[modality].append(signal)

        act_clues = list(signals_by_modality.get("act", []))
        tom_clues = list(signals_by_modality.get("tom", []))
        temporal_clues = list(signals_by_modality.get("temporal", []))
        entity_clues = list(signals_by_modality.get("entity", []))

        context = ValidationContext(
            known_scenes=set(scene_ids),
            representative_act_ids={getattr(a, "id", "") for a in act_clues if getattr(a, "id", "")},
        )
        validation_results = self.validator.validate_batch(all_signals, context=context)

        fabula_rank = self.temporal.reconstruct(scene_ids, all_signals, metadata)

        unique_names: set[str] = set()
        appearances: dict[str, set[int]] = defaultdict(set)
        for clue in [*act_clues, *tom_clues]:
            pair = getattr(clue, "pair", None)
            scene = int(getattr(clue, "scene", 0))
            if not pair:
                continue
            unique_names.update(pair)
            appearances[pair[0]].add(scene)
            appearances[pair[1]].add(scene)
        unique_names_list = sorted(unique_names)
        appearances_int = {k: sorted(v) for k, v in appearances.items()}
        alias_groups, alias_map = self.aliaser.resolve(
            unique_names_list, appearances_int, entity_clues
        )

        acts_representative = list(self._bundle_acts(act_clues))
        acts_directed = list(self._explode_acts(act_clues))
        bags = build_bags(tom_clues, acts_representative, acts_directed)
        dyad_results = self.synthesizer.run_batch(bags.items())

        return {
            "act_clues": act_clues,
            "tom_clues": tom_clues,
            "temporal_clues": temporal_clues,
            "entity_clues": entity_clues,
            "validation": validation_results,
            "fabula_rank": fabula_rank,
            "alias_groups": alias_groups,
            "alias_map": alias_map,
            "acts_representative": acts_representative,
            "acts_directed": acts_directed,
            "dyad_results": dyad_results,
        }

    def run_from_dir(
        self,
        scenes_path: Path,
        metadata_path: Path | None = None,
    ) -> dict:
        import json

        with scenes_path.open("r", encoding="utf-8") as f:
            scenes = [json.loads(line) for line in f if line.strip()]
        metadata = {}
        if metadata_path and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self.run(scenes, metadata=metadata)
