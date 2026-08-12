"""Modality ablation conditions, defined so that only the modality varies.

.. warning::

   **The conditions below isolate the modality in code, and the data does not cooperate.**
   The audit in ``audio.md`` found that the microphone channel of the 2025-08 collection
   contains 37 distinct waveforms across 100 recordings, that 83 of those recordings carry
   a waveform bit-identical (after a circular rotation) to another *participant's*
   recording of the same condition, and that all four S01-S04 quiet-rest recordings share
   one waveform. A held-out participant's audio is therefore already in the training fold,
   so these conditions were not evaluated leave-one-subject-out on that channel. Every
   difference they produce bounds what a duplicated, unaligned, envelope-band channel
   contributes under this architecture; none of them measures what a microphone
   contributes, and the direction of the bias is not uniform -- it is optimistic where the
   duplicated waveform carries the matching label and pessimistic where it does not. The
   harness is correct; the input is not.
   :func:`~bruxism.runner.assert_modality_is_supported_by_data` now refuses such a run
   unless the experiment declares ``mic_defect_acknowledged_by``.

The reviewer concern this addresses (``Temp.md`` item E) is that the microphone may mainly
be detecting *eating* rather than contributing to the clinically relevant clench/grind
distinction. Answering it requires the EMG-only, audio-only and fusion conditions to be
identical in every respect except which modality the model sees: same windows, same folds,
same seeds, same selection budget, same evaluation code.

:class:`AblationSpec` is the unit that guarantees that -- the runner iterates specs and
changes nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from bruxism.models.dual_branch import Modality

__all__ = [
    "MODALITY_CONDITIONS",
    "AblationSpec",
    "chewing_contrast_pairs",
    "modality_specs",
]

#: The three matched modality conditions, in reporting order.
MODALITY_CONDITIONS: tuple[Modality, ...] = ("fusion", "emg_only", "audio_only")


@dataclass(frozen=True)
class AblationSpec:
    """One fully specified experimental condition.

    Every field that could change a result is explicit, so two specs that differ only in
    ``modality`` are guaranteed comparable.
    """

    task_id: str
    model_id: str
    modality: Modality

    @property
    def condition_id(self) -> str:
        """Stable identifier recorded on every prediction row."""
        return f"{self.task_id}::{self.model_id}::{self.modality}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "modality": self.modality,
        }


def modality_specs(
    task_ids: Sequence[str],
    *,
    model_id: str = "dual_branch_wavelet_cnn",
    modalities: Sequence[Modality] = MODALITY_CONDITIONS,
) -> list[AblationSpec]:
    """Cross ``task_ids`` with ``modalities`` for a single architecture."""
    return [
        AblationSpec(task_id=task_id, model_id=model_id, modality=modality)
        for task_id in task_ids
        for modality in modalities
    ]


def baseline_specs(
    task_id: str,
    model_ids: Sequence[str],
    *,
    modality: Modality = "fusion",
) -> list[AblationSpec]:
    """Architecture comparison on one task, all models receiving the same modality."""
    return [
        AblationSpec(task_id=task_id, model_id=model_id, modality=modality)
        for model_id in model_ids
    ]


def chewing_contrast_pairs(
    specs: Sequence[AblationSpec],
    *,
    with_chewing_task: str = "five_class",
    without_chewing_task: str = "no_chewing_four_class",
) -> Iterator[tuple[AblationSpec, AblationSpec]]:
    """Pair each with-chewing condition with its no-chewing twin.

    The audio benefit must be reported *both* on the full five-class task and after chewing
    is removed. A gain that survives only in the first case means audio is detecting
    eating, and the manuscript has to say so.
    """
    without = {
        (spec.model_id, spec.modality): spec
        for spec in specs
        if spec.task_id == without_chewing_task
    }
    for spec in specs:
        if spec.task_id != with_chewing_task:
            continue
        twin = without.get((spec.model_id, spec.modality))
        if twin is not None:
            yield spec, twin
