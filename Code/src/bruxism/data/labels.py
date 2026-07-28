"""Condition taxonomy, task families and the classification tasks derived from them.

Two naming layers are kept deliberately separate:

* the **raw condition token** as written by the acquisition software
  (``natural_bruxing``, ``deviation_left_right``, ...), preserved verbatim in manifests;
* the **canonical condition** and **task family** used for reporting.

The token ``natural_bruxing`` is a filename artifact, not evidence that the recorded
behaviour was spontaneous. Its task family is :data:`TaskFamily.INSTRUCTED_GRINDING` and
it is reported as "instructed grinding" everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = [
    "CONDITION_TO_FAMILY",
    "RAW_TOKEN_TO_CONDITION",
    "TASK_DEFINITIONS",
    "ClassificationTask",
    "TaskFamily",
    "family_of_token",
    "get_task",
    "list_tasks",
]


class TaskFamily(StrEnum):
    """The five reporting families the manuscript's primary task is built from."""

    REST = "rest"
    MOVEMENT = "movement"
    CLENCH = "clench"
    INSTRUCTED_GRINDING = "instructed_grinding"
    CHEWING = "chewing"


#: Raw filename token -> canonical condition name. Identity except where the acquisition
#: token is abbreviated or scientifically misleading.
RAW_TOKEN_TO_CONDITION: Final[dict[str, str]] = {
    "rest": "rest",
    "open_close": "open_close",
    "deviation_left_right": "deviation",
    "protrusion_retrusion": "protrusion",
    "bite_left": "bite_left",
    "bite_right": "bite_right",
    "molar_clench": "molar_clench",
    "incisor_clench": "incisor_clench",
    # Raw token preserved in the manifest; reported as instructed grinding.
    "natural_bruxing": "instructed_grinding",
    "cheese": "cheese",
    "carrots": "carrots",
    "gum": "gum",
}

CONDITION_TO_FAMILY: Final[dict[str, TaskFamily]] = {
    "rest": TaskFamily.REST,
    "open_close": TaskFamily.MOVEMENT,
    "deviation": TaskFamily.MOVEMENT,
    "protrusion": TaskFamily.MOVEMENT,
    "bite_left": TaskFamily.CLENCH,
    "bite_right": TaskFamily.CLENCH,
    "molar_clench": TaskFamily.CLENCH,
    "incisor_clench": TaskFamily.CLENCH,
    "instructed_grinding": TaskFamily.INSTRUCTED_GRINDING,
    "cheese": TaskFamily.CHEWING,
    "carrots": TaskFamily.CHEWING,
    "gum": TaskFamily.CHEWING,
}


def family_of_token(raw_token: str) -> TaskFamily:
    """Map a raw filename condition token to its reporting family.

    Raises
    ------
    KeyError
        If the token is not part of the approved taxonomy. Unknown tokens must be
        resolved by a human, never silently dropped.
    """
    try:
        condition = RAW_TOKEN_TO_CONDITION[raw_token]
    except KeyError as exc:
        raise KeyError(
            f"unknown condition token {raw_token!r}; add it to RAW_TOKEN_TO_CONDITION "
            f"after confirming its meaning with the investigators"
        ) from exc
    return CONDITION_TO_FAMILY[condition]


@dataclass(frozen=True)
class ClassificationTask:
    """One classification problem defined over :class:`TaskFamily` groupings.

    Attributes
    ----------
    task_id
        Stable identifier recorded on every prediction row.
    class_names
        Ordered class names; the index in this tuple is the integer label.
    family_to_class
        Mapping from task family to class name. A family absent from this mapping is
        **excluded** from the task (e.g. chewing in the no-chewing task).
    primary
        Whether this is a purpose-trained scientific endpoint rather than a
        provenance-only or diagnostic configuration.
    notes
        Free text carried into generated tables so a reader sees the caveat with the number.
    """

    task_id: str
    description: str
    class_names: tuple[str, ...]
    family_to_class: dict[TaskFamily, str]
    primary: bool = True
    notes: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.class_names)}

    @property
    def included_families(self) -> frozenset[TaskFamily]:
        return frozenset(self.family_to_class)

    def label_for_family(self, family: TaskFamily) -> int | None:
        """Integer label for ``family``, or ``None`` when the family is excluded."""
        name = self.family_to_class.get(family)
        return None if name is None else self.class_to_index[name]

    def __post_init__(self) -> None:
        unknown = set(self.family_to_class.values()) - set(self.class_names)
        if unknown:
            raise ValueError(f"task {self.task_id!r} maps families to unknown classes: {unknown}")
        unused = set(self.class_names) - set(self.family_to_class.values())
        if unused:
            raise ValueError(f"task {self.task_id!r} declares unreachable classes: {unused}")


_ALL_FIVE: Final[dict[TaskFamily, str]] = {
    TaskFamily.REST: "rest",
    TaskFamily.MOVEMENT: "movement",
    TaskFamily.CLENCH: "clench",
    TaskFamily.INSTRUCTED_GRINDING: "instructed_grinding",
    TaskFamily.CHEWING: "chewing",
}

TASK_DEFINITIONS: Final[dict[str, ClassificationTask]] = {
    task.task_id: task
    for task in (
        ClassificationTask(
            task_id="five_class",
            description="Primary endpoint: rest / movement / clench / instructed grinding / chewing",
            class_names=("rest", "movement", "clench", "instructed_grinding", "chewing"),
            family_to_class=dict(_ALL_FIVE),
        ),
        ClassificationTask(
            task_id="no_chewing_four_class",
            description="Chewing removed, to test whether the easiest class props up the headline",
            class_names=("rest", "movement", "clench", "instructed_grinding"),
            family_to_class={
                TaskFamily.REST: "rest",
                TaskFamily.MOVEMENT: "movement",
                TaskFamily.CLENCH: "clench",
                TaskFamily.INSTRUCTED_GRINDING: "instructed_grinding",
            },
            notes="Chewing windows are excluded from training and evaluation, not merged.",
        ),
        ClassificationTask(
            task_id="binary_tooth_contact",
            description="Tooth-contact activity (clench + instructed grinding) vs everything else",
            class_names=("non_tooth_contact", "tooth_contact"),
            family_to_class={
                TaskFamily.REST: "non_tooth_contact",
                TaskFamily.MOVEMENT: "non_tooth_contact",
                TaskFamily.CHEWING: "non_tooth_contact",
                TaskFamily.CLENCH: "tooth_contact",
                TaskFamily.INSTRUCTED_GRINDING: "tooth_contact",
            },
            notes=(
                "Positive class is instructed tooth-contact behaviour, not a clinical "
                "bruxism diagnosis."
            ),
        ),
        ClassificationTask(
            task_id="ternary",
            description="Tooth contact vs chewing vs quiet/movement",
            class_names=("rest_or_movement", "tooth_contact", "chewing"),
            family_to_class={
                TaskFamily.REST: "rest_or_movement",
                TaskFamily.MOVEMENT: "rest_or_movement",
                TaskFamily.CLENCH: "tooth_contact",
                TaskFamily.INSTRUCTED_GRINDING: "tooth_contact",
                TaskFamily.CHEWING: "chewing",
            },
        ),
        ClassificationTask(
            task_id="legacy_active_four_class",
            description=(
                "Provenance only: the original four active classes with rest omitted. "
                "Reproduces the historical framing; not a scientific endpoint."
            ),
            class_names=("movement", "clench", "instructed_grinding", "chewing"),
            family_to_class={
                TaskFamily.MOVEMENT: "movement",
                TaskFamily.CLENCH: "clench",
                TaskFamily.INSTRUCTED_GRINDING: "instructed_grinding",
                TaskFamily.CHEWING: "chewing",
            },
            primary=False,
            notes=(
                "Historical framing that omits rest. Reported for provenance comparison "
                "with the original manuscript only."
            ),
        ),
    )
}


def get_task(task_id: str) -> ClassificationTask:
    """Look up a task definition by id, failing loudly on an unknown id."""
    try:
        return TASK_DEFINITIONS[task_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown task_id {task_id!r}; available: {sorted(TASK_DEFINITIONS)}"
        ) from exc


def list_tasks(*, primary_only: bool = False) -> list[ClassificationTask]:
    """All registered tasks, optionally restricted to scientific endpoints."""
    tasks = list(TASK_DEFINITIONS.values())
    return [task for task in tasks if task.primary] if primary_only else tasks
