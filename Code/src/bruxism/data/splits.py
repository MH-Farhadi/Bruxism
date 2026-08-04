"""Nested leave-one-subject-out splits, with a hard guard against outer-fold leakage.

The primary evaluation is outer LOSO over all participants. For each outer fold the held
-out participant is *sealed*: :class:`OuterFold` exposes its test sample ids only through
:meth:`OuterFold.release_test_ids`, which raises unless the caller declares
``purpose="final_evaluation"``. Selection code therefore cannot reach the outer subject
even by accident.

With five participants an outer fold leaves four training participants, so the inner
participant-grouped LOSO has **four** folds -- not five. Any request for a different
number of participant-grouped inner folds raises, because it is arithmetically impossible.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from bruxism.data.segments import WindowIndex
from bruxism.utils.logging import get_logger

__all__ = [
    "InnerFold",
    "NestedLOSOSplitter",
    "OuterFold",
    "OuterFoldSealError",
    "assert_group_disjoint",
]

logger = get_logger(__name__)


class OuterFoldSealError(RuntimeError):
    """Raised when code attempts to read outer-test data outside final evaluation."""


def assert_group_disjoint(**groups: Sequence[str]) -> None:
    """Assert that no sample id appears in more than one named group.

    Raises
    ------
    AssertionError
        Naming both offending groups and an example of the shared ids.
    """
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = set(groups[left]) & set(groups[right])
            if overlap:
                raise AssertionError(
                    f"leakage: {len(overlap)} sample ids shared between {left!r} and "
                    f"{right!r} (e.g. {sorted(overlap)[:3]})"
                )


@dataclass(frozen=True)
class InnerFold:
    """One inner participant-grouped fold used for hyperparameter and epoch selection."""

    fold_index: int
    outer_test_subject: str
    val_subject: str
    train_subjects: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.val_subject in self.train_subjects:
            raise AssertionError(
                f"inner fold {self.fold_index}: validation subject {self.val_subject} is "
                f"also in the training subjects"
            )
        if self.outer_test_subject in {*self.train_subjects, self.val_subject}:
            raise OuterFoldSealError(
                f"inner fold {self.fold_index} contains the outer test subject "
                f"{self.outer_test_subject}; the outer participant must never appear in "
                f"any inner split"
            )
        assert_group_disjoint(train=self.train_sample_ids, val=self.val_sample_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "outer_test_subject": self.outer_test_subject,
            "val_subject": self.val_subject,
            "train_subjects": list(self.train_subjects),
            "n_train": len(self.train_sample_ids),
            "n_val": len(self.val_sample_ids),
        }


@dataclass
class OuterFold:
    """One outer leave-one-subject-out fold.

    The held-out participant's sample ids are stored privately. Selection code sees only
    :attr:`n_test`; the ids themselves require an explicit declaration of intent.
    """

    fold_index: int
    test_subject: str
    train_subjects: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]
    _test_sample_ids: tuple[str, ...] = field(repr=False, default=())
    _released: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.test_subject in self.train_subjects:
            raise AssertionError(
                f"outer fold {self.fold_index}: test subject {self.test_subject} is also a "
                f"training subject"
            )
        assert_group_disjoint(outer_train=self.train_sample_ids, outer_test=self._test_sample_ids)
        expected = len(self.train_subjects)
        if len(self.inner_folds) != expected:
            raise ValueError(
                f"outer fold {self.fold_index} has {len(self.train_subjects)} training "
                f"participants, so participant-grouped inner LOSO must have {expected} "
                f"folds, but {len(self.inner_folds)} were supplied"
            )

    @property
    def n_test(self) -> int:
        """Number of held-out examples. Safe for logging; reveals no sample identity."""
        return len(self._test_sample_ids)

    @property
    def n_inner_folds(self) -> int:
        return len(self.inner_folds)

    @property
    def released(self) -> bool:
        """Whether the outer test ids have already been handed out."""
        return self._released

    def release_test_ids(self, *, purpose: str) -> tuple[str, ...]:
        """Hand out the held-out sample ids.

        Parameters
        ----------
        purpose
            Must be exactly ``"final_evaluation"``. Any other value raises, which is what
            makes accidental use during selection impossible rather than merely discouraged.

        Raises
        ------
        OuterFoldSealError
            If ``purpose`` is not ``"final_evaluation"``, or if the ids have already been
            released once (each outer participant is evaluated exactly once per
            task/model/seed).
        """
        if purpose != "final_evaluation":
            raise OuterFoldSealError(
                f"outer test data for {self.test_subject} requested with purpose "
                f"{purpose!r}; only 'final_evaluation' may read it. Hyperparameters, "
                f"epoch budgets, thresholds, normalisation and class weights must come "
                f"from inner folds."
            )
        if self._released:
            raise OuterFoldSealError(
                f"outer test data for {self.test_subject} has already been released once; "
                f"a second evaluation of the same held-out participant would violate the "
                f"evaluate-exactly-once protocol"
            )
        object.__setattr__(self, "_released", True)
        return self._test_sample_ids

    def peek_test_labels_for_reporting(self) -> tuple[str, ...]:
        """Return test ids without consuming the seal, for artifact bookkeeping only.

        Used by the prediction-ledger writer *after* :meth:`release_test_ids` has already
        been called, so it never widens access.
        """
        if not self._released:
            raise OuterFoldSealError(
                "test ids have not been released for final evaluation yet; call "
                "release_test_ids(purpose='final_evaluation') first"
            )
        return self._test_sample_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "test_subject": self.test_subject,
            "train_subjects": list(self.train_subjects),
            "n_train": len(self.train_sample_ids),
            "n_test": self.n_test,
            "inner_folds": [fold.to_dict() for fold in self.inner_folds],
        }


class NestedLOSOSplitter:
    """Build outer LOSO folds, each carrying its participant-grouped inner LOSO folds.

    Parameters
    ----------
    window_index
        Source of examples. Only its ``subject_id`` and ``sample_id`` fields are used.
    subjects
        Restrict to these participants (default: all present in ``window_index``).
    exclude_sample_ids
        Windows withheld from **every** split. Used for the per-participant calibration
        block: a window that set a participant's normalisation scale must not also be
        trained on or scored, or the calibrated arm would be flattered for a reason that
        has nothing to do with calibration working. Excluding here rather than in the
        trainer means no split can reach them at all.

    Raises
    ------
    ValueError
        If fewer than three participants are available -- an outer fold would then leave
        fewer than two training participants and inner LOSO would be degenerate.
    """

    def __init__(
        self,
        window_index: WindowIndex,
        *,
        subjects: Sequence[str] | None = None,
        exclude_sample_ids: Sequence[str] | frozenset[str] = (),
    ):
        self._index = window_index
        self.excluded_sample_ids: frozenset[str] = frozenset(exclude_sample_ids)
        by_subject: dict[str, list[str]] = {}
        for window in window_index.windows:
            if window.sample_id in self.excluded_sample_ids:
                continue
            by_subject.setdefault(window.subject_id, []).append(window.sample_id)
        if self.excluded_sample_ids:
            logger.info(
                "withholding %d calibration window(s) from every split",
                len(self.excluded_sample_ids),
                extra={"n_excluded": len(self.excluded_sample_ids)},
            )
        self._by_subject = {
            subject: tuple(sorted(ids)) for subject, ids in sorted(by_subject.items())
        }
        self.subjects: tuple[str, ...] = tuple(
            sorted(subjects) if subjects is not None else self._by_subject
        )
        missing = [s for s in self.subjects if s not in self._by_subject]
        if missing:
            raise ValueError(f"requested subjects not present in the window index: {missing}")
        if len(self.subjects) < 3:
            raise ValueError(
                f"nested LOSO needs at least 3 participants, got {len(self.subjects)}; "
                f"with fewer, an outer fold leaves too few participants for inner LOSO"
            )

    @property
    def n_outer_folds(self) -> int:
        return len(self.subjects)

    def n_inner_folds(self) -> int:
        """Inner folds per outer fold: one fewer than the participant count."""
        return len(self.subjects) - 1

    def _ids_for(self, subjects: Sequence[str]) -> tuple[str, ...]:
        ids: list[str] = []
        for subject in subjects:
            ids.extend(self._by_subject[subject])
        return tuple(sorted(ids))

    def outer_folds(self) -> list[OuterFold]:
        """Materialise every outer fold, with its inner folds already validated."""
        return list(self)

    def __iter__(self) -> Iterator[OuterFold]:
        for fold_index, test_subject in enumerate(self.subjects):
            train_subjects = tuple(s for s in self.subjects if s != test_subject)
            inner: list[InnerFold] = []
            for inner_index, val_subject in enumerate(train_subjects):
                inner_train = tuple(s for s in train_subjects if s != val_subject)
                inner.append(
                    InnerFold(
                        fold_index=inner_index,
                        outer_test_subject=test_subject,
                        val_subject=val_subject,
                        train_subjects=inner_train,
                        train_sample_ids=self._ids_for(inner_train),
                        val_sample_ids=self._ids_for([val_subject]),
                    )
                )
            yield OuterFold(
                fold_index=fold_index,
                test_subject=test_subject,
                train_subjects=train_subjects,
                train_sample_ids=self._ids_for(train_subjects),
                inner_folds=tuple(inner),
                _test_sample_ids=self._ids_for([test_subject]),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable description of the split plan, written to ``folds.json``."""
        folds = self.outer_folds()
        return {
            "scheme": "nested_leave_one_subject_out",
            "subjects": list(self.subjects),
            "n_outer_folds": self.n_outer_folds,
            "n_inner_folds_per_outer": self.n_inner_folds(),
            "n_windows_withheld_for_calibration": len(self.excluded_sample_ids),
            "window_index_hash": self._index.index_hash,
            "segmentation_policy": str(self._index.config.policy),
            "safe_for_inference": self._index.safe_for_inference,
            "folds": [fold.to_dict() for fold in folds],
        }
