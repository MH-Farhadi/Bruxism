"""Reproducible classification of instructed awake jaw / tooth-contact tasks.

Scope statement (read before using any result produced by this package):

This software classifies **instructed, awake jaw and tooth-contact tasks** recorded in a
controlled laboratory session from five participants. It is not a clinical bruxism
detector, it does not address sleep bruxism, and it has not been validated on
spontaneous or naturalistic behaviour. The raw filename token ``natural_bruxing`` is a
task label chosen at acquisition time; throughout this package the corresponding class is
named ``instructed_grinding``.

Sub-packages
------------
``bruxism.data``          Manifest construction, labelling policy, windowing, splits.
``bruxism.preprocessing`` Filters, normalisation, wavelets, training-only augmentation.
``bruxism.features``      Deterministic time-frequency feature extraction.
``bruxism.models``        Dual-branch network, modality ablations, baselines.
``bruxism.training``      Nested leave-one-subject-out engine, losses, selection rules.
``bruxism.evaluation``    Prediction ledger metrics, aggregation, runtime benchmarks.
``bruxism.visualization`` Paper figures generated exclusively from saved artifacts.
``bruxism.utils``         Reproducibility, atomic IO, structured logging.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
