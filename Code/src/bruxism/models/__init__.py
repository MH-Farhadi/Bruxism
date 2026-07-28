"""Model definitions and the interface every model in this project satisfies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class BruxismModel(Protocol):
    """The full contract every model in this project satisfies.

    ``torch.nn.Module.__getattr__`` is typed as returning ``Tensor | Module``, so a static
    checker cannot see project-specific methods on a value annotated as ``nn.Module``.
    Declaring the contract explicitly restores that visibility without weakening the check,
    and doubles as documentation of what the training engine actually requires of a model.
    """

    # --- project interface ---
    def __call__(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, num_classes)`` **logits**, never softmax probabilities."""
        ...

    def embed(self, emg: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        """Return the penultimate representation, ``(batch, embedding_dim)``."""
        ...

    def parameter_counts(self) -> dict[str, int]:
        """Programmatically computed parameter counts, per component and in total."""
        ...

    def architecture_record(self, sampling_rate: float = 1200.0) -> dict[str, Any]:
        """Everything needed to rebuild and describe the model, saved with the checkpoint."""
        ...

    # --- the torch.nn.Module surface the training engine relies on ---
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def to(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...
    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]: ...
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: Any, strict: bool = True) -> Any: ...


__all__ = ["BruxismModel"]
