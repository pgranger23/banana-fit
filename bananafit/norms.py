"""Normalization systematics (DESIGN.md §5).

A norm parameter is a flat multiplicative weight on some selection of events
(a sample, a mode, a flavor slice, ...) -- structurally a 1-knot spline, but
cheap enough to deserve its own path: no segment search, no coefficient
gather, just a masked affine map evaluated once per event.

``x_n`` has prior mean 1 (nominal = unchanged); ``mask[n, i]`` is normally a
0/1 selector but may hold fractional values (e.g. a norm that only partially
covers a bin, or a smoothly-varying acceptance correction) -- both read the
same way:

    w[i] = prod_n (1 + (x_n - 1) * mask[n, i])

which is exactly identity (``w[i] = 1``) for events with ``mask[n, i] == 0``
under every norm, and scales linearly in ``x_n`` for ``mask[n, i] == 1``.
Every term is a polynomial in finite inputs (no sqrt/log/division), so no
``where``-guarding is needed for gradient safety.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NormSet:
    """Masked multiplicative norm systematics.

    ``names``: static, one per norm parameter (``n_norm``).
    ``mask``: ``(n_norm, N)``, in ``{0, 1}`` or fractional response weights.
    """

    names: tuple[str, ...] = dataclasses.field(metadata=dict(static=True))
    mask: jax.Array  # (n_norm, N)

    def __call__(self, x: jax.Array) -> jax.Array:
        """``x``: ``(n_norm,)``, prior mean 1 -> per-event weight ``(N,)`` f64."""
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape[0] != self.mask.shape[0]:
            raise ValueError(
                f"x has {x.shape[0]} entries, expected n_norm={self.mask.shape[0]}"
            )
        mask = self.mask.astype(jnp.float64)
        factor = 1.0 + (x[:, None] - 1.0) * mask  # (n_norm, N)
        return jnp.prod(factor, axis=0)


def from_selections(
    names: Sequence[str],
    selections: Sequence[np.ndarray],
    n_events: int,
) -> NormSet:
    """Build a :class:`NormSet` from boolean (or fractional) selection arrays.

    ``selections[i]`` is a ``(n_events,)`` array (bool mask, or float in
    [0, 1] for a fractional/partial selection) naming which events norm
    ``names[i]`` applies to.
    """
    names = tuple(names)
    if len(selections) != len(names):
        raise ValueError(
            f"got {len(selections)} selections for {len(names)} names"
        )
    mask = np.zeros((len(names), n_events), dtype=np.float32)
    for i, sel in enumerate(selections):
        sel = np.asarray(sel)
        if sel.shape != (n_events,):
            raise ValueError(
                f"selection {i} ({names[i]!r}) has shape {sel.shape}, "
                f"expected ({n_events},)"
            )
        mask[i] = sel.astype(np.float32)
    return NormSet(names=names, mask=jnp.asarray(mask))
