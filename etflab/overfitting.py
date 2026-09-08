"""How much of a backtest's edge is selection bias.

If you evaluate thirty configurations and report the best, the reported result is
the maximum of thirty noisy estimates. That maximum is biased upward even when
every configuration is worthless. Combinatorially Symmetric Cross-Validation
answers the operational version of the question -- *if I select on in-sample
performance, how often does the winner land below median out of sample?* -- and
calls that the Probability of Backtest Overfitting.

Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J. (2017),
"The Probability of Backtest Overfitting", *Journal of Computational Finance*
20(4), 39-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OverfittingResult:
    """Everything CSCV produced, including the parts that are inconvenient."""

    pbo: float
    n_combinations: int
    n_configs: int
    n_splits: int
    logits: np.ndarray
    selected_configs: list[str]
    is_oos_slope: float
    is_oos_correlation: float
    median_oos_rank: float

    @property
    def verdict(self) -> str:
        if self.pbo < 0.20:
            return "low: in-sample selection generalises"
        if self.pbo < 0.50:
            return "moderate: selection carries real risk"
        return "high: in-sample ranking is close to noise"

    def as_dict(self) -> dict[str, Any]:
        counts = (
            pd.Series(self.selected_configs).value_counts(normalize=True)
            if self.selected_configs
            else pd.Series(dtype=float)
        )
        return {
            "pbo": self.pbo,
            "verdict": self.verdict,
            "n_combinations": self.n_combinations,
            "n_configs": self.n_configs,
            "n_splits": self.n_splits,
            "is_oos_slope": self.is_oos_slope,
            "is_oos_correlation": self.is_oos_correlation,
            "median_oos_rank": self.median_oos_rank,
            "selection_frequency": {str(k): float(v) for k, v in counts.head(10).items()},
        }


def probability_of_backtest_overfitting(
    performance: pd.DataFrame,
    *,
    n_splits: int = 12,
    max_combinations: int = 2000,
) -> OverfittingResult:
    """Run CSCV over a matrix of per-period performance contributions.

    Parameters
    ----------
    performance:
        ``(n_periods, n_configs)``. Each column is one configuration's per-period
        performance, higher being better. For a tracking problem the natural
        choice is the negative squared active return, so that "better" means
        "tracked more tightly".
    n_splits:
        Number of contiguous, equal sub-periods ``S``. Must be even. Every
        balanced split of the ``S`` blocks into train and test halves is used, so
        the procedure is symmetric: unlike a single walk-forward split it never
        privileges one arbitrary cut of history.
    max_combinations:
        Cap on ``C(S, S/2)``, sampled deterministically from the front of the
        enumeration when exceeded. The cap is reported, never silent.

    Notes
    -----
    Ties in the out-of-sample ranking are broken by averaging, and the logit is
    clipped away from infinity so a single perfect or catastrophic combination
    cannot dominate the mean.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so train and test halves are the same size.")
    frame = performance.dropna()
    n_periods, n_configs = frame.shape
    if n_configs < 2:
        raise ValueError("PBO needs at least two configurations to rank.")
    if n_periods < n_splits * 10:
        raise ValueError(f"Need at least {n_splits * 10} periods for {n_splits} splits; got {n_periods}.")

    blocks = np.array_split(np.arange(n_periods), n_splits)
    values = frame.to_numpy(dtype=float)
    # Pre-aggregate per block so each combination is a cheap sum rather than a slice.
    block_sums = np.array([values[b].sum(axis=0) for b in blocks])  # (n_splits, n_configs)
    block_lengths = np.array([len(b) for b in blocks], dtype=float)

    all_blocks = set(range(n_splits))
    combos = list(combinations(range(n_splits), n_splits // 2))
    truncated = len(combos) > max_combinations
    if truncated:
        step = len(combos) / max_combinations
        combos = [combos[int(i * step)] for i in range(max_combinations)]

    logits: list[float] = []
    selected: list[str] = []
    is_scores: list[float] = []
    oos_scores: list[float] = []
    ranks: list[float] = []

    for train_blocks in combos:
        train = list(train_blocks)
        test = sorted(all_blocks - set(train))
        train_mean = block_sums[train].sum(axis=0) / block_lengths[train].sum()
        test_mean = block_sums[test].sum(axis=0) / block_lengths[test].sum()

        best = int(np.argmax(train_mean))
        # Relative rank of the in-sample winner among out-of-sample results,
        # averaged over ties, mapped into (0, 1).
        order = pd.Series(test_mean).rank(method="average").to_numpy()
        omega = float(order[best] / (n_configs + 1.0))
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(omega / (1.0 - omega))))
        selected.append(str(frame.columns[best]))
        is_scores.append(float(train_mean[best]))
        oos_scores.append(float(test_mean[best]))
        ranks.append(omega)

    logit_array = np.asarray(logits)
    pbo = float((logit_array < 0).mean())

    is_array, oos_array = np.asarray(is_scores), np.asarray(oos_scores)
    if is_array.std() > 0:
        slope = float(np.polyfit(is_array, oos_array, 1)[0])
        correlation = float(np.corrcoef(is_array, oos_array)[0, 1])
    else:
        slope, correlation = float("nan"), float("nan")

    return OverfittingResult(
        pbo=pbo,
        n_combinations=len(combos),
        n_configs=n_configs,
        n_splits=n_splits,
        logits=logit_array,
        selected_configs=selected,
        is_oos_slope=slope,
        is_oos_correlation=correlation,
        median_oos_rank=float(np.median(ranks)),
    )


def performance_from_active(active_by_config: pd.DataFrame) -> pd.DataFrame:
    """Turn active-return series into a CSCV performance matrix.

    Negative squared active return: higher is better, and the mean over any
    sub-period is exactly the negative mean squared tracking error over that
    sub-period, which is the quantity the optimiser is trying to minimise. Using
    the strategy's own objective as the selection criterion is the point --
    selecting on a different metric would understate the overfitting risk of the
    selection actually being performed.
    """
    return -(active_by_config**2)


def noise_reference_pbo(
    n_periods: int,
    n_configs: int,
    *,
    n_splits: int = 12,
    seed: int = 11,
    max_combinations: int = 2000,
    n_replications: int = 5,
) -> float:
    """PBO of a matched matrix of pure noise -- the empirical null for this shape.

    The textbook reading is "PBO near 0.5 means the ranking is noise". That is
    not quite right for CSCV on a *fixed* sample: because the in-sample and
    out-of-sample halves partition one finite realisation, a configuration that
    ran hot in-sample has, mechanically, less good luck left for the complement.
    That induces negative dependence and pushes the null above 0.5.

    So rather than compare against a constant, generate a matrix of the same
    shape from pure noise and compare against *that*. A reported PBO materially
    below this reference is evidence of real, transferable structure; one at or
    above it is not.

    Averaged over ``n_replications`` independent noise matrices, because the PBO
    of any *single* realisation is itself high variance: on one finite sample a
    configuration can be lucky throughout, and CSCV will correctly report a low
    PBO for that sample. Comparing an observed PBO against a one-draw reference
    would therefore be comparing it against a coin flip.
    """
    values = []
    for replication in range(max(1, n_replications)):
        rng = np.random.default_rng(seed + replication * 1_000)
        noise = pd.DataFrame(
            rng.standard_normal((n_periods, n_configs)),
            columns=[f"noise_{i}" for i in range(n_configs)],
        )
        values.append(
            probability_of_backtest_overfitting(noise, n_splits=n_splits, max_combinations=max_combinations).pbo
        )
    return float(np.mean(values))
