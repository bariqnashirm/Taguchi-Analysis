"""
taguchi_analyzer.py
===================

Core module for Taguchi Method Design of Experiments (DoE) analysis.

This module provides the :class:`TaguchiAnalyzer` class, an object-oriented
wrapper for the typical workflow of Taguchi analysis:

1. Loading experimental data from a CSV file.
2. Computing Signal-to-Noise (S/N) ratios using the three classical criteria:
   - Larger-the-better
   - Smaller-the-better
   - Nominal-the-best
3. Computing the mean response (and mean S/N) per level of each control factor
   to identify the optimal level combination.
4. Running a simple one-way ANOVA on the S/N ratios to estimate the relative
   contribution of each factor.
5. Generating Main Effects Plots for the S/N ratios.

Author: Taguchi-DoE-Toolkit contributors
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Type aliases & constants
# ---------------------------------------------------------------------------

SNCriterion = str  # one of {"larger", "smaller", "nominal"}

_VALID_CRITERIA = {"larger", "smaller", "nominal"}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FactorEffect:
    """Container for the effect of a single control factor.

    Attributes
    ----------
    name : str
        Name of the factor (column name in the input DataFrame).
    level_means : pd.Series
        Mean S/N ratio (or mean response) per level, indexed by level value.
    delta : float
        Range of the level means (max - min). A larger delta indicates a more
        influential factor.
    rank : int
        Rank of this factor based on ``delta``. Rank 1 is the most influential.
        Set later by :meth:`TaguchiAnalyzer.compute_main_effects`.
    """

    name: str
    level_means: pd.Series
    delta: float
    rank: int = 0


@dataclass
class AnovaResult:
    """Container for a simple one-way ANOVA decomposition on S/N ratios.

    The decomposition is performed on the S/N values, treating each factor
    independently (as is conventional in basic Taguchi analysis when the
    orthogonal array is balanced).

    Attributes
    ----------
    table : pd.DataFrame
        Per-factor ANOVA table with columns:
        ``["DoF", "SS", "MS", "F", "Contribution_%"]``.
    total_ss : float
        Total sum of squares of the S/N values.
    grand_mean : float
        Grand mean of the S/N values.
    """

    table: pd.DataFrame
    total_ss: float
    grand_mean: float


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class TaguchiAnalyzer:
    """Object-oriented analyzer for Taguchi Method experiments.

    Parameters
    ----------
    factors : sequence of str
        Names of the control factor columns in the data (e.g. ``["A", "B", "C"]``).
    response : str
        Name of the response column. If multiple replicates were collected per
        run, pass ``replicate_columns`` instead and leave ``response`` as the
        label you want to give to the aggregated response.
    replicate_columns : sequence of str, optional
        Names of replicate response columns (e.g. ``["y1", "y2", "y3"]``).
        When provided, S/N ratios are computed across the replicates within each
        experimental run. When omitted, the single ``response`` column is used
        and S/N is computed treating each row as a single observation
        (degenerate case, mostly useful for already-aggregated data).
    criterion : {"larger", "smaller", "nominal"}, default "larger"
        S/N criterion to use.

    Examples
    --------
    >>> analyzer = TaguchiAnalyzer(
    ...     factors=["A", "B", "C"],
    ...     response="strength",
    ...     replicate_columns=["y1", "y2", "y3"],
    ...     criterion="larger",
    ... )
    >>> analyzer.load_data("experiment.csv")
    >>> analyzer.compute_sn_ratio()
    >>> effects = analyzer.compute_main_effects()
    >>> anova = analyzer.run_anova()
    >>> analyzer.plot_main_effects()
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        factors: Sequence[str],
        response: str,
        replicate_columns: Sequence[str] | None = None,
        criterion: SNCriterion = "larger",
        target: float | None = None,
    ) -> None:
        if criterion not in _VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {_VALID_CRITERIA}, got {criterion!r}"
            )
        if not factors:
            raise ValueError("`factors` must contain at least one factor name.")

        self.factors: list[str] = list(factors)
        self.response: str = response
        self.replicate_columns: list[str] | None = (
            list(replicate_columns) if replicate_columns else None
        )
        self.criterion: SNCriterion = criterion
        self.target: float | None = target

        # Populated by methods below
        self.data: pd.DataFrame | None = None
        self.sn_data: pd.DataFrame | None = None
        self.effects: dict[str, FactorEffect] = {}
        self.anova: AnovaResult | None = None

    # ----------------------------------------------------------- data input
    def load_data(self, csv_path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
        """Load experimental data from a CSV file.

        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file containing factor columns and response columns.
        **read_csv_kwargs
            Additional keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame, also stored in ``self.data``.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If declared factor / response columns are missing from the CSV.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, **read_csv_kwargs)
        self._validate_columns(df)
        self.data = df.copy()
        return self.data

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load experimental data from an existing DataFrame (for tests / programmatic use).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the factor columns and the response columns.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame, also stored in ``self.data``.
        """
        self._validate_columns(df)
        self.data = df.copy()
        return self.data

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Internal: ensure all declared columns exist in the DataFrame."""
        required: list[str] = list(self.factors)
        if self.replicate_columns:
            required.extend(self.replicate_columns)
        else:
            required.append(self.response)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following required columns are missing from the data: {missing}"
            )

    # ----------------------------------------------------- S/N ratio kernel
    @staticmethod
    def _sn_larger_the_better(values: np.ndarray) -> float:
        """S/N for larger-the-better: ``-10 * log10(mean(1/y^2))``.

        Notes
        -----
        Requires strictly positive values; zeros or negatives raise a
        :class:`ValueError` because ``1/y^2`` would be undefined or misleading.
        """
        values = np.asarray(values, dtype=float)
        if np.any(values <= 0):
            raise ValueError(
                "Larger-the-better S/N requires strictly positive responses."
            )
        return -10.0 * math.log10(float(np.mean(1.0 / (values ** 2))))

    @staticmethod
    def _sn_smaller_the_better(values: np.ndarray) -> float:
        """S/N for smaller-the-better: ``-10 * log10(mean(y^2))``."""
        values = np.asarray(values, dtype=float)
        return -10.0 * math.log10(float(np.mean(values ** 2)))

    @staticmethod
    def _sn_nominal_the_best(values: np.ndarray, target: float | None = None) -> float:
        """S/N for nominal-the-best.

        If ``target`` is ``None`` (the classical Taguchi formulation), uses
        ``10 * log10(mean^2 / variance)``. If ``target`` is provided, uses
        the mean-squared-deviation form ``-10 * log10(mean((y - target)^2))``,
        which is appropriate when the target value is a known design spec.
        """
        values = np.asarray(values, dtype=float)
        if target is None:
            mean = float(np.mean(values))
            # population variance (ddof=0) is conventional in many Taguchi
            # textbooks; switch to ddof=1 if your reference prefers sample variance.
            var = float(np.var(values, ddof=0))
            if var <= 0:
                # All replicates identical — S/N tends to +infinity. Cap it.
                return float("inf") if mean != 0 else 0.0
            return 10.0 * math.log10((mean ** 2) / var)
        else:
            msd = float(np.mean((values - target) ** 2))
            if msd <= 0:
                return float("inf")
            return -10.0 * math.log10(msd)

    def _compute_sn_for_row(self, row_values: np.ndarray) -> float:
        """Dispatch S/N computation to the chosen criterion."""
        if self.criterion == "larger":
            return self._sn_larger_the_better(row_values)
        if self.criterion == "smaller":
            return self._sn_smaller_the_better(row_values)
        # nominal
        return self._sn_nominal_the_best(row_values, target=self.target)

    def compute_sn_ratio(self) -> pd.DataFrame:
        """Compute the S/N ratio for each experimental run.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the original factor columns plus two new columns:
            ``"mean_response"`` (the per-run mean of replicates) and
            ``"SN"`` (the S/N ratio according to the selected criterion).
            Also stored in ``self.sn_data``.

        Raises
        ------
        RuntimeError
            If :meth:`load_data` was not called first.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call `load_data()` first.")

        df = self.data
        factor_block = df[self.factors].copy()

        if self.replicate_columns:
            response_block = df[self.replicate_columns].to_numpy(dtype=float)
        else:
            # Single response column treated as a 1-replicate observation.
            response_block = df[[self.response]].to_numpy(dtype=float)

        sn_values = np.array(
            [self._compute_sn_for_row(row) for row in response_block]
        )
        mean_values = response_block.mean(axis=1)

        result = factor_block
        result["mean_response"] = mean_values
        result["SN"] = sn_values

        self.sn_data = result.reset_index(drop=True)
        return self.sn_data

    # --------------------------------------------------------- main effects
    def compute_main_effects(self, on: str = "SN") -> dict[str, FactorEffect]:
        """Compute the mean of ``on`` (S/N or mean response) per level of each factor.

        Parameters
        ----------
        on : {"SN", "mean_response"}, default "SN"
            Which column of ``self.sn_data`` to summarise. Use ``"SN"`` for the
            classical Taguchi main-effects analysis, or ``"mean_response"`` to
            inspect the raw response.

        Returns
        -------
        dict[str, FactorEffect]
            Mapping from factor name to a :class:`FactorEffect` describing the
            level means, the delta (range), and the influence rank.
        """
        if self.sn_data is None:
            raise RuntimeError("Call `compute_sn_ratio()` before `compute_main_effects()`.")
        if on not in {"SN", "mean_response"}:
            raise ValueError("`on` must be 'SN' or 'mean_response'.")

        effects: dict[str, FactorEffect] = {}
        for factor in self.factors:
            grouped = self.sn_data.groupby(factor)[on].mean().sort_index()
            delta = float(grouped.max() - grouped.min())
            effects[factor] = FactorEffect(
                name=factor,
                level_means=grouped,
                delta=delta,
            )

        # Rank by delta (largest delta = most influential = rank 1)
        sorted_factors = sorted(effects.values(), key=lambda e: e.delta, reverse=True)
        for rank, effect in enumerate(sorted_factors, start=1):
            effect.rank = rank

        self.effects = effects
        return effects

    def optimal_levels(self, on: str = "SN") -> dict[str, object]:
        """Return the level of each factor that maximises ``on``.

        For S/N ratios, the optimum is always **maximisation**, regardless of
        the criterion — the criterion choice is already baked into the S/N
        formula. For ``mean_response``, maximisation is appropriate for
        larger-the-better; for smaller-the-better you should pass
        ``on="SN"`` (recommended) or invert the result manually.

        Returns
        -------
        dict[str, object]
            Mapping ``{factor_name: optimal_level_value}``.
        """
        if not self.effects:
            self.compute_main_effects(on=on)

        return {name: eff.level_means.idxmax() for name, eff in self.effects.items()}

    # ------------------------------------------------------------ ANOVA
    def run_anova(self) -> AnovaResult:
        """Run a simple one-way ANOVA decomposition on the S/N ratios.

        For each factor :math:`i` with levels indexed by :math:`j`, with
        :math:`n_{ij}` runs at level :math:`j` and grand mean
        :math:`\\bar{S}`, the sum of squares is

        .. math::
            SS_i = \\sum_j n_{ij} (\\bar{S}_{ij} - \\bar{S})^2

        Degrees of freedom are :math:`L_i - 1` where :math:`L_i` is the
        number of levels of factor :math:`i`. The error SS is computed as
        the residual after subtracting all factor SS from the total SS;
        this is a simplified estimate that assumes negligible interaction
        — the standard Taguchi screening assumption.

        Returns
        -------
        AnovaResult
            Container with the per-factor table, total SS, and grand mean.
            Also stored in ``self.anova``.
        """
        if self.sn_data is None:
            raise RuntimeError("Call `compute_sn_ratio()` before `run_anova()`.")

        sn = self.sn_data["SN"].to_numpy(dtype=float)
        grand_mean = float(np.mean(sn))
        total_ss = float(np.sum((sn - grand_mean) ** 2))
        n_total = len(sn)

        rows: list[dict[str, float]] = []
        factors_dof_total = 0
        factors_ss_total = 0.0

        for factor in self.factors:
            grouped = self.sn_data.groupby(factor)["SN"]
            level_means = grouped.mean()
            level_counts = grouped.count()
            ss = float(
                np.sum(level_counts.values * (level_means.values - grand_mean) ** 2)
            )
            dof = int(len(level_means) - 1)
            ms = ss / dof if dof > 0 else float("nan")

            rows.append(
                {
                    "Factor": factor,
                    "DoF": dof,
                    "SS": ss,
                    "MS": ms,
                    "Contribution_%": 100.0 * ss / total_ss if total_ss > 0 else 0.0,
                }
            )
            factors_dof_total += dof
            factors_ss_total += ss

        # Error term (residual)
        error_dof = max(n_total - 1 - factors_dof_total, 0)
        error_ss = max(total_ss - factors_ss_total, 0.0)
        error_ms = error_ss / error_dof if error_dof > 0 else float("nan")

        # F-ratio = MS_factor / MS_error
        for row in rows:
            row["F"] = row["MS"] / error_ms if error_ms and not math.isnan(error_ms) else float("nan")

        rows.append(
            {
                "Factor": "Error",
                "DoF": error_dof,
                "SS": error_ss,
                "MS": error_ms,
                "F": float("nan"),
                "Contribution_%": 100.0 * error_ss / total_ss if total_ss > 0 else 0.0,
            }
        )
        rows.append(
            {
                "Factor": "Total",
                "DoF": n_total - 1,
                "SS": total_ss,
                "MS": float("nan"),
                "F": float("nan"),
                "Contribution_%": 100.0,
            }
        )

        # Reorder columns for readability
        table = pd.DataFrame(rows)[["Factor", "DoF", "SS", "MS", "F", "Contribution_%"]]
        self.anova = AnovaResult(
            table=table,
            total_ss=total_ss,
            grand_mean=grand_mean,
        )
        return self.anova

    # ------------------------------------------------------------- plotting
    def plot_main_effects(
        self,
        on: str = "SN",
        figsize: tuple[float, float] | None = None,
        save_path: str | Path | None = None,
        show: bool = True,
        style: str = "journal",
        ylabel: str | None = None,
        highlight_optimum: bool = True,
        show_grand_mean: bool = True,
    ) -> plt.Figure:
        """Plot the Main Effects Plot for S/N ratios (or mean response).

        Two visual styles are provided:

        - ``style="journal"`` (default) — black-and-white, publication-quality
          look mirroring typical Taguchi DoE figures in mechanical-engineering
          journals: open circles at each level, a filled circle highlighting
          the optimum, dashed grid, and vertical separators between factors
          with the factor name as the x-axis label.
        - ``style="modern"`` — colourful matplotlib defaults with per-subplot
          titles showing Δ and the influence rank. Good for slides and
          dashboards.

        Parameters
        ----------
        on : {"SN", "mean_response"}, default "SN"
            Which quantity to plot.
        figsize : (float, float), optional
            Figure size in inches. Defaults to ``(2.4 * n_factors, 4)`` for
            "journal" style, ``(4 * n_factors, 4)`` for "modern".
        save_path : str or Path, optional
            If provided, save the figure to this path (PNG, PDF, SVG, ...).
        show : bool, default True
            Call ``plt.show()`` at the end. Set to ``False`` for headless
            environments or when embedding in tests / notebooks.
        style : {"journal", "modern"}, default "journal"
            Visual style preset (see above).
        ylabel : str, optional
            Custom y-axis label. If omitted, a sensible default is derived
            from ``on`` (e.g. ``"Mean S/N Ratio (dB)"``). Useful when you want
            the axis to show the actual response name and unit, e.g.
            ``"PD (g)"`` or ``"Ra (μm)"``.
        highlight_optimum : bool, default True
            If True (journal style only), the level that maximises S/N for
            each factor is drawn as a solid filled circle.
        show_grand_mean : bool, default True
            If True, draw a horizontal dashed line at the grand mean.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure (useful for further customisation).
        """
        valid_styles = {"journal", "modern"}
        if style not in valid_styles:
            raise ValueError(f"style must be one of {valid_styles}, got {style!r}")

        if not self.effects or list(self.effects.values())[0].level_means.name != on:
            self.compute_main_effects(on=on)

        if style == "journal":
            return self._plot_journal_style(
                on=on,
                figsize=figsize,
                save_path=save_path,
                show=show,
                ylabel=ylabel,
                highlight_optimum=highlight_optimum,
                show_grand_mean=show_grand_mean,
            )
        return self._plot_modern_style(
            on=on,
            figsize=figsize,
            save_path=save_path,
            show=show,
            ylabel=ylabel,
            show_grand_mean=show_grand_mean,
        )

    # ........................................................ journal style
    def _plot_journal_style(
        self,
        on: str,
        figsize: tuple[float, float] | None,
        save_path: str | Path | None,
        show: bool,
        ylabel: str | None,
        highlight_optimum: bool,
        show_grand_mean: bool,
    ) -> plt.Figure:
        """Black-and-white, publication-style Main Effects Plot."""
        n = len(self.factors)
        figsize = figsize or (max(2.4 * n, 6), 4)

        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        if n == 1:
            axes = [axes]

        default_ylabel = "Mean S/N Ratio (dB)" if on == "SN" else f"Mean of {on}"
        y_label_text = ylabel if ylabel is not None else default_ylabel
        grand_mean = float(self.sn_data[on].mean())

        # Find the optimal level per factor (max S/N for "SN", max for response too)
        optimal_per_factor = {
            f: e.level_means.idxmax() for f, e in self.effects.items()
        }

        for i, (ax, factor) in enumerate(zip(axes, self.factors)):
            effect = self.effects[factor]
            x = list(range(1, len(effect.level_means) + 1))
            x_labels = [str(lvl) for lvl in effect.level_means.index]
            y = effect.level_means.values

            # Main line — thin black with open circles
            ax.plot(
                x, y,
                color="black",
                linewidth=1.0,
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=3,
            )

            # Highlight optimum with a filled black circle
            if highlight_optimum:
                opt_level = optimal_per_factor[factor]
                opt_idx = list(effect.level_means.index).index(opt_level)
                ax.plot(
                    x[opt_idx], y[opt_idx],
                    marker="o",
                    markersize=8,
                    markerfacecolor="black",
                    markeredgecolor="black",
                    zorder=4,
                )

            # Grand mean reference line (horizontal dashed)
            if show_grand_mean:
                ax.axhline(
                    grand_mean,
                    linestyle=":",
                    linewidth=0.8,
                    color="gray",
                )

            # Dashed grid — typical of journal figures
            ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
            ax.set_axisbelow(True)

            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(factor)  # factor name on x-axis (matches journal style)
            ax.set_xlim(0.5, len(x) + 0.5)

            # Subtle vertical separator hint via spine emphasis
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

            # Only the leftmost subplot shows the y-axis label
            if i == 0:
                ax.set_ylabel(y_label_text)
            ax.tick_params(direction="in", length=4)

        # Tight layout with small horizontal spacing → looks like one continuous figure
        fig.subplots_adjust(wspace=0.08)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ........................................................ modern style
    def _plot_modern_style(
        self,
        on: str,
        figsize: tuple[float, float] | None,
        save_path: str | Path | None,
        show: bool,
        ylabel: str | None,
        show_grand_mean: bool,
    ) -> plt.Figure:
        """Colourful, dashboard-friendly Main Effects Plot (legacy style)."""
        n = len(self.factors)
        figsize = figsize or (max(4 * n, 6), 4)
        fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
        if n == 1:
            axes = [axes]

        grand_mean = float(self.sn_data[on].mean())
        default_ylabel = "Mean S/N Ratio (dB)" if on == "SN" else f"Mean of {on}"
        y_label_text = ylabel if ylabel is not None else default_ylabel

        for ax, factor in zip(axes, self.factors):
            effect = self.effects[factor]
            ax.plot(
                effect.level_means.index.astype(str),
                effect.level_means.values,
                marker="o",
                linewidth=2,
            )
            if show_grand_mean:
                ax.axhline(grand_mean, linestyle="--", linewidth=1, color="gray", alpha=0.7)
            ax.set_title(f"{factor}  (Δ = {effect.delta:.3f}, rank {effect.rank})")
            ax.set_xlabel("Level")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(y_label_text)
        title = "Main Effects Plot for S/N Ratios" if on == "SN" else "Main Effects Plot (Mean Response)"
        fig.suptitle(f"{title}  —  criterion: {self.criterion}-the-better", y=1.02)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ----------------------------------------------------- pretty summaries
    def summary(self) -> str:
        """Return a human-readable text summary of the analysis.

        Includes the S/N criterion used, the per-factor delta and rank table,
        and the recommended optimal level combination.
        """
        if not self.effects:
            self.compute_main_effects()

        lines: list[str] = []
        lines.append("=" * 64)
        lines.append("Taguchi DoE Analysis Summary")
        lines.append("=" * 64)
        lines.append(f"  S/N criterion : {self.criterion}-the-better")
        lines.append(f"  Factors       : {', '.join(self.factors)}")
        lines.append(f"  Runs          : {len(self.sn_data) if self.sn_data is not None else 0}")
        lines.append("")

        lines.append("Response table (mean S/N per level):")
        for factor, effect in sorted(self.effects.items(), key=lambda kv: kv[1].rank):
            lines.append(f"  [{factor}]  delta={effect.delta:.4f}  rank={effect.rank}")
            for level, value in effect.level_means.items():
                lines.append(f"      level {level!s:<6} -> {value: .4f}")

        lines.append("")
        lines.append("Optimal level combination (max S/N):")
        for factor, level in self.optimal_levels().items():
            lines.append(f"  {factor} = {level}")

        if self.anova is not None:
            lines.append("")
            lines.append("ANOVA on S/N (simplified):")
            lines.append(self.anova.table.to_string(index=False, float_format="%.4f"))

        return "\n".join(lines)
