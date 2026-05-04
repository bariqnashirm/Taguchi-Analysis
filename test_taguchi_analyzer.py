"""
Unit tests for the TaguchiAnalyzer class.

Run with:
    pytest -q
"""

import math

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # headless backend for CI

from taguchi_toolkit import TaguchiAnalyzer


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def l9_dataframe() -> pd.DataFrame:
    """A small balanced L9 dataset (3 factors x 3 levels, 3 replicates)."""
    return pd.DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "C": [1, 2, 3, 2, 3, 1, 3, 1, 2],
            "y1": [42.1, 45.6, 38.2, 52.3, 49.8, 47.4, 55.1, 58.7, 53.5],
            "y2": [41.8, 46.0, 38.6, 51.9, 50.1, 47.0, 55.6, 59.2, 53.0],
            "y3": [42.3, 45.4, 37.9, 52.7, 49.5, 47.7, 54.8, 58.4, 53.8],
        }
    )


@pytest.fixture
def analyzer_larger(l9_dataframe):
    a = TaguchiAnalyzer(
        factors=["A", "B", "C"],
        response="y",
        replicate_columns=["y1", "y2", "y3"],
        criterion="larger",
    )
    a.load_dataframe(l9_dataframe)
    return a


# ------------------------------------------------------------------- S/N math

def test_sn_larger_known_value():
    """Verify the larger-the-better formula against a hand-computed value."""
    values = np.array([10.0, 12.0, 11.0])
    expected = -10.0 * math.log10(np.mean(1.0 / (values ** 2)))
    assert TaguchiAnalyzer._sn_larger_the_better(values) == pytest.approx(expected)


def test_sn_smaller_known_value():
    values = np.array([1.0, 2.0, 3.0])
    expected = -10.0 * math.log10(np.mean(values ** 2))
    assert TaguchiAnalyzer._sn_smaller_the_better(values) == pytest.approx(expected)


def test_sn_nominal_classical():
    values = np.array([10.0, 11.0, 9.0])  # mean=10, var(ddof=0) = 2/3
    expected = 10.0 * math.log10((10.0 ** 2) / (2.0 / 3.0))
    assert TaguchiAnalyzer._sn_nominal_the_best(values) == pytest.approx(expected)


def test_sn_nominal_with_target():
    values = np.array([9.0, 10.0, 11.0])
    target = 10.0
    msd = float(np.mean((values - target) ** 2))
    expected = -10.0 * math.log10(msd)
    assert TaguchiAnalyzer._sn_nominal_the_best(values, target=target) == pytest.approx(expected)


def test_sn_larger_rejects_nonpositive():
    with pytest.raises(ValueError):
        TaguchiAnalyzer._sn_larger_the_better(np.array([1.0, 0.0, 2.0]))


# -------------------------------------------------------------------- pipeline

def test_compute_sn_ratio_shape(analyzer_larger):
    sn = analyzer_larger.compute_sn_ratio()
    assert {"A", "B", "C", "mean_response", "SN"}.issubset(sn.columns)
    assert len(sn) == 9
    assert sn["SN"].notna().all()


def test_main_effects_and_ranking(analyzer_larger):
    analyzer_larger.compute_sn_ratio()
    effects = analyzer_larger.compute_main_effects()
    assert set(effects.keys()) == {"A", "B", "C"}
    # Ranks must be a permutation of 1..n
    ranks = sorted(e.rank for e in effects.values())
    assert ranks == [1, 2, 3]
    # Delta must be non-negative
    for e in effects.values():
        assert e.delta >= 0


def test_optimal_levels_returns_value_per_factor(analyzer_larger):
    analyzer_larger.compute_sn_ratio()
    optima = analyzer_larger.optimal_levels()
    assert set(optima.keys()) == {"A", "B", "C"}


def test_anova_table_shape(analyzer_larger):
    analyzer_larger.compute_sn_ratio()
    result = analyzer_larger.run_anova()
    table = result.table
    # 3 factors + Error + Total = 5 rows
    assert len(table) == 5
    assert set(table.columns) == {"Factor", "DoF", "SS", "MS", "F", "Contribution_%"}
    # Total SS should equal sum of factor SS + error SS
    factor_rows = table[~table["Factor"].isin(["Error", "Total"])]
    error_ss = float(table.loc[table["Factor"] == "Error", "SS"].iloc[0])
    total_ss = float(table.loc[table["Factor"] == "Total", "SS"].iloc[0])
    assert factor_rows["SS"].sum() + error_ss == pytest.approx(total_ss, rel=1e-9)


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        TaguchiAnalyzer(factors=["A"], response="y", criterion="medium")


def test_missing_columns_raises(l9_dataframe):
    a = TaguchiAnalyzer(
        factors=["A", "B", "Z_nonexistent"],
        response="y",
        replicate_columns=["y1", "y2", "y3"],
        criterion="larger",
    )
    with pytest.raises(ValueError):
        a.load_dataframe(l9_dataframe)


def test_plot_runs_headless(analyzer_larger, tmp_path):
    analyzer_larger.compute_sn_ratio()
    out = tmp_path / "plot.png"
    fig = analyzer_larger.plot_main_effects(save_path=out, show=False)
    assert out.exists()
    assert fig is not None
