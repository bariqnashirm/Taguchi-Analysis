"""
dashboard.py
============

Interactive web dashboard for Taguchi-DoE-Toolkit, built with Streamlit.

Run from the repository root with:

    streamlit run dashboard.py

The dashboard lets the user:
    1. Upload a CSV file (or use the bundled L9 demo).
    2. Select factor columns, replicate columns, and S/N criterion.
    3. View the S/N table, main-effects response table, ANOVA table,
       optimal level combination, and the Main Effects Plot — all updated
       reactively.
    4. Download the resulting plot and a summary report.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make the local `src/` package importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from taguchi_toolkit import TaguchiAnalyzer  # noqa: E402


# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Taguchi-DoE-Toolkit Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Taguchi-DoE-Toolkit Dashboard")
st.caption(
    "Interactive analysis of Taguchi Method experiments — "
    "upload your data, configure the analysis, and explore the results."
)


# ---------------------------------------------------------------------------
# Sidebar — data input & configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # ---------- Data source ----------
    st.subheader("1. Data Source")
    data_source = st.radio(
        "Choose data source",
        options=["Use bundled L9 demo", "Upload my own CSV"],
        index=0,
        label_visibility="collapsed",
    )

    df: pd.DataFrame | None = None

    if data_source == "Use bundled L9 demo":
        demo_path = REPO_ROOT / "data" / "example_L9.csv"
        if demo_path.exists():
            df = pd.read_csv(demo_path)
            st.success(f"Loaded demo dataset ({len(df)} runs).")
        else:
            st.error(f"Demo file not found at {demo_path}. Use the upload option.")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success(f"Uploaded: {uploaded.name} ({len(df)} runs).")

    # ---------- Column mapping ----------
    factors: list[str] = []
    replicate_cols: list[str] = []
    criterion = "larger"
    target_value: float | None = None

    if df is not None:
        st.subheader("2. Column Mapping")
        all_cols = list(df.columns)

        # Heuristic defaults: numeric columns whose name starts with "y" or contains digits
        # are likely replicates; the rest are likely factors.
        default_replicates = [
            c for c in all_cols
            if c.lower().startswith(("y", "rep", "trial", "obs"))
        ]
        default_factors = [c for c in all_cols if c not in default_replicates and c.lower() != "run"]

        factors = st.multiselect(
            "Factor columns",
            options=all_cols,
            default=default_factors,
            help="Control parameters whose levels you varied (e.g. speed, feed, depth).",
        )
        replicate_cols = st.multiselect(
            "Replicate response columns",
            options=[c for c in all_cols if c not in factors],
            default=[c for c in default_replicates if c not in factors],
            help="One column per replicate measurement (e.g. y1, y2, y3).",
        )

        # ---------- S/N criterion ----------
        st.subheader("3. S/N Criterion")
        criterion_label = st.radio(
            "How should the response be optimised?",
            options=[
                "Larger-the-better (maximise)",
                "Smaller-the-better (minimise)",
                "Nominal-the-best (target value)",
            ],
            index=0,
        )
        criterion_map = {
            "Larger-the-better (maximise)": "larger",
            "Smaller-the-better (minimise)": "smaller",
            "Nominal-the-best (target value)": "nominal",
        }
        criterion = criterion_map[criterion_label]

        if criterion == "nominal":
            use_target = st.checkbox(
                "Specify a target value",
                value=False,
                help=(
                    "If unchecked, uses the classical formula 10·log10(μ²/σ²). "
                    "If checked, uses -10·log10(MSD) with the given target."
                ),
            )
            if use_target:
                target_value = st.number_input("Target value", value=0.0, format="%.4f")

    # ---------- Run analysis ----------
    st.subheader("4. Run Analysis")
    run_button = st.button("▶️  Analyse", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

def _validation_errors(
    df: pd.DataFrame | None,
    factors: list[str],
    replicate_cols: list[str],
) -> list[str]:
    """Return a list of human-readable validation errors (empty list = OK)."""
    errors: list[str] = []
    if df is None:
        errors.append("No data loaded — pick the demo or upload a CSV in the sidebar.")
        return errors
    if not factors:
        errors.append("Select at least one factor column.")
    if not replicate_cols:
        errors.append("Select at least one replicate column.")
    overlap = set(factors) & set(replicate_cols)
    if overlap:
        errors.append(f"These columns appear in both factors and replicates: {sorted(overlap)}")
    if len(replicate_cols) < 2:
        errors.append("At least 2 replicate columns are recommended for a meaningful S/N.")
    # Replicates must be numeric
    if df is not None and replicate_cols:
        non_numeric = [c for c in replicate_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            errors.append(f"These replicate columns are not numeric: {non_numeric}")
    return errors


# Show data preview as soon as data is loaded — even before running analysis
if df is not None:
    with st.expander("👁️  Data preview", expanded=False):
        st.dataframe(df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", df.shape[1])
        col3.metric("Numeric columns", df.select_dtypes(include="number").shape[1])

# Run analysis
if run_button:
    errors = _validation_errors(df, factors, replicate_cols)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    # Build & run the analyzer
    try:
        analyzer = TaguchiAnalyzer(
            factors=factors,
            response="response",  # logical name only
            replicate_columns=replicate_cols,
            criterion=criterion,
            target=target_value,
        )
        analyzer.load_dataframe(df)
        sn_table = analyzer.compute_sn_ratio()
        effects = analyzer.compute_main_effects()
        anova_result = analyzer.run_anova()
        optima = analyzer.optimal_levels()
    except Exception as exc:  # surface any analytical error to the user
        st.error(f"Analysis failed: {exc}")
        st.stop()

    # ----- Top-level KPIs -----
    st.subheader("🎯 Optimal Level Combination")
    kpi_cols = st.columns(len(optima))
    for col, (factor, level) in zip(kpi_cols, optima.items()):
        rank = effects[factor].rank
        delta = effects[factor].delta
        col.metric(
            label=f"{factor}  (rank {rank})",
            value=f"Level {level}",
            delta=f"Δ = {delta:.3f}",
            delta_color="off",
        )
    st.caption(
        f"Ranking is based on Δ (range of mean S/N across levels). "
        f"S/N criterion: **{criterion}-the-better**."
    )

    # ----- Tabs for the four result blocks -----
    tab_sn, tab_effects, tab_anova, tab_plot = st.tabs(
        ["S/N Table", "Main Effects", "ANOVA", "Main Effects Plot"]
    )

    # ---- Tab 1: per-run S/N table
    with tab_sn:
        st.markdown("Per-run mean response and S/N ratio:")
        st.dataframe(
            sn_table.style.format({"mean_response": "{:.4f}", "SN": "{:.4f}"}),
            use_container_width=True,
        )
        csv_buf = sn_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download S/N table (CSV)",
            data=csv_buf,
            file_name="sn_table.csv",
            mime="text/csv",
        )

    # ---- Tab 2: main-effects response table
    with tab_effects:
        st.markdown("Mean S/N ratio per level of each factor (sorted by influence):")
        # Build a wide-format response table
        response_rows: list[dict] = []
        max_levels = max(len(e.level_means) for e in effects.values())
        for factor, effect in sorted(effects.items(), key=lambda kv: kv[1].rank):
            row: dict[str, object] = {
                "Factor": factor,
                "Rank": effect.rank,
                "Delta": round(effect.delta, 4),
            }
            for i, (level, value) in enumerate(effect.level_means.items(), start=1):
                row[f"Level {level}"] = round(float(value), 4)
            response_rows.append(row)
        response_df = pd.DataFrame(response_rows)
        st.dataframe(response_df, use_container_width=True)

        # Bar chart of deltas (a quick visual of factor influence)
        st.markdown("Factor influence (Δ):")
        delta_series = pd.Series(
            {f: e.delta for f, e in effects.items()}
        ).sort_values(ascending=False)
        st.bar_chart(delta_series, use_container_width=True)

    # ---- Tab 3: ANOVA
    with tab_anova:
        st.markdown("Simplified one-way ANOVA on S/N ratios:")
        st.dataframe(
            anova_result.table.style.format(
                {
                    "DoF": "{:.0f}",
                    "SS": "{:.4f}",
                    "MS": "{:.4f}",
                    "F": "{:.4f}",
                    "Contribution_%": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
        col_a, col_b = st.columns(2)
        col_a.metric("Grand mean of S/N", f"{anova_result.grand_mean:.4f}")
        col_b.metric("Total SS", f"{anova_result.total_ss:.4f}")

        # Contribution pie chart (factors only, exclude Total)
        contrib_df = anova_result.table[
            ~anova_result.table["Factor"].isin(["Total"])
        ].set_index("Factor")["Contribution_%"]
        st.markdown("Contribution to S/N variation:")
        st.bar_chart(contrib_df, use_container_width=True)

    # ---- Tab 4: Main Effects Plot
    with tab_plot:
        col_style, col_label, col_opt = st.columns([1, 1.5, 1])
        with col_style:
            plot_style = st.radio(
                "Plot style",
                options=["journal", "modern"],
                index=0,
                help=(
                    "**journal** — black-and-white publication style "
                    "(filled marker = optimum level).\n\n"
                    "**modern** — colourful dashboard style with delta info."
                ),
            )
        with col_label:
            custom_ylabel = st.text_input(
                "Y-axis label (optional)",
                value="",
                placeholder="e.g. Ra (μm), PD (g), Strength (MPa)",
                help="Leave empty to auto-generate (e.g. 'Mean S/N Ratio (dB)').",
            )
        with col_opt:
            highlight_opt = st.checkbox(
                "Highlight optimum",
                value=True,
                help="Filled circle marks the level with maximum S/N (journal style only).",
            )

        st.markdown("**Main Effects Plot for S/N Ratios:**")
        fig = analyzer.plot_main_effects(
            show=False,
            style=plot_style,
            ylabel=custom_ylabel.strip() or None,
            highlight_optimum=highlight_opt,
        )
        st.pyplot(fig, use_container_width=True)

        # Allow PNG download
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        png_buf.seek(0)
        st.download_button(
            "⬇️  Download plot (PNG, 300 DPI)",
            data=png_buf,
            file_name=f"main_effects_plot_{plot_style}.png",
            mime="image/png",
        )
        plt.close(fig)

    # ----- Text summary at the bottom -----
    with st.expander("📝 Text summary (also downloadable)", expanded=False):
        summary_text = analyzer.summary()
        st.code(summary_text, language="text")
        st.download_button(
            "⬇️  Download summary (TXT)",
            data=summary_text.encode("utf-8"),
            file_name="taguchi_summary.txt",
            mime="text/plain",
        )

else:
    # Idle state — invite the user
    st.info(
        "👈 Configure your analysis in the sidebar and click **Analyse** to begin.\n\n"
        "Don't have data handy? Pick **Use bundled L9 demo** to see the dashboard in action."
    )

    with st.expander("ℹ️  About the three S/N criteria"):
        st.markdown(
            """
| Criterion | Use when… | Formula |
|---|---|---|
| **Larger-the-better** | The response should be **maximised** (e.g. tensile strength, MRR, yield). | `-10·log₁₀(mean(1/yᵢ²))` |
| **Smaller-the-better** | The response should be **minimised** (e.g. roughness, defects, wear). | `-10·log₁₀(mean(yᵢ²))` |
| **Nominal-the-best** | A specific **target value** is best (e.g. dimension on a tolerance band). | `10·log₁₀(μ²/σ²)` or `-10·log₁₀(MSD)` with target |

In all three cases, the optimum is the level combination that **maximises the S/N ratio**.
"""
        )
