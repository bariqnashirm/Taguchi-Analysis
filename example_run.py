"""
example_run.py
==============

End-to-end demo of the Taguchi-DoE-Toolkit using the bundled L9 example
(3 control factors, 3 levels each, 3 replicates per run).

Scenario:
    A turning experiment where we want to MAXIMISE the surface tensile
    strength (MPa) of a workpiece. Larger-the-better S/N is the right
    criterion.

Run from the repository root:

    python examples/example_run.py
"""

from pathlib import Path

from taguchi_toolkit import TaguchiAnalyzer


def main() -> None:
    # Path resolution — works whether you run from repo root or examples/
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "data" / "example_L9.csv"

    # 1. Configure the analyzer
    analyzer = TaguchiAnalyzer(
        factors=["A_speed", "B_feed", "C_depth"],
        response="strength",          # logical name; replicates carry the data
        replicate_columns=["y1", "y2", "y3"],
        criterion="larger",            # larger-the-better
    )

    # 2. Load the data
    analyzer.load_data(csv_path)
    print(f"Loaded {len(analyzer.data)} runs from {csv_path.name}\n")

    # 3. Compute S/N ratios
    sn_table = analyzer.compute_sn_ratio()
    print("S/N table:")
    print(sn_table.to_string(index=False, float_format="%.4f"))
    print()

    # 4. Main effects + ANOVA
    analyzer.compute_main_effects()
    analyzer.run_anova()

    # 5. Pretty summary (factors ranked by influence + optimal levels)
    print(analyzer.summary())

    # 6. Plot — saved to disk so the demo also works headless
    output_path = repo_root / "main_effects_plot.png"
    analyzer.plot_main_effects(save_path=output_path, show=False)
    print(f"\nMain-effects plot saved to: {output_path}")


if __name__ == "__main__":
    main()
