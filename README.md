# Taguchi-DoE-Toolkit

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green)](tests/)

> Lightweight, open-source Python toolkit for analyzing **Taguchi Method Design of Experiments (DoE)** — built for mechanical engineers, researchers, and students.

The Taguchi Method is a robust statistical approach for optimizing process parameters with a minimum number of experimental runs. This toolkit automates the most repetitive parts of the analysis: **S/N ratio calculation, main-effects analysis, simplified ANOVA, and visualization**, so you can focus on the engineering decisions, not the spreadsheet gymnastics.

---

## ✨ Features

- 📂 **CSV loader** — read experimental data with arbitrary factor and replicate columns.
- 📊 **Three S/N criteria** — Larger-the-better, Smaller-the-better, Nominal-the-best (with optional target).
- 🏆 **Main effects analysis** — mean response per level, delta, and automatic factor ranking.
- 📈 **Simplified ANOVA** — sum of squares, degrees of freedom, F-ratio, and percentage contribution per factor.
- 🎨 **Main Effects Plot** — auto-generated multi-subplot figure with grand-mean reference line.
- 🧱 **Clean OOP design** — single `TaguchiAnalyzer` class, fully type-hinted and docstring-documented.
- ✅ **Tested** — pytest suite covering math, pipeline, edge cases, and headless plotting.

---

## 📁 Project Structure

```
Taguchi-DoE-Toolkit/
├── src/
│   └── taguchi_toolkit/
│       ├── __init__.py
│       └── taguchi_analyzer.py     # Core class: TaguchiAnalyzer
├── data/
│   └── example_L9.csv              # Bundled L9 demo dataset
├── examples/
│   └── example_run.py              # End-to-end usage script
├── tests/
│   └── test_taguchi_analyzer.py    # Pytest unit tests
├── notebooks/                      # (optional) Jupyter walkthroughs
├── docs/                           # (optional) extended documentation
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### From source (development)

```bash
git clone https://github.com/<your-username>/Taguchi-DoE-Toolkit.git
cd Taguchi-DoE-Toolkit
pip install -e .
```

### Or just install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy, Pandas, Matplotlib.

---

## ⚡ Quick Start

```python
from taguchi_toolkit import TaguchiAnalyzer

analyzer = TaguchiAnalyzer(
    factors=["A_speed", "B_feed", "C_depth"],
    response="strength",
    replicate_columns=["y1", "y2", "y3"],
    criterion="larger",          # "larger" | "smaller" | "nominal"
)

analyzer.load_data("data/example_L9.csv")
analyzer.compute_sn_ratio()
analyzer.compute_main_effects()
analyzer.run_anova()

print(analyzer.summary())
analyzer.plot_main_effects(save_path="main_effects.png")
```

Run the bundled demo end-to-end:

```bash
python examples/example_run.py
```

---

## 📐 The Three S/N Criteria

| Criterion | Use when… | Formula |
|---|---|---|
| **Larger-the-better** | The response should be maximized (e.g. tensile strength, yield) | `-10·log₁₀(mean(1/yᵢ²))` |
| **Smaller-the-better** | The response should be minimized (e.g. surface roughness, defects) | `-10·log₁₀(mean(yᵢ²))` |
| **Nominal-the-best** | A specific target value is best (e.g. dimension on a tolerance band) | `10·log₁₀(μ²/σ²)` or `-10·log₁₀(MSD)` if a target is given |

In all three cases, **maximizing the S/N ratio** identifies the optimal level combination.

---

## 🔬 Example Workflow

The bundled `example_L9.csv` simulates a turning experiment with 3 factors at 3 levels each, 3 replicates per run, and "strength" as the response (larger = better). Running the demo prints the response table, ranks the factors by influence, recommends the optimal level for each factor, and saves the main-effects plot.

---

## 🧪 Running Tests

```bash
pip install pytest
pytest -q
```

---

## 🗺️ Roadmap

- [ ] Built-in catalogue of standard orthogonal arrays (L4, L8, L9, L16, L18, L27)
- [ ] Two-way interaction plots
- [ ] Confidence intervals and confirmation-run prediction
- [ ] Pareto chart of factor contributions
- [ ] CLI entry point (`taguchi analyze data.csv ...`)
- [ ] Export of results to Excel / Markdown report

---

## 🤝 Contributing

Contributions are very welcome! If you have an improvement, bug fix, or new feature in mind:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-improvement`.
3. Add tests for any new behavior.
4. Make sure `pytest` passes.
5. Submit a Pull Request describing your change.

For larger changes, please open an issue first to discuss the design.

---

## 📚 References

- Taguchi, G., Chowdhury, S., & Wu, Y. (2005). *Taguchi's Quality Engineering Handbook*. Wiley.
- Roy, R. K. (2010). *A Primer on the Taguchi Method* (2nd ed.). SME.
- Phadke, M. S. (1995). *Quality Engineering Using Robust Design*. Prentice Hall.

---

## 📄 License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.
