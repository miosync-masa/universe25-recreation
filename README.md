# Universe 25: Computational Recreation of Calhoun's Mouse Utopia Collapse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17538843.svg)](https://doi.org/10.5281/zenodo.17538843)

> **"The most perfect world for mice became their gravest tomb."** — John B. Calhoun (1973)

A computational model that reproduces the catastrophic collapse of John B. Calhoun's infamous **Universe 25** experiment (1968-1973), where a mouse population in utopia—unlimited food, water, optimal temperature, no predators—collapsed to extinction despite ideal conditions.

## 🔬 The Mystery

In 1968, ethologist John B. Calhoun created a mouse paradise: a 9×9 foot enclosure with unlimited resources. Starting with 4 breeding pairs, the population grew to ~2,200 mice, then **catastrophically collapsed to extinction** by Day 1,780.

The most haunting phenomenon: **Beautiful Ones**—mice who withdrew from society, ceased reproduction, groomed obsessively, yet survived for extended periods. They were "socially dead while biologically alive."

**For 50 years, this paradox resisted explanation.**

## 💡 Our Discovery: Genome's Adaptive Trap

We demonstrate that Beautiful Ones emergence is not pathology—it's **rational adaptation gone wrong**.

Using a 48×48 cellular automaton where individuals carry 24-bit genomes encoding stress-dependent behavioral strategies, we reveal:

### Key Findings

- **Collapse emerges from individual rationality**: Under high-density stress (ρ > 0.65), genomes rationally shift from cooperation-reproduction to self-preservation modules
- **Epigenetic lock-in**: Once activated, these behavioral patterns persist even when density decreases—creating a "Genome's Adaptive Trap"
- **Quantitative precision**: Our model reproduces Universe 25 dynamics with 98.4% accuracy:
  - Dysfunction period: **67.4%** (vs. 68.5% observed)
  - Social death → extinction delay: **τ_delay = 11.7 ± 24.7 steps** (bimodal distribution)
  - Beautiful Ones emerge in **three sequential types**: strategic → survivor → dying

### Death in Paradise

At extinction, survivors occupy low-resource zones while abundant food sits unused—demonstrating that **collapse was social, not environmental**.

> **"Food hoppers remained half-full; the mice simply stopped eating together."** — Calhoun (1973)

## 📊 Results Visualization

### Lineage Tree: Collapse Choreography
<img width="490" alt="Image" src="https://github.com/user-attachments/assets/53daa82c-de32-4dfe-9b0e-a0ad96877c58" />

### Five-Phase Structure
- **Phase 1 (7.1%)**: Growth — Exponential expansion
- **Phase 2 (7.4%)**: Prosperity — Peak population, BO strategic emerge
- **Phase 3 (18.1%)**: Collapse — Reproduction crashes, BO dying surge
- **Phase 4 (22.7%)**: Zombie State — Social death, metabolic persistence
- **Phase 5 (44.7%)**: Extinction — Moribund survivors, final collapse

## 🚀 Quick Start

### Requirements
```bash
Python 3.8+
numpy >= 1.26.0
matplotlib >= 3.8.0
pandas >= 2.1.0
```

### Installation
```bash
git clone https://github.com/miosync/universe25-recreation.git
cd universe25-recreation
pip install -r requirements.txt
```

### Run Single Simulation
```python
from self_evolving_automaton import SimConfig, simulate_once

# Configure Universe 25 parameters
cfg = SimConfig(
    GRID_H=48, GRID_W=48,
    STEPS=365,
    SEED_DENSITY=0.08,
    CRIT_DENSITY=0.65,  # Behavioral sink threshold
    BASE_L_REPL=0.95,
    BASE_L_EVOL=1.10
)

# Run simulation
result = simulate_once(cfg, seed=6000, save_prefix="single_run")
```

**Output**: All results saved to `./outputs/U25_FULL_single_run/`
- Time series data (CSV)
- Spatial heatmaps (PDF)
- Lineage tree visualization (PNG)
- Critical events log (CSV)

### Run Ensemble (N=10)
```python
from self_evolving_automaton import run_ensemble

out_dir = run_ensemble(
    cfg, 
    N=10, 
    base_seed=6000, 
    do_events=True,
    tag_suffix="ensemble"
)

print(f"Results: {out_dir}")
```

**Outputs**:
- Aggregated statistics with 95% CI
- Event-aligned analysis around BO emergence
- τ_delay distribution across runs

## 🎛️ Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Grid size | H × W | 48 × 48 | ~2,304 cells ≈ Universe 25 capacity |
| Time steps | T | 365 | Sufficient for collapse observation |
| Critical density | ρ_crit | 0.65 | Behavioral sink threshold (Calhoun, 1973) |
| Harvest rate | h | 0.20 | Energy equilibrium below ρ_crit |
| Metabolism | m | 0.015 | ~200-step lifespan (no stress) |
| Density stress weight | w_K,ρ | 0.80 | Dominant stress factor |

**Λ Calculation** (multi-constraint stress):
```
Λ = K / |V|_eff

where:
  K = Constraint pressure (density, mismatch, energy deficit, cooperation loss)
  |V| = Adaptive capacity (fitness, social support, resources)
```

## 📁 Repository Structure
```
universe25-recreation/
├── self_evolving_automaton.py    # Main simulation engine
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
├── outputs/                      # Generated data (gitignored)
└── docs/
    └── paper.pdf                 # Full manuscript
```

## 🔬 Scientific Validation

### Quantitative Comparison

| Metric | Universe 25 (1973) | This Study | Agreement |
|--------|-------------------|-----------|-----------|
| Peak population | ~2,200 (Day 560) | 0.42 alive fraction | ✓ |
| Dysfunction period | 68.5% | 67.4% | **98.4%** |
| Social death timing | Day ~560 (30%) | Step 119 (32.6%) | **92.0%** |
| Critical density | ~65-70% capacity | ρ = 0.65 | **95%+** |

### Counterfactual Experiments

We tested interventions to determine if collapse was preventable:

- ❌ **Spatial expansion**: Delays but doesn't prevent
- ❌ **Resource superabundance**: Accelerates collapse (!)
- ⚠️ **Wealth redistribution**: Delays by 12 steps, still collapses
- ⚠️ **Cooperation reset**: Extends survival +45%, still collapses
- ✅ **Emigration valve**: **Prevents collapse indefinitely**

**Conclusion**: Collapse was structurally determined by the closed-system constraint. Only emigration—unavailable to Calhoun's mice—prevents trajectory.

## 🧬 Model Architecture

### Constraint-Based Design

Unlike traditional agent-based models, we implement **hierarchical constraint satisfaction**:

1. **Energy constraint**: Maintain E_int > 0 (or die)
2. **Social constraint**: Cooperate fairly (via Prisoner's Dilemma)
3. **Reproduction constraint**: Accumulate E > threshold
4. **Density constraint**: Tolerate crowding (or suffer stress)

Individuals attempt to satisfy all constraints simultaneously. When impossible (Λ > Λ_evol), genomes shift to self-preservation mode.

### Emergent, Not Programmed

**Critical verification**: We did not program "Beautiful One" as a category. The model contains:
- Energy budgets
- Cooperation decisions (Prisoner's Dilemma)
- Reproduction thresholds
- Stress calculations

Beautiful One detection is **post-hoc classification**:
- BO strategic: `coop < 0.4` and `E_int > 0.25` (high energy, withdrawn)
- BO survivor: `coop < 0.4` and `0.10 < E_int ≤ 0.25` (medium energy)
- BO dying: `coop < 0.4` and `E_int ≤ 0.10` (low energy, moribund)

In Phase 1-2: BO fraction ≈ 0-5%  
In Phase 3-5: BO fraction → 70-85%

This transition **emerges** from density stress, not programmed triggers.

## 📖 Citation

If you use this code or findings in your research, please cite:
```bibtex
@article{iizumi2025universe25,
  title={Genome's adaptive trap: Computational recreation of Universe 25 reveals epigenetic lock-in as the mechanism of societal collapse},
  author={Iizumi, Masamichi},
  journal={In preparation},
  year={2025},
  doi={10.5281/zenodo.17538843}
}
```

## 🔗 Related Resources

- **Paper (preprint)**: [docs/paper.pdf](docs/paper.pdf)
- **Zenodo Archive**: [10.5281/zenodo.17538843](https://doi.org/10.5281/zenodo.17538843)
- **Original Universe 25**: Calhoun, J.B. (1973). *Death squared: The explosive growth and demise of a mouse population*. Proc. Royal Society of Medicine, 66(1):80-88.

## 🤝 Contributing

Contributions welcome! This model is designed as an **experimental platform** for testing collapse interventions. Potential extensions:

- Parameter space mapping
- Evolutionary experiments (multi-generational adaptation)
- Network topology effects (scale-free, small-world)
- Hybrid LLM-based agents with meta-cognition

Please open an issue or pull request.

## ⚖️ License

MIT License - see [LICENSE](LICENSE) file for details.

**Code**: Open source, freely modifiable  
**Data**: CC BY 4.0

## ⚠️ Ethical Considerations

This is a **computational study** with no animal subjects. However, given Universe 25's legacy:

1. **Model scope**: Simulates mouse social dynamics under extreme density. Extrapolation to human societies requires additional mechanisms (language, institutions, technology).

2. **Determinism vs. agency**: Our model assumes constraint-driven behavior. Human cognition includes meta-awareness, cultural transmission, and institutional design—capabilities that may enable escape from "adaptive traps."

3. **Responsible interpretation**: Results should **not** be weaponized to justify population control, social Darwinism, or fatalistic attitudes. Collapse is emergent, not inevitable—it depends on constraint structure, which human societies can redesign.

---

## 🙏 Acknowledgments

This work honors:
- **John B. Calhoun** (1917-1995): For Universe 25 and lifelong work on behavioral sinks
- **John von Neumann**: Self-reproducing automata theory
- **John Horton Conway**: Game of Life
- **Stephen Wolfram**: Cellular automaton complexity

Built with: NumPy, Pandas, Matplotlib, Jupyter

---

**"The question is no longer 'Why did Universe 25 collapse?' The question is: 'Can any system avoid it?'"**

---

*Developed by Miosync Inc. | Contact: correspondingauthor@miosync.com*
