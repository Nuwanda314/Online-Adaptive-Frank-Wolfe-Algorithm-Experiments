# An Online Domain and Objective Adaptive (Stochastic) Frank-Wolfe Algorithm

## Abstract: 
We investigate a stochastic online variant of the classical Frank-Wolfe algorithm for minimizing a convex, differentiable objective function over a convex and compact domain. Unlike the traditional setting, we assume that both the objective function and the feasible domain are initially unknown and must be learned from data. To address this, we integrate statistical estimators into the optimization process, allowing the algorithm to iteratively refine approximations of the domain and the objective function. Our approach maintains the projection-free nature of Frank-Wolfe while adapting to the uncertainty inherent in data-driven settings. We establish convergence guarantees for the online method, showing that the optimization error scales with the accuracy of the learned estimators. Extensive experiments support our theoretical findings, demonstrating that the proposed method achieves convergence behavior comparable to classical Frank-Wolfe in scenarios with exact knowledge of domain and objective function.

## Repository Structure
- `main.jl`: entry point for running experiments
- `structures.jl`: problem-related data structures
- `objective_function.jl`: objective function definitions
- `domain_approximation.jl`: domain estimation / approximation routines
- `oracles.jl`: linear minimization oracle and related components
- `analyzing_procedure.jl`: postprocessing and evaluation
- `plot_generation.jl`: generation of figures
- `figures/`: output plots and visualizations

## Requirements
- Julia version
- package dependencies

## Installation
Schritte zum Aktivieren des Environments und Installieren der Pakete.

## How to Run
Konkrete Befehle:
- Experimente starten
- Auswertung erzeugen
- Plots erzeugen

## Reproducibility
Erklären:
- welche Datei welchen Teil des Papers reproduziert
- wo Parameter eingestellt werden
- wo Outputs landen

## Results
Optional 1–2 Abbildungen mit kurzer Erklärung.

## Citation
BibTeX oder Paper-Zitat.

## Notes
Optional: work in progress, planned extensions, numerical caveats.
