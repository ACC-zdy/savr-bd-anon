# SaVR-BD: Saturation-Aware Variance-Reduced Barker Dynamics

Anonymous code release accompanying a double-blind submission.

SaVR-BD is a variance-reduced stochastic gradient Barker sampler for Bayesian posterior inference over large tables. It couples SVRG or SAGA control variates with an unbiased finite-population variance estimator and a smooth curvature-matched calibration map.

## Layout

- `src/savr_bd.py`: SaVR-BD (SVRG and SAGA variants) and the Barker baselines (v-SGBD, c-SGBD, sc-SGBD, NaiveVR, HardGate, UniVR).
- `src/sgld_baselines.py`: Langevin baselines (SGLD, SGLD-CV, SVRG-SGLD).
- `experiments/run_journal_experiments.py`: experiment driver (`--exp 1..8` or `all`).
- `experiments/generate_figures.py`, `experiments/generate_hero_overview.py`: regenerate all figures from `results/*.json`.
- `results/`: seed-level JSON archives for every reported number.

## Reproduce

```bash
python -m pip install -r requirements.txt
python experiments/run_journal_experiments.py --exp all   # downloads UCI a9a and Wine Quality via OpenML
python experiments/generate_figures.py
python experiments/generate_hero_overview.py
```

Figures are written to `figures/`. Pre-computed results are included, so figures can be regenerated without rerunning the experiments.
