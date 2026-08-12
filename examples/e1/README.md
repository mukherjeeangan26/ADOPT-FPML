# E1: three-CSTR dynamic reactor

E1 generates six manipulated/feed variables (`CA0`, `T_in`, `v`, and three heat inputs) as 35-sample step signals. The true three-stage plant uses Arrhenius kinetics, reaction order 1.2, and RK4 integration. Its measurements contain two deliberately coupled bias/noise layers. The intentionally mismatched FP model uses a constant lumped kinetic coefficient, reaction order 1.0, and the same three-stage balances.

The four targets are `CA`, `CB`, `CC`, and `T_out`, with RMSE thresholds 5, 2, 3, and 10 respectively.

Run reference inference:

```bash
python examples/e1/run_pretrained.py
```

Regenerate the 500-row dataset and compare it to the bundled reference:

```bash
python examples/e1/generate_data.py --rows 500 --output e1_regenerated.csv
```

Run all four structural configurations (parameter estimation and repeated ML fitting can take several minutes):

```bash
python examples/e1/run_discovery.py
```

The bundled continuous 750-row trajectory preserves a 500-row discovery segment and a strictly held-out 250-row segment. The portable pretrained configuration reproduces the manuscript’s training RMSE values (approximately 2.9, 1.5, 1.4, and 6.4). Its held-out values are documented in `docs/reproducibility.md`.
