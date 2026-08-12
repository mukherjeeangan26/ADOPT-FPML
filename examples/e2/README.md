# E2: cooler and isothermal flash

E2 uses the E1 outlet trajectory (`CA`, `CB`, `CC`, `T_out`, and `v`) as its feed. Appendix B specifies a target-vapor-fraction cooler, temperature/pressure-dependent equilibrium ratios, enthalpy calculations, and an isothermal flash. Flows, phase compositions, and temperatures receive controlled measurement noise.

The original E2 working archive referenced a generator module that was not present. `generate_e2_data` reconstructs it from the Appendix B equations and validates it against the retained authoritative 500-row dataset. Seed 42 and the archived random draw order are fixed for exact reproducibility within root-solver tolerance.

Run reference training or held-out inference:

```bash
python examples/e2/run_pretrained.py
python examples/e2/run_pretrained.py --split test
```

Regenerate E2 from the E1 training feed:

```bash
python examples/e2/generate_data.py --output e2_regenerated.csv
```

Run all four discovery configurations:

```bash
python examples/e2/run_discovery.py
```

The example supplies `E2ParameterTargetGenerator`, which performs bounded row-wise SciPy estimation of `UA`, `K_A`, `K_B`, and `K_C` once per search stage. Other systems should supply their own `ParameterTargetGenerator`, as explained in `docs/defining-a-system.md`.
