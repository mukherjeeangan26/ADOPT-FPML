# Algorithm and terminology

## Reduced superstructure

ADOPT-FPML represents route choices with binary information-flow decisions. The public API exposes their four feasible interaction patterns as `Structure`: parallel, Series I, Series II, and integrated. Every requested target must be supplied by at least the standalone FP route or an ML/hybrid route.

The four configurations are distinguished by the FP-to-ML and ML-to-FP interaction decisions:

| Configuration | FP → ML | ML → FP |
|---|---:|---:|
| Parallel | 0 | 0 |
| Series I | 1 | 0 |
| Series II | 0 | 1 |
| Integrated | 1 | 1 |

Series I/integrated candidates must select at least one FP-derived synthetic input. Series II/integrated candidates also require estimated row-wise FP parameter targets.

## Stage-wise branch and prune

Let `J(b)` be unresolved outputs and `U(b)` the available measured and retained synthetic inputs at stage `b`.

1. FP screening creates the initially resolved set and `J(1)`.
2. A selector ranks `U(b)` against `J(b)`.
3. Each configuration is a branch. Inputs are added in rank order.
4. Every candidate is retrained and assigned local AICc.
5. The default greedy policy ends a branch when local AICc first fails to improve.
6. Among candidates resolving at least one new output, the stage winner has minimum local AICc.
7. The winner is retained if its relative global-AICc increase is strictly below `tau_AIC`.
8. Relevant, nonredundant predicted variables are added to `U(b+1)`; resolved outputs are removed from `J(b+1)`.

The implementation evaluates AICc as

```text
n log(SSE / n) + 2 k + 2 k (k + 1) / (n - k - 1)
```

where `n` is the number of scalar residuals and `k` is the ML parameter count. A common Gaussian constant is omitted because it does not change candidate ordering. AICc is infinite when `n <= k + 1`.

## Local versus global selection

Local AICc chooses the best block within a stage. Global AICc then decides whether that locally optimal block is acceptable in the accumulated model. This separation is important: using global AICc to pick the within-stage winner changes the proposed algorithm. The public implementation follows the local-selection rule used in the manuscript and final E2 code.

## Termination

Search stops when one of the following occurs:

- all outputs meet their accuracy rules;
- no branch resolves a new output;
- the selected branch violates the global-AICc continuation tolerance;
- the configured maximum stage count is reached;
- there are no candidate inputs.

Search history records skipped structures, failed candidates, selected inputs, local/global objectives, resolved outputs, and termination reasons.

