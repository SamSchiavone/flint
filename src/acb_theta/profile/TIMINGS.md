# acb_theta timing experiments (branch `theta-optimize`)

Base commit: 657a2839c ("SIMD-based nmod matrix multiplication (#2842)").
Machine: 2-core x86-64 Linux container, gcc -O2 -march=native, FLINT built with
`./configure --disable-static --disable-assert`.

All runs use z = 0 and an exact, exactly symmetric, reduced tau produced by
`acb_siegel_randtest_reduced(tau, state, prec, 2)` with seed (1, 18), followed
by `acb_get_mid` on every entry and copying the upper triangle onto the lower one.
(Without the symmetrisation step, `acb_siegel_reduce` silently fails at g = 7, 8
and `acb_theta_all` returns indeterminate balls.)

Programs: `src/acb_theta/profile/p-all_breakdown.c`, `src/acb_theta/profile/p-all_compare.c`.

## 1. Where the time goes in the original code (single thread)

`p-all_breakdown g prec 1 1`

| g | prec | acb_theta_all(sqr=0) | sum_jet(ord=2) error pass | ql_jet_error x 4^g | ql_exact(all=1) | ql_setup | acb_theta_all(sqr=1) | ql_jet(all=0), theta_{a,0} | acb_theta_sum (all a,b) |
|---|------|------|------|------|------|------|------|------|------|
| 4 | 256  | 0.025 | 0.010 | 0.002 | 0.005 | 0.002 | 0.009 | 0.006 | 0.041 |
| 5 | 256  | 0.564 | 0.514 | 0.014 | 0.038 | 0.033 | 0.032 | 0.041 | 1.458 |
| 6 | 256  | 3.189 | 2.867 | 0.060 | 0.145 | (lower-dim path) | 0.138 | 0.145 | 4.639 |
| 6 | 2000 | 2.193 | 1.841 | 0.074 | 0.300 | 0.056 | 0.261 | 0.226 | - |
| 7 | 256  | 21.57 | 17.31 | 0.301 | 0.631 | (lower-dim path) | 0.342 | 0.288 | - |
| 8 | 256  | > 12 min (killed) | - | - | 3.254 | (lower-dim path) | 1.237 | 1.130 | - |

Times in seconds. The "error pass" columns are the low-precision order-2
derivatives of all 4^g theta functions computed in `acb_theta_ql_jet_fd` only to
propagate the radius of tau (which is nonzero after reduction even for exact
input). The g = 6, prec = 256 row has a different random tau from the 2000-bit row
(pattern 6 6 6 6 6 0, i.e. dimension lowering), hence not monotone in prec.

## 2. Original vs. cheap error bound (`p-all_compare`)

`p-all_compare g prec [threads]`, built against the base commit (original) and the branch (cheap).

| g | prec | original | cheap, 1 thread | cheap, 2 threads | radius original -> cheap |
|---|------|------|------|------|------|
| 4 | 256  | 0.025 | 0.010 | - | 2^-247 -> 2^-247 |
| 5 | 256  | 0.763 | 0.054 | - | 2^-248 -> 2^-228 |
| 6 | 256  | 0.454 | 0.144 | - | 2^-243 -> 2^-216 |
| 6 | 2000 | 2.216 | 0.284 | - | 2^-1990 -> 2^-1964 |
| 7 | 256  | 21.6  | 0.798 | 0.672 | 2^-252 -> 2^-252 |
| 7 | 2000 | -     | 0.956 | - | 2^-1962 |
| 8 | 256  | > 12 min | 3.586 | 2.986 | 2^-209 |

The overlap check passed on all 4^g values in every run where both paths finished.
The looser radius is log2 of c / (2 pi rho^2) (times g^2) from the Cauchy bound;
compensate by requesting ~50 extra bits.

## 3. Threading alone (parallel `acb_theta_sum` over a in {0,1}^g)

| what | g | prec | 1 thread | 2 threads | speedup |
|---|---|---|---|---|---|
| acb_theta_sum (all a,b) | 5 | 256 | 1.423 | 0.764 | 1.86x |
| ql_exact (all=1) | 5 | 256 | 0.029 | 0.022 | 1.3x |
| ql_exact (all=1) | 6 | 2000 | 0.232 | 0.208 | 1.1x |
| ql_exact (all=1) | 7 | 256 | 0.631 | 0.475 | 1.3x |

Plain summation scales almost linearly; the fast algorithm does not yet, because
`ql_perform_steps`, `ql_exact_lower_dim` and `ql_setup` are still serial.

## 4. Option 2: reduce (z, tau) with guard bits, then cheap error bound

`acb_theta_jet` now calls `acb_theta_reduce_tau` / `acb_theta_reduce_z` at
`prec + ACB_THETA_LOW_PREC + 8 g` bits, so the radii of the reduced input are
far below 2^(-prec) and the crude Cauchy bound no longer costs output precision.
Best of three runs, single thread. This is the configuration now on the branch
(no switch: the cheap bound is always used for ord = 0).

| g | prec | original time | option 2 time | speedup | radius original | radius option 2 |
|---|------|------|------|------|------|------|
| 4 | 256  | 0.025 | 0.012 | 2x    | 2^-247  | 2^-247  |
| 5 | 256  | 0.763 | 0.038 | 20x   | 2^-248  | 2^-248  |
| 6 | 256  | 0.454 | 0.163 | 2.8x  | 2^-243  | 2^-243  |
| 6 | 2000 | 2.216 | 0.287 | 7.7x  | 2^-1990 | 2^-1990 |
| 7 | 256  | 21.6  | 0.733 | 29x   | 2^-252  | 2^-252  |
| 7 | 2000 | -     | 0.863 | -     | -       | 2^-1986 |
| 8 | 256  | > 12 min | 3.67 | > 200x | -     | 2^-246  |

Cross-check (same tau, cheap vs tight bound in one process): all 4^g values
overlap at g = 4, 5, 6, with identical radii.
