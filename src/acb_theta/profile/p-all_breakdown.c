/*
    Copyright (C) 2026 Sam Schiavone

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

/* Time breakdown of acb_theta_all for theta constants (z = 0) at an exact,
   reduced tau: full call, sqr=1 variant, theta_{a,0} only, and the phases of
   the all=1 path (error-propagation derivatives, ql_setup, duplication steps).

   usage: p-all_breakdown g prec [seed] [threads] [fast]
     threads: value for flint_set_num_threads (default 1)
     fast:    any 5th argument skips the slow parts (full acb_theta_all,
              the order-2 derivative pass and plain summation).
   On the theta-optimize branch the "sum_jet(ord=2) for err" line measures
   the derivative pass that the original code ran inside acb_theta_all and
   that the branch no longer performs for ord = 0. */

#include <stdlib.h>
#include <stdio.h>
#include "profiler.h"
#include "acb.h"
#include "acb_mat.h"
#include "arb_mat.h"
#include "acb_theta.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char * argv[])
{
    slong g, prec, n, nbjet2, j, k, lp;
    ulong seed = 1;
    flint_rand_t state;
    acb_mat_t tau, tau2;
    acb_ptr z, th, th_a0, dth;
    arb_ptr err, distances;
    slong * pattern;
    acb_theta_ctx_tau_t ctx_tau;
    acb_theta_ctx_z_struct * vec;
    acb_ptr rts, rts_all, t;
    slong guard, easy_steps[1];
    double t0, t_all, t_sqr, t_a0, t_dth, t_err, t_setup, t_qlexact, t_sum;
    int res;

    if (argc < 3) { printf("usage: %s g prec [seed]\n", argv[0]); return 1; }
    g = atol(argv[1]); prec = atol(argv[2]);
    if (argc > 3) seed = strtoul(argv[3], NULL, 10);
    if (argc > 4) flint_set_num_threads(atoi(argv[4]));
    int fast = (argc > 5);
    n = 1 << g;
    setvbuf(stdout, NULL, _IONBF, 0);
    nbjet2 = acb_theta_jet_nb(2, g);

    flint_rand_init(state);
    flint_rand_set_seed(state, seed, seed + 17);
    acb_mat_init(tau, g, g);
    acb_mat_init(tau2, g, g);
    z = _acb_vec_init(g);
    th = _acb_vec_init(n * n);
    th_a0 = _acb_vec_init(n);
    dth = _acb_vec_init(n * n * nbjet2);
    err = _arb_vec_init(n * n);
    distances = _arb_vec_init(n);
    pattern = flint_malloc(g * sizeof(slong));
    rts = NULL; rts_all = NULL;
    t = _acb_vec_init(g);

    /* exact reduced tau */
    acb_siegel_randtest_reduced(tau, state, prec, 2);
    for (j = 0; j < g; j++)
        for (k = 0; k < g; k++)
            { acb_get_mid(acb_mat_entry(tau, j, k), acb_mat_entry(tau, j, k)); if (k < j) acb_set(acb_mat_entry(tau, j, k), acb_mat_entry(tau, k, j)); }

    printf("g = %ld, prec = %ld\n", (long) g, (long) prec);
    acb_theta_ql_nb_steps(pattern, tau, 1, prec);
    printf("pattern:");
    for (j = 0; j < g; j++) printf(" %ld", (long) pattern[j]);
    printf("\n");

    /* 1. full acb_theta_all, all 4^g theta constants */
    if (!fast) {
    t0 = now();
    acb_theta_all(th, z, tau, 0, prec);
    t_all = now() - t0;
    printf("acb_theta_all(sqr=0)            : %8.3f s   (4^g = %ld values, rad ~ 2^%ld)\n",
        t_all, (long) (n * n), (long) mag_get_d_log2_approx(arb_radref(acb_realref(&th[0]))));
    }

    /* 2. acb_theta_all with sqr = 1 */
    t0 = now();
    acb_theta_all(th, z, tau, 1, prec);
    t_sqr = now() - t0;
    printf("acb_theta_all(sqr=1)            : %8.3f s\n", t_sqr);

    /* 3. only theta_{a,0}(0,tau), all a (internal ql_jet with all=0) */
    t0 = now();
    acb_theta_ql_jet(th_a0, z, 1, tau, 0, 0, prec);
    t_a0 = now() - t0;
    printf("acb_theta_ql_jet(all=0) [a,0]   : %8.3f s   (2^g values)\n", t_a0);

    /* --- Breakdown of the all=1, sqr=0 path (mirrors acb_theta_ql_jet_fd) --- */
    printf("--- breakdown of all=1 path ---\n");

    /* 3a. low-precision order-2 derivatives for error propagation */
    if (!fast) {
    acb_theta_ctx_tau_init(ctx_tau, 0, g);
    vec = acb_theta_ctx_z_vec_init(1, g);
    t0 = now();
    for (lp = 8; lp <= 64; lp *= 2)
    {
        acb_theta_ctx_tau_set(ctx_tau, tau, lp + ACB_THETA_LOW_PREC);
        acb_theta_ctx_z_set(&vec[0], z, ctx_tau, lp + ACB_THETA_LOW_PREC);
        acb_theta_sum_jet(dth, vec, 1, ctx_tau, 2, 1, 1, lp);
        if (_acb_vec_is_finite(dth, n * n * nbjet2)) break;
    }
    t_dth = now() - t0;
    printf("  sum_jet(ord=2, lp=%ld) for err  : %8.3f s   [skippable when tau exact]\n", (long) lp, t_dth);

    t0 = now();
    for (k = 0; k < n * n; k++)
        acb_theta_ql_jet_error(err + k, z, tau, dth + k * nbjet2, 0, ACB_THETA_LOW_PREC);
    t_err = now() - t0;
    printf("  ql_jet_error x 4^g             : %8.3f s   [skippable when tau exact]\n", t_err);
    acb_theta_ctx_tau_clear(ctx_tau);
    acb_theta_ctx_z_vec_clear(vec, 1);
    }

    /* 3b. ql_exact total, and ql_setup part of it */
    acb_theta_ql_nb_steps(pattern, tau, 0, prec);
    t0 = now();
    acb_theta_ql_exact(th, z, 1, tau, pattern, 1, 0, prec);
    t_qlexact = now() - t0;
    printf("  ql_exact (all=1)               : %8.3f s\n", t_qlexact);

    if (pattern[g - 1] > 0)
    {
        slong nb_steps = pattern[g - 1];
        rts = _acb_vec_init(3 * n * nb_steps);
        rts_all = _acb_vec_init(n * n);
        acb_theta_eld_distances(distances, z, 1, tau, ACB_THETA_LOW_PREC);
        t0 = now();
        res = acb_theta_ql_setup(rts, rts_all, t, &guard, easy_steps, z, 1, tau,
            distances, nb_steps, 1, prec);
        t_setup = now() - t0;
        printf("    of which ql_setup            : %8.3f s   (nb_steps=%ld, easy=%ld, guard=%ld, ok=%d)\n",
            t_setup, (long) nb_steps, (long) easy_steps[0], (long) guard, res);
        printf("    remainder (init sum + steps) : %8.3f s\n", t_qlexact - t_setup);
        _acb_vec_clear(rts, 3 * n * nb_steps);
        _acb_vec_clear(rts_all, n * n);
    }

    /* 4. plain summation at tau for comparison (all a, all b) */
    if (prec <= 512 && !fast) {
    acb_theta_ctx_tau_init(ctx_tau, 1, g);
    vec = acb_theta_ctx_z_vec_init(1, g);
    acb_theta_ctx_tau_set(ctx_tau, tau, prec + ACB_THETA_LOW_PREC);
    acb_theta_ctx_z_set(&vec[0], z, ctx_tau, prec + ACB_THETA_LOW_PREC);
    _arb_vec_zero(distances, n);
    t0 = now();
    acb_theta_sum(th, vec, 1, ctx_tau, distances, 1, 1, 1, prec);
    t_sum = now() - t0;
    printf("acb_theta_sum (all a,b) at tau  : %8.3f s\n", t_sum);
    acb_theta_ctx_tau_clear(ctx_tau);
    acb_theta_ctx_z_vec_clear(vec, 1);
    }

    acb_mat_clear(tau); acb_mat_clear(tau2);
    _acb_vec_clear(z, g); _acb_vec_clear(th, n * n); _acb_vec_clear(th_a0, n);
    _acb_vec_clear(dth, n * n * nbjet2); _arb_vec_clear(err, n * n);
    _arb_vec_clear(distances, n); flint_free(pattern); _acb_vec_clear(t, g);
    flint_rand_clear(state);
    flint_cleanup();
    return 0;
}
