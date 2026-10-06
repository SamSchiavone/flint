/*
    Copyright (C) 2026 Sam Schiavone

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

/* Timing + consistency check for acb_theta_all on all 4^g theta constants.

   Runs acb_theta_all(sqr = 0) at z = 0 on an exact reduced tau, then on the
   same tau with a radius of 2^(-2 prec) added to every entry, and checks that
   all 4^g output balls overlap. Build it against the base commit and against
   the theta-optimize branch to compare the original and optimized code.

   usage: p-all_compare g prec [threads] */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "acb.h"
#include "acb_mat.h"
#include "acb_theta.h"

static double
now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char * argv[])
{
    slong g, prec, n, j, k;
    flint_rand_t state;
    acb_mat_t tau, tau_r;
    acb_ptr z, th1, th2;
    mag_t r;
    double t0;
    int ok = 1;

    if (argc < 3)
    {
        printf("usage: %s g prec [threads]\n", argv[0]);
        return 1;
    }
    g = atol(argv[1]);
    prec = atol(argv[2]);
    if (argc > 3)
    {
        flint_set_num_threads(atoi(argv[3]));
    }
    n = 1 << g;

    flint_rand_init(state);
    flint_rand_set_seed(state, 1, 18);
    acb_mat_init(tau, g, g);
    acb_mat_init(tau_r, g, g);
    z = _acb_vec_init(g);
    th1 = _acb_vec_init(n * n);
    th2 = _acb_vec_init(n * n);
    mag_init(r);

    /* Exact, exactly symmetric, reduced tau */
    acb_siegel_randtest_reduced(tau, state, prec, 2);
    for (j = 0; j < g; j++)
    {
        for (k = 0; k < g; k++)
        {
            acb_get_mid(acb_mat_entry(tau, j, k), acb_mat_entry(tau, j, k));
            if (k < j)
            {
                acb_set(acb_mat_entry(tau, j, k), acb_mat_entry(tau, k, j));
            }
        }
    }

    t0 = now();
    acb_theta_all(th1, z, tau, 0, prec);
    printf("g=%ld prec=%ld  exact tau:   %.3f s, rad 2^%ld\n", (long) g, (long) prec,
        now() - t0, (long) mag_get_d_log2_approx(arb_radref(acb_realref(&th1[1]))));

    acb_mat_set(tau_r, tau);
    mag_set_ui_2exp_si(r, 1, -2 * prec);
    for (j = 0; j < g; j++)
    {
        for (k = 0; k < g; k++)
        {
            acb_add_error_mag(acb_mat_entry(tau_r, j, k), r);
        }
    }

    t0 = now();
    acb_theta_all(th2, z, tau_r, 0, prec);
    printf("                 inexact tau: %.3f s, rad 2^%ld\n",
        now() - t0, (long) mag_get_d_log2_approx(arb_radref(acb_realref(&th2[1]))));

    for (j = 0; j < n * n; j++)
    {
        if (!acb_overlaps(&th1[j], &th2[j]))
        {
            ok = 0;
        }
    }
    printf("all %ld values overlap: %s\n", (long) (n * n), ok ? "yes" : "NO");

    acb_mat_clear(tau);
    acb_mat_clear(tau_r);
    _acb_vec_clear(z, g);
    _acb_vec_clear(th1, n * n);
    _acb_vec_clear(th2, n * n);
    mag_clear(r);
    flint_rand_clear(state);
    flint_cleanup();
    return ok ? 0 : 1;
}
